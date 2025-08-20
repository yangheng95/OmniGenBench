"""
Utilities for the Transcription Factor Binding (TFB) demo notebook.

This module centralizes reusable logic so the notebook stays concise:
- DeepSEA-style dataset wrapper around OmniDataset
- Tokenizer and model loader for multi-label classification
- Dataset and DataLoader builders
- Training, evaluation, and simple inference helpers
"""
import zipfile
from typing import Optional, List, Dict, Any

import os
import json
import requests
import torch
import torch.nn as nn
import findfile

from transformers import AutoTokenizer, AutoModel

from omnigenbench import (
    OmniDataset,
    ClassificationMetric,
    AccelerateTrainer,
    ModelHub,
    OmniModelForMultiLabelSequenceClassification,
    OmniModel,
)




def download_deepsea_dataset(local_dir):
    if not findfile.find_cwd_dir(local_dir, disable_alert=True):
        os.makedirs(local_dir, exist_ok=True)
    url_to_download = "https://huggingface.co/datasets/yangheng/deepsea_tfb_prediction/resolve/main/deepsea_tfb_prediction.zip"
    zip_path = os.path.join(local_dir, "deepsea_tfb_prediction.zip")
    if not os.path.exists(zip_path):
        print(f"Downloading deepsea_tfb_prediction.zip from {url_to_download}...")
        response = requests.get(url_to_download, stream=True)
        response.raise_for_status()

        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {zip_path}")

    # Unzip the dataset if the zip file exists
    ZIP_DATASET = findfile.find_cwd_file("deepsea_tfb_prediction.zip")
    if ZIP_DATASET:
        with zipfile.ZipFile(ZIP_DATASET, 'r') as zip_ref:
            zip_ref.extractall(local_dir)
        print(f"Extracted deepsea_tfb_prediction.zip into {local_dir}")
        os.remove(ZIP_DATASET)
    else:
        print("deepsea_tfb_prediction.zip not found. Skipping extraction.")


class DeepSEADataset(OmniDataset):
    """
    A dataset for the DeepSEA task that converts DNA sequences to tokenized inputs.
    Accepts JSONL where each line contains a `sequence` field and `label(s)`.
    """

    def __init__(
        self,
        data_source: str,
        tokenizer,
        max_length: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(data_source, tokenizer, max_length, **kwargs)
        self.label_indices = None
        for key, value in kwargs.items():
            self.metadata[key] = value

    def prepare_input(self, instance: Dict[str, Any], **kwargs) -> Dict[str, torch.Tensor]:
        def truncate(seq: str, windowsize: Optional[int]) -> str:
            if windowsize is None:
                return seq
            if len(seq) == windowsize:
                return seq
            if len(seq) > windowsize:
                left = (len(seq) - windowsize) // 2
                return seq[left:left + windowsize]
            return seq + ("N" * (windowsize - len(seq)))

        sequence = instance['sequence']
        labels = instance.get('label', None) if 'label' in instance else instance.get('labels', None)

        tokenized_inputs = self.tokenizer(
            truncate(sequence, self.max_length),
            padding="do_not_pad",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            **kwargs,
        )

        labels_tensor = torch.tensor(labels, dtype=torch.float32) if labels is not None else None
        if labels_tensor is not None and hasattr(self, 'label_indices') and self.label_indices is not None:
            labels_tensor = labels_tensor[torch.tensor(self.label_indices, dtype=torch.long)]
        tokenized_inputs["labels"] = labels_tensor

        for col in tokenized_inputs:
            if isinstance(tokenized_inputs[col], torch.Tensor) and tokenized_inputs[col].ndim > 1:
                tokenized_inputs[col] = tokenized_inputs[col].squeeze(0)

        return tokenized_inputs


def load_tokenizer_and_model(
    model_name_or_path: str,
    num_labels: int,
    threshold: float = 0.5,
    device: Optional[torch.device] = None,
):
    """
    Load tokenizer and OmniGenome-based multi-label classification model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    base_model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)

    model = OmniModelForMultiLabelSequenceClassification(
        base_model,
        tokenizer,
        num_labels=num_labels,
        threshold=threshold,
    )
    if device is not None:
        model = model.to(device).to(torch.float32)

    return tokenizer, model


def build_datasets(
    tokenizer,
    train_file: str,
    test_file: str,
    valid_file: Optional[str],
    max_length: int,
    max_examples: Optional[int],
    label_indices: Optional[List[int]] = None,
):
    """
    Create DeepSEADataset instances for train/valid/test.
    """
    def make_ds(path: str) -> DeepSEADataset:
        ds = DeepSEADataset(
            data_source=path,
            tokenizer=tokenizer,
            max_length=max_length,
            max_examples=max_examples,
            force_padding=False,
        )
        # attach label indices for internal selection
        ds.label_indices = label_indices
        return ds

    train_set = make_ds(train_file)
    test_set = make_ds(test_file)
    valid_set = make_ds(valid_file) if (valid_file and os.path.exists(valid_file)) else None
    return train_set, valid_set, test_set


def create_dataloaders(
    train_set: torch.utils.data.Dataset,
    valid_set: Optional[torch.utils.data.Dataset],
    test_set: Optional[torch.utils.data.Dataset],
    batch_size: int,
):
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size) if test_set is not None else None
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size) if valid_set is not None else None
    return train_loader, valid_loader, test_loader


def run_finetuning(
    model: torch.nn.Module,
    train_loader,
    valid_loader,
    test_loader,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    device: torch.device,
    save_dir: str = "tfb_model",
):
    """
    Train the model with AccelerateTrainer and save to `save_dir`.
    Returns the trainer and the best metrics.
    """
    metric_fn = [ClassificationMetric(ignore_y=-100).roc_auc_score]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    trainer = AccelerateTrainer(
        model=model,
        train_loader=train_loader,
        eval_loader=valid_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        epochs=epochs,
        compute_metrics=metric_fn,
        patience=patience,
        device=device,
    )

    metrics_best = None
    if not os.path.exists(save_dir):
        metrics_best = trainer.train()
        trainer.save_model(save_dir)
    else:
        metrics_best = {"info": f"Found existing '{save_dir}'. Skipped training."}

    return trainer, metrics_best



def run_inference(
    model_dir: str,
    tokenizer,
    sample_sequence: str,
    max_length: int,
    device: torch.device,
):
    """
    Load a saved model via ModelHub and run a single-sequence inference.
    Returns a dict containing predictions and probabilities when available.
    """
    model = ModelHub.load(model_dir)
    model.to(device)

    inputs = tokenizer(sample_sequence, return_tensors="pt", max_length=max_length, truncation=True)
    with torch.no_grad():
        outputs = model.inference({k: v.to(device) for k, v in inputs.items()})
    return outputs


class _MaskedGlobalMaxPool1d(nn.Module):
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if attention_mask is None:
            return x.max(dim=1).values
        masked_x = x.masked_fill(attention_mask.unsqueeze(-1).eq(0), float("-inf"))
        return masked_x.max(dim=1).values


class OmniCNNBaseline(OmniModel):
    """TextCNN-style baseline for DeepSEA-style multilabel classification.

    Implements forward/predict/inference for trainer compatibility.
    """

    def __init__(self, tokenizer, *args, **kwargs):
        embed_dim = kwargs.pop("embed_dim", 128)
        num_filters = kwargs.pop("num_filters", 128)
        kernel_sizes = kwargs.pop("kernel_sizes", (3, 5, 7))
        dropout = kwargs.pop("dropout", 0.1)
        num_labels = kwargs.pop("num_labels")

        # Minimal config stub
        class Cfg:
            pass

        cfg = Cfg()
        cfg.hidden_size = embed_dim
        cfg.num_labels = num_labels
        cfg.label2id = {str(i): i for i in range(num_labels)}
        cfg.id2label = {i: str(i) for i in range(num_labels)}
        cfg.name_or_path = "CNNBaseline"
        cfg.model_type = "cnn"
        cfg.architectures = ["CNNBaseline"]
        cfg.pad_token_id = getattr(tokenizer, "pad_token_id", -100)
        # Persist hyperparameters for reload
        cfg.embed_dim = embed_dim
        cfg.num_filters = num_filters
        cfg.kernel_sizes = list(kernel_sizes)
        cfg.dropout = dropout

        class _Stub(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                # buffer tracks device/dtype so OmniModel can query .device/.dtype
                self.register_buffer("_dev_tracker", torch.empty(0))

            @property
            def device(self):
                return self._dev_tracker.device

            @property
            def dtype(self):
                return self._dev_tracker.dtype

        super().__init__(_Stub(cfg), tokenizer, num_labels=num_labels, *args, **kwargs)

        vocab_size = getattr(self.tokenizer, "vocab_size", None) or len(self.tokenizer.get_vocab())
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=self.pad_token_id)
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(embed_dim, num_filters, k, padding=k // 2), nn.ReLU()) for k in kernel_sizes
        ])
        self.pool = _MaskedGlobalMaxPool1d()
        self.classifier = nn.Linear(num_filters * len(kernel_sizes), num_labels)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.loss_fn = nn.BCELoss()

    # ---------------- Serialization -----------------
    def _build_config_dict(self):
        return {
            "model_type": self.config.model_type,
            "architectures": self.config.architectures,
            "num_labels": self.config.num_labels,
            "label2id": self.config.label2id,
            "id2label": self.config.id2label,
            "pad_token_id": self.config.pad_token_id,
            "hidden_size": self.config.hidden_size,
            "embed_dim": getattr(self.config, "embed_dim", None),
            "num_filters": getattr(self.config, "num_filters", None),
            "kernel_sizes": getattr(self.config, "kernel_sizes", None),
            "dropout": getattr(self.config, "dropout", None),
            "vocab_size": self.embedding.num_embeddings,
            "model_cls": self.__class__.__name__,
            "library_name": "OMNIGENBENCH",
        }

    def save_pretrained(self, save_directory: str, overwrite: bool = True):
        os.makedirs(save_directory, exist_ok=True)
        # Save config
        with open(os.path.join(save_directory, "config.json"), "w", encoding="utf8") as f:
            json.dump(self._build_config_dict(), f, ensure_ascii=False, indent=2)
        # Save tokenizer
        if hasattr(self.tokenizer, "save_pretrained"):
            self.tokenizer.save_pretrained(save_directory)
        else:
            with open(os.path.join(save_directory, "tokenizer.bin"), "wb") as f:
                torch.save(self.tokenizer, f)
        # Save state dict
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        # Metadata
        metadata = getattr(self, "metadata", {})
        metadata["model_cls"] = self.__class__.__name__
        metadata["library_name"] = metadata.get("library_name", "OMNIGENBENCH")
        with open(os.path.join(save_directory, "metadata.json"), "w", encoding="utf8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    @classmethod
    def from_pretrained(cls, save_directory: str, tokenizer=None, map_location=None, **kwargs):
        # Load config
        with open(os.path.join(save_directory, "config.json"), "r", encoding="utf8") as f:
            cfg_dict = json.load(f)
        if tokenizer is None:
            # Try standard huggingface style first
            if os.path.exists(os.path.join(save_directory, "tokenizer_config.json")) or os.path.exists(os.path.join(save_directory, "vocab.json")):
                try:
                    from transformers import AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained(save_directory)
                except Exception:
                    tokenizer = None
            if tokenizer is None and os.path.exists(os.path.join(save_directory, "tokenizer.bin")):
                with open(os.path.join(save_directory, "tokenizer.bin"), "rb") as f:
                    tokenizer = torch.load(f, map_location=map_location)
            if tokenizer is None:
                raise ValueError("Tokenizer could not be loaded; please provide one.")
        model = cls(
            tokenizer,
            embed_dim=cfg_dict.get("embed_dim", cfg_dict.get("hidden_size")),
            num_filters=cfg_dict.get("num_filters", 128),
            kernel_sizes=tuple(cfg_dict.get("kernel_sizes", (3, 5, 7))),
            dropout=cfg_dict.get("dropout", 0.1),
            num_labels=cfg_dict["num_labels"],
            label2id=cfg_dict.get("label2id"),
            **kwargs,
        )
        # Load weights
        state_dict = torch.load(os.path.join(save_directory, "pytorch_model.bin"), map_location=map_location or "cpu")
        model.load_state_dict(state_dict, strict=False)
        # Load metadata if exists
        meta_path = os.path.join(save_directory, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf8") as f:
                model.metadata = json.load(f)
        return model

    def forward(self, **inputs) -> Dict[str, Any]:
        labels = inputs.pop("labels", None)
        x = self.embedding(inputs["input_ids"])  # [B,L,E]
        x = self.dropout(x)
        feats = [m(x.transpose(1, 2)) for m in self.convs]  # list of [B,C,L]
        feats = torch.cat(feats, dim=1).transpose(1, 2)  # [B,L,C*]
        pooled = self.pool(feats, inputs.get("attention_mask"))  # [B,C*]
        pooled = self.dropout(pooled)
        logits = self.sigmoid(self.classifier(pooled))  # [B,num_labels]
        return {"logits": logits, "last_hidden_state": pooled, "labels": labels}

    def predict(self, sequence_or_inputs, **kwargs):
        out = self._forward_from_raw_input(sequence_or_inputs, **kwargs)
        return {"predictions": out["logits"], "logits": out["logits"], "last_hidden_state": out["last_hidden_state"]}

    def inference(self, sequence_or_inputs, threshold: float = 0.5, **kwargs):
        out = self._forward_from_raw_input(sequence_or_inputs, **kwargs)
        logits = out["logits"]
        preds = (logits >= threshold).to(torch.int)
        if not isinstance(sequence_or_inputs, list):
            return {"predictions": preds[0].cpu(), "logits": logits[0].cpu(), "confidence": torch.max(logits[0]).cpu(), "last_hidden_state": out["last_hidden_state"][0].cpu()}
        return {"predictions": preds.cpu(), "logits": logits.cpu(), "confidence": torch.max(logits, dim=-1)[0].cpu(), "last_hidden_state": out["last_hidden_state"]}

    def loss_function(self, logits, labels):
        return self.loss_fn(logits.view(-1), labels.view(-1).to(torch.float32))


class OmniRNNBaseline(OmniModel):
    """BiLSTM baseline for DeepSEA-style multilabel classification."""

    def __init__(self, tokenizer, *args, **kwargs):
        embed_dim = kwargs.pop("embed_dim", 128)
        hidden_dim = kwargs.pop("hidden_dim", 256)
        num_layers = kwargs.pop("num_layers", 1)
        bidirectional = kwargs.pop("bidirectional", True)
        dropout = kwargs.pop("dropout", 0.1)
        num_labels = kwargs.pop("num_labels")

        class Cfg:
            pass

        cfg = Cfg()
        cfg.hidden_size = hidden_dim * (2 if bidirectional else 1)
        cfg.num_labels = num_labels
        cfg.label2id = {str(i): i for i in range(num_labels)}
        cfg.id2label = {i: str(i) for i in range(num_labels)}
        cfg.name_or_path = "RNNBaseline"
        cfg.model_type = "rnn"
        cfg.architectures = ["RNNBaseline"]
        cfg.pad_token_id = getattr(tokenizer, "pad_token_id", -100)
        # Persist hyperparameters for reload
        cfg.embed_dim = embed_dim
        cfg.hidden_dim = hidden_dim
        cfg.num_layers = num_layers
        cfg.bidirectional = bidirectional
        cfg.dropout = dropout

        class _Stub(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.register_buffer("_dev_tracker", torch.empty(0))

            @property
            def device(self):
                return self._dev_tracker.device

            @property
            def dtype(self):
                return self._dev_tracker.dtype

        super().__init__(_Stub(cfg), tokenizer, num_labels=num_labels, *args, **kwargs)

        vocab_size = getattr(self.tokenizer, "vocab_size", None) or len(self.tokenizer.get_vocab())
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=self.pad_token_id)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.0 if num_layers == 1 else dropout, bidirectional=bidirectional)
        self.classifier = nn.Linear(hidden_dim * (2 if bidirectional else 1), num_labels)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.loss_fn = nn.BCELoss()

    # ---------------- Serialization -----------------
    def _build_config_dict(self):
        return {
            "model_type": self.config.model_type,
            "architectures": self.config.architectures,
            "num_labels": self.config.num_labels,
            "label2id": self.config.label2id,
            "id2label": self.config.id2label,
            "pad_token_id": self.config.pad_token_id,
            "hidden_size": self.config.hidden_size,
            "embed_dim": getattr(self.config, "embed_dim", None),
            "hidden_dim": getattr(self.config, "hidden_dim", None),
            "num_layers": getattr(self.config, "num_layers", None),
            "bidirectional": getattr(self.config, "bidirectional", None),
            "dropout": getattr(self.config, "dropout", None),
            "vocab_size": self.embedding.num_embeddings,
            "model_cls": self.__class__.__name__,
            "library_name": "OMNIGENBENCH",
        }

    def save_pretrained(self, save_directory: str, overwrite: bool = True):
        os.makedirs(save_directory, exist_ok=True)
        with open(os.path.join(save_directory, "config.json"), "w", encoding="utf8") as f:
            json.dump(self._build_config_dict(), f, ensure_ascii=False, indent=2)
        if hasattr(self.tokenizer, "save_pretrained"):
            self.tokenizer.save_pretrained(save_directory)
        else:
            with open(os.path.join(save_directory, "tokenizer.bin"), "wb") as f:
                torch.save(self.tokenizer, f)
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        metadata = getattr(self, "metadata", {})
        metadata["model_cls"] = self.__class__.__name__
        metadata["library_name"] = metadata.get("library_name", "OMNIGENBENCH")
        with open(os.path.join(save_directory, "metadata.json"), "w", encoding="utf8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    @classmethod
    def from_pretrained(cls, save_directory: str, tokenizer=None, map_location=None, **kwargs):
        with open(os.path.join(save_directory, "config.json"), "r", encoding="utf8") as f:
            cfg_dict = json.load(f)
        if tokenizer is None:
            if os.path.exists(os.path.join(save_directory, "tokenizer_config.json")) or os.path.exists(os.path.join(save_directory, "vocab.json")):
                try:
                    from transformers import AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained(save_directory)
                except Exception:
                    tokenizer = None
            if tokenizer is None and os.path.exists(os.path.join(save_directory, "tokenizer.bin")):
                with open(os.path.join(save_directory, "tokenizer.bin"), "rb") as f:
                    tokenizer = torch.load(f, map_location=map_location)
            if tokenizer is None:
                raise ValueError("Tokenizer could not be loaded; please provide one.")
        model = cls(
            tokenizer,
            embed_dim=cfg_dict.get("embed_dim", cfg_dict.get("hidden_size")),
            hidden_dim=cfg_dict.get("hidden_dim", 256),
            num_layers=cfg_dict.get("num_layers", 1),
            bidirectional=cfg_dict.get("bidirectional", True),
            dropout=cfg_dict.get("dropout", 0.1),
            num_labels=cfg_dict["num_labels"],
            label2id=cfg_dict.get("label2id"),
            **kwargs,
        )
        state_dict = torch.load(os.path.join(save_directory, "pytorch_model.bin"), map_location=map_location or "cpu")
        model.load_state_dict(state_dict, strict=False)
        meta_path = os.path.join(save_directory, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf8") as f:
                model.metadata = json.load(f)
        return model

    def _mask_mean_pool(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor]):
        if attention_mask is None:
            return x.mean(dim=1)
        mask = attention_mask.unsqueeze(-1).to(x.dtype)
        return (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-6)

    def forward(self, **inputs) -> Dict[str, Any]:
        labels = inputs.pop("labels", None)
        x = self.embedding(inputs["input_ids"])  # [B,L,E]
        x = self.dropout(x)
        out, _ = self.lstm(x)
        pooled = self._mask_mean_pool(out, inputs.get("attention_mask"))
        pooled = self.dropout(pooled)
        logits = self.sigmoid(self.classifier(pooled))
        return {"logits": logits, "last_hidden_state": pooled, "labels": labels}

    def predict(self, sequence_or_inputs, **kwargs):
        out = self._forward_from_raw_input(sequence_or_inputs, **kwargs)
        return {"predictions": out["logits"], "logits": out["logits"], "last_hidden_state": out["last_hidden_state"]}

    def inference(self, sequence_or_inputs, threshold: float = 0.5, **kwargs):
        out = self._forward_from_raw_input(sequence_or_inputs, **kwargs)
        logits = out["logits"]
        preds = (logits >= threshold).to(torch.int)
        if not isinstance(sequence_or_inputs, list):
            return {"predictions": preds[0].cpu(), "logits": logits[0].cpu(), "confidence": torch.max(logits[0]).cpu(), "last_hidden_state": out["last_hidden_state"][0].cpu()}
        return {"predictions": preds.cpu(), "logits": logits.cpu(), "confidence": torch.max(logits, dim=-1)[0].cpu(), "last_hidden_state": out["last_hidden_state"]}

    def loss_function(self, logits, labels):
        return self.loss_fn(logits.view(-1), labels.view(-1).to(torch.float32))


__all__ = [
    "DeepSEADataset",
    "load_tokenizer_and_model",
    "build_datasets",
    "create_dataloaders",
    "run_finetuning",
    "run_inference",
    "OmniCNNBaseline",
    "OmniRNNBaseline",
]
