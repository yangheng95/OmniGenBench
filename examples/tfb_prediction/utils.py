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

import findfile
import requests
import torch

from transformers import AutoTokenizer, AutoModel

from omnigenbench import (
    OmniDataset,
    ClassificationMetric,
    AccelerateTrainer,
    ModelHub,
    OmniModelForMultiLabelSequenceClassification,
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


def run_fintuning(
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


__all__ = [
    "DeepSEADataset",
    "load_tokenizer_and_model",
    "build_datasets",
    "create_dataloaders",
    "run_fintuning",
    "run_inference",
]



