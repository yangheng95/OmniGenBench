from __future__ import annotations
import os
import json
from typing import Optional
import torch
import torch.nn as nn
from dataclasses import dataclass, field
import warnings

from ..abc.abstract_model import OmniModel

__all__ = [
    "OmniCNNBaseline",
    "OmniRNNBaseline",
    "OmniBPNetBaseline",
    "OmniBasenjiBaseline",
    "OmniDeepSTARRBaseline",
    "OmniGenericBaseline",
    "create_baseline",
]


# ---------------- Utility -----------------
class _MaskedGlobalMaxPool1d(nn.Module):
    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Masked global max-pooling over the sequence dimension.

        Parameters
        ----------
        x : torch.Tensor
            Input features with shape ``(batch_size, seq_len, hidden_size)`` if channels-last,
            or after internal transpose ``(batch_size, channels, seq_len)`` when used inside CNN blocks.
        attention_mask : torch.Tensor, optional
            Binary mask with shape ``(batch_size, seq_len)`` where 1 marks valid tokens and 0 marks padding.

        Returns
        -------
        torch.Tensor
            Pooled tensor of shape ``(batch_size, hidden_size)``.

        Notes
        -----
        - If ``attention_mask`` is provided, positions with 0 are masked using ``-inf`` before max.
        - If ``attention_mask`` is ``None``, a simple ``max(dim=1)`` is applied.

        Pseudocode
        ----------
        .. code-block:: python

            if attention_mask is None:
                return x.max(dim=1).values
            masked_x = x.masked_fill(attention_mask.unsqueeze(-1) == 0, -inf)
            return masked_x.max(dim=1).values
        """
        if attention_mask is None:
            return x.max(dim=1).values
        masked_x = x.masked_fill(attention_mask.unsqueeze(-1).eq(0), float("-inf"))
        return masked_x.max(dim=1).values


# ---------------- Simple Baselines with Heads (legacy style) -----------------
class OmniCNNBaseline(OmniModel):
    """A simple 1D-CNN baseline with global max pooling for multi-label tasks.

    This legacy-style model builds an embedding layer followed by multiple
    convolutional filters with kernel sizes specified in ``kernel_sizes``,
    concatenates the features, pools them, and applies a linear classifier
    with a sigmoid for multi-label probabilities.

    Parameters
    ----------
    tokenizer : Any
        Tokenizer providing ``vocab_size`` or ``get_vocab()`` and ``pad_token_id``.
    num_labels : int
        Number of output labels.
    embed_dim : int, optional
        Token embedding dimension, by default 128.
    num_filters : int, optional
        Number of output channels per convolutional filter, by default 128.
    kernel_sizes : tuple[int, ...], optional
        Convolution kernel sizes, by default ``(3, 5, 7)``.
    dropout : float, optional
        Dropout probability, by default 0.1.

    Inputs
    ------
    input_ids : torch.LongTensor of shape ``(batch_size, seq_len)``
    attention_mask : torch.LongTensor of shape ``(batch_size, seq_len)``, optional
    labels : torch.FloatTensor of shape ``(batch_size, num_labels)``, optional

    Outputs
    -------
    dict
        - ``logits``: ``(batch_size, num_labels)`` in ``[0,1]`` after sigmoid.
        - ``last_hidden_state``: pooled hidden vector ``(batch_size, hidden_size)``.
        - ``labels``: passthrough of input labels if provided.

    Notes
    -----
    - Loss used is BCE (``nn.BCELoss``) expecting labels in ``{0,1}`` floats.
    - See :meth:`predict` and :meth:`inference` for convenience wrappers.

    Pseudocode
    ----------
    .. code-block:: python

        x = Embedding(input_ids)
        x = Dropout(x)
        feats = [Conv1D_k(ReLU)(x_T) for k in kernel_sizes]
        feats = concat(feats, dim=channels).T
        pooled = masked_global_max_pool(feats, attention_mask)
        logits = sigmoid(Linear(pooled))
    """

    def __init__(self, tokenizer, *args, **kwargs):
        embed_dim = kwargs.pop("embed_dim", 128)
        num_filters = kwargs.pop("num_filters", 128)
        kernel_sizes = kwargs.pop("kernel_sizes", (3, 5, 7))
        dropout = kwargs.pop("dropout", 0.1)
        num_labels = kwargs.pop("num_labels")

        class Cfg: ...

        cfg = Cfg()
        cfg.hidden_size = num_filters * len(kernel_sizes)
        cfg.num_labels = num_labels
        cfg.label2id = {str(i): i for i in range(num_labels)}
        cfg.id2label = {i: str(i) for i in range(num_labels)}
        cfg.name_or_path = "CNNBaseline"
        cfg.model_type = "cnn"
        cfg.architectures = ["CNNBaseline"]
        cfg.pad_token_id = getattr(tokenizer, "pad_token_id", -100)
        cfg.embed_dim = embed_dim
        cfg.num_filters = num_filters
        cfg.kernel_sizes = list(kernel_sizes)
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
        vocab_size = getattr(self.tokenizer, "vocab_size", None) or len(
            self.tokenizer.get_vocab()
        )
        _pad = self.pad_token_id
        if isinstance(_pad, torch.Tensor):
            _pad = int(_pad.item())
        elif isinstance(_pad, int) or _pad is None:
            pass
        else:
            _pad = -100
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=_pad)
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(embed_dim, num_filters, k, padding=k // 2), nn.ReLU()
                )
                for k in kernel_sizes
            ]
        )
        self.pool = _MaskedGlobalMaxPool1d()
        self.classifier = nn.Linear(num_filters * len(kernel_sizes), num_labels)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.loss_fn = nn.BCELoss()

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
            num_filters=cfg_dict.get("num_filters", 128),
            kernel_sizes=tuple(cfg_dict.get("kernel_sizes", (3, 5, 7))),
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

    def forward(self, **inputs):
        """Forward pass.

        Parameters
        ----------
        input_ids : torch.LongTensor
            Token ids of shape ``(batch_size, seq_len)``.
        attention_mask : torch.LongTensor, optional
            Mask of shape ``(batch_size, seq_len)`` with 1 for valid tokens.
        labels : torch.FloatTensor, optional
            Multi-hot label matrix ``(batch_size, num_labels)``.

        Returns
        -------
        dict
            Dictionary with ``logits``, ``last_hidden_state``, and optional ``labels``.
        """
        labels = inputs.pop("labels", None)
        x = self.embedding(inputs["input_ids"])
        x = self.dropout(x)
        feats = [m(x.transpose(1, 2)) for m in self.convs]
        feats = torch.cat(feats, dim=1).transpose(1, 2)
        pooled = self.pool(feats, inputs.get("attention_mask"))
        pooled = self.dropout(pooled)
        logits = self.sigmoid(self.classifier(pooled))
        return {"logits": logits, "last_hidden_state": pooled, "labels": labels}

    def predict(self, sequence_or_inputs, **kwargs):
        """Return probabilities for each label.

        This calls the internal convenience routine to accept either raw sequences
        or already-tokenized inputs, then returns the probabilities.

        Returns
        -------
        dict
            Keys: ``predictions`` (alias of ``logits``), ``logits``, ``last_hidden_state``.
        """
        out = self._forward_from_raw_input(sequence_or_inputs, **kwargs)
        return {
            "predictions": out["logits"],
            "logits": out["logits"],
            "last_hidden_state": out["last_hidden_state"],
        }

    def inference(self, sequence_or_inputs, threshold: float = 0.5, **kwargs):
        """Return binary predictions with a threshold.

        Parameters
        ----------
        threshold : float, optional
            Decision threshold in ``[0,1]`` applied to probabilities, by default 0.5.
        """
        out = self._forward_from_raw_input(sequence_or_inputs, **kwargs)
        logits = out["logits"]
        preds = (logits >= threshold).to(torch.int)
        if not isinstance(sequence_or_inputs, list):
            return {
                "predictions": preds[0].cpu(),
                "logits": logits[0].cpu(),
                "confidence": torch.max(logits[0]).cpu(),
                "last_hidden_state": out["last_hidden_state"][0].cpu(),
            }
        return {
            "predictions": preds.cpu(),
            "logits": logits.cpu(),
            "confidence": torch.max(logits, dim=-1)[0].cpu(),
            "last_hidden_state": out["last_hidden_state"],
        }

    def loss_function(self, logits, labels):
        """Binary cross-entropy loss with logits already in probability space.

        Notes
        -----
        This legacy baseline uses ``BCELoss`` assuming inputs were passed through
        a sigmoid already. Prefer ``BCEWithLogitsLoss`` for numerical stability
        in new code.
        """
        return self.loss_fn(logits.view(-1), labels.view(-1).to(torch.float32))


class OmniRNNBaseline(OmniModel):
    """A simple BiLSTM baseline for sequence modeling.

    Embeds tokens, applies a multi-layer LSTM (optionally bidirectional), then
    mean-pools over valid tokens and classifies with a sigmoid layer for
    multi-label probabilities.

    Parameters
    ----------
    tokenizer : Any
        Tokenizer with ``vocab_size`` and ``pad_token_id``.
    num_labels : int
        Number of output labels.
    embed_dim : int, optional
        Embedding dimension, by default 128.
    hidden_dim : int, optional
        LSTM hidden size per direction, by default 256.
    num_layers : int, optional
        Number of LSTM layers, by default 1.
    bidirectional : bool, optional
        Whether to use a bidirectional LSTM, by default True.
    dropout : float, optional
        Dropout probability, by default 0.1.

    Outputs
    -------
    dict
        - ``logits``: probabilities after sigmoid ``(batch_size, num_labels)``.
        - ``last_hidden_state``: pooled vector ``(batch_size, hidden_size)``.
        - ``labels``: passthrough if provided.

    Pseudocode
    ----------
    .. code-block:: python

        x = Embedding(input_ids)
        x = Dropout(x)
        seq_out, _ = LSTM(x)
        pooled = masked_mean(seq_out, attention_mask)
        logits = sigmoid(Linear(pooled))
    """

    def __init__(self, tokenizer, *args, **kwargs):
        embed_dim = kwargs.pop("embed_dim", 128)
        hidden_dim = kwargs.pop("hidden_dim", 256)
        num_layers = kwargs.pop("num_layers", 1)
        bidirectional = kwargs.pop("bidirectional", True)
        dropout = kwargs.pop("dropout", 0.1)
        num_labels = kwargs.pop("num_labels")

        class Cfg: ...

        cfg = Cfg()
        cfg.hidden_size = hidden_dim * (2 if bidirectional else 1)
        cfg.num_labels = num_labels
        cfg.label2id = {str(i): i for i in range(num_labels)}
        cfg.id2label = {i: str(i) for i in range(num_labels)}
        cfg.name_or_path = "RNNBaseline"
        cfg.model_type = "rnn"
        cfg.architectures = ["RNNBaseline"]
        cfg.pad_token_id = getattr(tokenizer, "pad_token_id", -100)
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
        vocab_size = getattr(self.tokenizer, "vocab_size", None) or len(
            self.tokenizer.get_vocab()
        )
        _pad = self.pad_token_id
        if isinstance(_pad, torch.Tensor):
            _pad = int(_pad.item())
        elif isinstance(_pad, int) or _pad is None:
            pass
        else:
            _pad = -100
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=_pad)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0 if num_layers == 1 else dropout,
            bidirectional=bidirectional,
        )
        self.classifier = nn.Linear(
            hidden_dim * (2 if bidirectional else 1), num_labels
        )
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.loss_fn = nn.BCELoss()

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
            embed_dim=cfg_dict.get("embed_dim", cfg_dict.get("hidden_size", 128)),
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

    def forward(self, **inputs):
        """Forward pass producing multi-label probabilities and hidden state.

        Returns
        -------
        dict
            Dictionary with keys ``logits``, ``last_hidden_state``, optional ``labels``.
        """
        labels = inputs.pop("labels", None)
        x = self.embedding(inputs["input_ids"])
        x = self.dropout(x)
        out, _ = self.lstm(x)
        pooled = self._mask_mean_pool(out, inputs.get("attention_mask"))
        pooled = self.dropout(pooled)
        logits = self.sigmoid(self.classifier(pooled))
        return {"logits": logits, "last_hidden_state": pooled, "labels": labels}

    def predict(self, sequence_or_inputs, **kwargs):
        """Return probabilities for each label (alias of forward logits)."""
        out = self._forward_from_raw_input(sequence_or_inputs, **kwargs)
        return {
            "predictions": out["logits"],
            "logits": out["logits"],
            "last_hidden_state": out["last_hidden_state"],
        }

    def inference(self, sequence_or_inputs, threshold: float = 0.5, **kwargs):
        """Return binary predictions using the specified probability threshold."""
        out = self._forward_from_raw_input(sequence_or_inputs, **kwargs)
        logits = out["logits"]
        preds = (logits >= threshold).to(torch.int)
        if not isinstance(sequence_or_inputs, list):
            return {
                "predictions": preds[0].cpu(),
                "logits": logits[0].cpu(),
                "confidence": torch.max(logits[0]).cpu(),
                "last_hidden_state": out["last_hidden_state"][0].cpu(),
            }
        return {
            "predictions": preds.cpu(),
            "logits": logits.cpu(),
            "confidence": torch.max(logits, dim=-1)[0].cpu(),
            "last_hidden_state": out["last_hidden_state"],
        }

    def loss_function(self, logits, labels):
        """Binary cross-entropy loss on probabilities."""
        return self.loss_fn(logits.view(-1), labels.view(-1).to(torch.float32))


class OmniBPNetBaseline(OmniModel):
    """A lightweight BPNet-like dilated-convolution baseline.

    The model converts token ids to one-hot nucleotides (A,C,G,T), applies a
    first convolution and a stack of exponentially-dilated 1D convolutions with
    residual connections, averages globally, then classifies with a sigmoid.

    Parameters
    ----------
    tokenizer : Any
        Tokenizer that can convert tokens to ids for "A", "C", "G", "T".
    num_labels : int
        Number of outputs.
    n_filters : int, optional
        Number of channels in convolution blocks, by default 64.
    n_dilated_layers : int, optional
        Number of dilated layers, by default 9.
    conv1_kernel_size : int, optional
        Kernel size of the first conv, by default 25.
    dil_kernel_size : int, optional
        Kernel size of dilated convs, by default 3.
    dropout : float, optional
        Dropout after global pooling, by default 0.1.

    Pseudocode
    ----------
    .. code-block:: python

        X = one_hot(input_ids)  # (B, 4, L)
        H = relu(Conv1D(4->C, k=25)(X))
        for i in range(n_layers):
            R = H
            H = relu(DilatedConv1D(C->C, k=3, d=2**i)(H))
            H = H + R
        g = GlobalAvgPool1D(H)  # (B, C)
        y = sigmoid(Linear(C->num_labels)(Dropout(g)))
    """

    def __init__(self, tokenizer, *args, **kwargs):
        n_outputs = kwargs.pop("num_labels")
        n_filters = kwargs.pop("n_filters", 64)
        n_dilated_layers = kwargs.pop("n_dilated_layers", 9)
        conv1_kernel_size = kwargs.pop("conv1_kernel_size", 25)
        dil_kernel_size = kwargs.pop("dil_kernel_size", 3)
        dropout = kwargs.pop("dropout", 0.1)

        class Cfg: ...

        cfg = Cfg()
        cfg.hidden_size = n_filters
        cfg.num_labels = n_outputs
        cfg.label2id = {str(i): i for i in range(n_outputs)}
        cfg.id2label = {i: str(i) for i in range(n_outputs)}
        cfg.name_or_path = "BPNetBaseline"
        cfg.model_type = "bpnet"
        cfg.architectures = ["BPNetBaseline"]
        cfg.pad_token_id = getattr(tokenizer, "pad_token_id", -100)
        cfg.n_filters = n_filters
        cfg.n_dilated_layers = n_dilated_layers
        cfg.conv1_kernel_size = conv1_kernel_size
        cfg.dil_kernel_size = dil_kernel_size
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

        super().__init__(_Stub(cfg), tokenizer, num_labels=n_outputs, *args, **kwargs)
        vocab_size = getattr(self.tokenizer, "vocab_size", None) or len(
            self.tokenizer.get_vocab()
        )
        weight = torch.zeros(vocab_size, 4)
        # Support both uppercase and lowercase nucleotide tokens
        for i, toks in enumerate([("A", "a"), ("C", "c"), ("G", "g"), ("T", "t")]):
            for tok in toks:
                try:
                    tid = self.tokenizer.convert_tokens_to_ids(tok)
                    if tid is not None and tid >= 0 and tid < weight.size(0):
                        weight[tid, i] = 1.0
                except Exception:
                    pass
        self.register_buffer("_one_hot_weight", weight, persistent=False)
        self.conv1 = nn.Conv1d(4, n_filters, conv1_kernel_size, padding="same")
        self.dilated_convs = nn.ModuleList(
            [
                nn.Conv1d(
                    n_filters, n_filters, dil_kernel_size, padding="same", dilation=2**i
                )
                for i in range(n_dilated_layers)
            ]
        )
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(n_filters, n_outputs)
        self.sigmoid = nn.Sigmoid()
        self.loss_fn = nn.BCELoss()
        self.dropout_layer = nn.Dropout(dropout)

    def _tokens_to_one_hot(self, input_ids: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.embedding(input_ids, self._one_hot_weight).permute(
            0, 2, 1
        )

    def forward(self, **inputs):
        """Forward pass returning probabilities and last hidden state.

        Returns
        -------
        dict
            Keys: ``logits`` (probabilities), ``last_hidden_state`` (pooled), optional ``labels``.
        """
        labels = inputs.pop("labels", None)
        x = self._tokens_to_one_hot(inputs["input_ids"])  # [B,4,L]
        x = torch.relu(self.conv1(x))
        for layer in self.dilated_convs:
            residual = x
            x = torch.relu(layer(x))
            x = x + residual
        x = self.global_avg_pool(x).squeeze(-1)
        x = self.dropout_layer(x)
        logits = self.sigmoid(self.classifier(x))
        return {"logits": logits, "last_hidden_state": x, "labels": labels}

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
            "n_filters": getattr(self.config, "n_filters", None),
            "n_dilated_layers": getattr(self.config, "n_dilated_layers", None),
            "conv1_kernel_size": getattr(self.config, "conv1_kernel_size", None),
            "dil_kernel_size": getattr(self.config, "dil_kernel_size", None),
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
            n_filters=cfg_dict.get("n_filters", 64),
            n_dilated_layers=cfg_dict.get("n_dilated_layers", 9),
            conv1_kernel_size=cfg_dict.get("conv1_kernel_size", 25),
            dil_kernel_size=cfg_dict.get("dil_kernel_size", 3),
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

    def forward(self, **inputs):
        """Forward pass returning probabilities and last hidden state.

        Returns
        -------
        dict
            Keys: ``logits`` (probabilities), ``last_hidden_state`` (pooled), optional ``labels``.
        """
        labels = inputs.pop("labels", None)
        x = self._tokens_to_one_hot(inputs["input_ids"])  # [B,4,L]
        x = torch.relu(self.conv1(x))
        for layer in self.dilated_convs:
            residual = x
            x = torch.relu(layer(x))
            x = x + residual
        x = self.global_avg_pool(x).squeeze(-1)
        x = self.dropout_layer(x)
        logits = self.sigmoid(self.classifier(x))
        return {"logits": logits, "last_hidden_state": x, "labels": labels}

    def predict(self, sequence_or_inputs, **kwargs):
        """Return probabilities for each label."""
        out = self._forward_from_raw_input(sequence_or_inputs, **kwargs)
        return {
            "predictions": out["logits"],
            "logits": out["logits"],
            "last_hidden_state": out["last_hidden_state"],
        }

    def inference(self, sequence_or_inputs, threshold: float = 0.5, **kwargs):
        """Return thresholded predictions along with confidence scores."""
        out = self._forward_from_raw_input(sequence_or_inputs, **kwargs)
        logits = out["logits"]
        preds = (logits >= threshold).to(torch.int)
        if not isinstance(sequence_or_inputs, list):
            return {
                "predictions": preds[0].cpu(),
                "logits": logits[0].cpu(),
                "confidence": torch.max(logits[0]).cpu(),
                "last_hidden_state": out["last_hidden_state"][0].cpu(),
            }
        return {
            "predictions": preds.cpu(),
            "logits": logits.cpu(),
            "confidence": torch.max(logits, dim=-1)[0].cpu(),
            "last_hidden_state": out["last_hidden_state"],
        }

    def loss_function(self, logits, labels):
        return self.loss_fn(logits.view(-1), labels.view(-1).to(torch.float32))


class OmniBasenjiBaseline(OmniModel):
    """Basenji-like 1D CNN with dilations adapted for tokenizer-driven inputs.

    This baseline maps token ids to A/C/G/T channels, stacks conv+pool blocks,
    applies dilated residual blocks, then global-average-pools and classifies.
    """

    def __init__(self, tokenizer, *args, **kwargs):
        num_labels = kwargs.pop("num_labels")
        conv1kc = kwargs.pop("conv1kc", 64)
        conv1ks = kwargs.pop("conv1ks", 15)
        pool1ks = kwargs.pop("pool1ks", 8)
        conv2kc = kwargs.pop("conv2kc", 64)
        conv2ks = kwargs.pop("conv2ks", 5)
        pool2ks = kwargs.pop("pool2ks", 4)
        conv3kc = kwargs.pop("conv3kc", round(64 * 1.125))
        conv3ks = kwargs.pop("conv3ks", 5)
        pool3ks = kwargs.pop("pool3ks", 4)
        convdc = kwargs.pop("convdc", 6)
        dropout = kwargs.pop("dropout", 0.1)

        class Cfg: ...

        cfg = Cfg()
        cfg.hidden_size = 64
        cfg.num_labels = num_labels
        cfg.label2id = {str(i): i for i in range(num_labels)}
        cfg.id2label = {i: str(i) for i in range(num_labels)}
        cfg.name_or_path = "BasenjiBaseline"
        cfg.model_type = "basenji"
        cfg.architectures = ["BasenjiBaseline"]
        cfg.pad_token_id = getattr(tokenizer, "pad_token_id", -100)
        cfg.dropout = dropout
        cfg.convdc = convdc

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

        # Build token-id -> one-hot(A,C,G,T) projection
        vocab_size = getattr(self.tokenizer, "vocab_size", None) or len(
            self.tokenizer.get_vocab()
        )
        weight = torch.zeros(vocab_size, 4)
        for i, toks in enumerate([("A", "a"), ("C", "c"), ("G", "g"), ("T", "t")]):
            for tok in toks:
                try:
                    tid = self.tokenizer.convert_tokens_to_ids(tok)
                    if tid is not None and isinstance(tid, int) and 0 <= tid < weight.size(0):
                        weight[tid, i] = 1.0
                except Exception:
                    pass
        self.register_buffer("_one_hot_weight", weight, persistent=False)

        self.act = nn.GELU()
        self.conv_block_1 = nn.Sequential(
            self.act,
            nn.Conv1d(4, conv1kc, kernel_size=conv1ks, padding=conv1ks // 2, bias=False),
            nn.BatchNorm1d(conv1kc, momentum=0.9, affine=True),
            nn.MaxPool1d(kernel_size=pool1ks, ceil_mode=True),
            nn.Dropout(p=0.2),
        )
        self.conv_block_2 = nn.Sequential(
            self.act,
            nn.Conv1d(
                conv1kc, conv2kc, kernel_size=conv2ks, padding=conv2ks // 2, bias=False
            ),
            nn.BatchNorm1d(conv2kc, momentum=0.9, affine=True),
            nn.MaxPool1d(kernel_size=pool2ks, ceil_mode=True),
            nn.Dropout(p=0.2),
        )
        self.conv_block_3 = nn.Sequential(
            self.act,
            nn.Conv1d(
                conv2kc, conv3kc, kernel_size=conv3ks, padding=conv3ks // 2, bias=False
            ),
            nn.BatchNorm1d(conv3kc, momentum=0.9, affine=True),
            nn.MaxPool1d(kernel_size=pool3ks, ceil_mode=True),
            nn.Dropout(p=0.2),
        )
        self.dilations = nn.ModuleList()
        for i in range(convdc):
            self.dilations.append(
                nn.Sequential(
                    self.act,
                    nn.Conv1d(
                        conv3kc,
                        32,
                        kernel_size=3,
                        padding=2 ** i,
                        dilation=2 ** i,
                        bias=False,
                    ),
                    nn.BatchNorm1d(32, momentum=0.9, affine=True),
                    self.act,
                    nn.Conv1d(32, 72, kernel_size=1, padding=0, bias=False),
                    nn.BatchNorm1d(72, momentum=0.9, affine=True),
                    nn.Dropout(p=0.25),
                )
            )
        self.conv_block_4 = nn.Sequential(
            self.act,
            nn.Conv1d(72, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(64, momentum=0.9, affine=True),
            nn.Dropout(p=0.1),
        )
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(64, num_labels)
        self.sigmoid = nn.Sigmoid()
        self.loss_fn = nn.BCELoss()

    def _tokens_to_one_hot(self, input_ids: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.embedding(input_ids, self._one_hot_weight).permute(
            0, 2, 1
        )

    def forward(self, **inputs):
        """Forward pass returning probabilities and last hidden state.

        Returns
        -------
        dict
            Keys: ``logits`` (probabilities), ``last_hidden_state`` (pooled), optional ``labels``.
        """
        labels = inputs.pop("labels", None)
        x = self._tokens_to_one_hot(inputs["input_ids"])  # [B,4,L]
        x = torch.relu(self.conv1(x))
        for layer in self.dilated_convs:
            residual = x
            x = torch.relu(layer(x))
            x = x + residual
        x = self.conv_block_4(x)
        x = self.global_avg_pool(x).squeeze(-1)  # [B,64]
        logits = self.sigmoid(self.classifier(x))
        return {"logits": logits, "last_hidden_state": x, "labels": labels}

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
            "n_filters": getattr(self.config, "n_filters", None),
            "n_dilated_layers": getattr(self.config, "n_dilated_layers", None),
            "conv1_kernel_size": getattr(self.config, "conv1_kernel_size", None),
            "dil_kernel_size": getattr(self.config, "dil_kernel_size", None),
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
            n_filters=cfg_dict.get("n_filters", 64),
            n_dilated_layers=cfg_dict.get("n_dilated_layers", 9),
            conv1_kernel_size=cfg_dict.get("conv1_kernel_size", 25),
            dil_kernel_size=cfg_dict.get("dil_kernel_size", 3),
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

    def forward(self, **inputs):
        """Forward pass returning probabilities and last hidden state.

        Returns
        -------
        dict
            Keys: ``logits`` (probabilities), ``last_hidden_state`` (pooled), optional ``labels``.
        """
        labels = inputs.pop("labels", None)
        x = self._tokens_to_one_hot(inputs["input_ids"])  # [B,4,L]
        x = torch.relu(self.conv1(x))
        for layer in self.dilated_convs:
            residual = x
            x = torch.relu(layer(x))
            x = x + residual
        x = self.global_avg_pool(x).squeeze(-1)
        x = self.dropout_layer(x)
        logits = self.sigmoid(self.classifier(x))
        return {"logits": logits, "last_hidden_state": x, "labels": labels}

    def predict(self, sequence_or_inputs, **kwargs):
        """Return probabilities for each label."""
        out = self._forward_from_raw_input(sequence_or_inputs, **kwargs)
        return {
            "predictions": out["logits"],
            "logits": out["logits"],
            "last_hidden_state": out["last_hidden_state"],
        }

    def inference(self, sequence_or_inputs, threshold: float = 0.5, **kwargs):
        """Return thresholded predictions along with confidence scores."""
        out = self._forward_from_raw_input(sequence_or_inputs, **kwargs)
        logits = out["logits"]
        preds = (logits >= threshold).to(torch.int)
        if not isinstance(sequence_or_inputs, list):
            return {
                "predictions": preds[0].cpu(),
                "logits": logits[0].cpu(),
                "confidence": torch.max(logits[0]).cpu(),
                "last_hidden_state": out["last_hidden_state"][0].cpu(),
            }
        return {
            "predictions": preds.cpu(),
            "logits": logits.cpu(),
            "confidence": torch.max(logits, dim=-1)[0].cpu(),
            "last_hidden_state": out["last_hidden_state"],
        }

    def loss_function(self, logits, labels):
        return self.loss_fn(logits.view(-1), labels.view(-1).to(torch.float32))


class OmniDeepSTARRBaseline(OmniModel):
    """DeepSTARR-like CNN with global pooling and MLP head adapted for tokenizer inputs."""

    def __init__(self, tokenizer, *args, **kwargs):
        num_labels = kwargs.pop("num_labels")
        dropout_prob = kwargs.pop("dropout_prob", 0.4)
        num_filters1 = kwargs.pop("num_filters1", 256)
        kernel_size1 = kwargs.pop("kernel_size1", 7)
        num_filters2 = kwargs.pop("num_filters2", 60)
        kernel_size2 = kwargs.pop("kernel_size2", 3)
        num_filters3 = kwargs.pop("num_filters3", 60)
        kernel_size3 = kwargs.pop("kernel_size3", 5)
        num_filters4 = kwargs.pop("num_filters4", 120)
        kernel_size4 = kwargs.pop("kernel_size4", 3)
        dense_neurons1 = kwargs.pop("dense_neurons1", 256)
        dense_neurons2 = kwargs.pop("dense_neurons2", 256)

        class Cfg: ...

        cfg = Cfg()
        cfg.hidden_size = num_filters4
        cfg.num_labels = num_labels
        cfg.label2id = {str(i): i for i in range(num_labels)}
        cfg.id2label = {i: str(i) for i in range(num_labels)}
        cfg.name_or_path = "DeepSTARRBaseline"
        cfg.model_type = "deepstarr"
        cfg.architectures = ["DeepSTARRBaseline"]
        cfg.pad_token_id = getattr(tokenizer, "pad_token_id", -100)

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

        # Build token-id -> one-hot(A,C,G,T) projection
        vocab_size = getattr(self.tokenizer, "vocab_size", None) or len(
            self.tokenizer.get_vocab()
        )
        weight = torch.zeros(vocab_size, 4)
        for i, toks in enumerate([("A", "a"), ("C", "c"), ("G", "g"), ("T", "t")]):
            for tok in toks:
                try:
                    tid = self.tokenizer.convert_tokens_to_ids(tok)
                    if tid is not None and isinstance(tid, int) and 0 <= tid < weight.size(0):
                        weight[tid, i] = 1.0
                except Exception:
                    pass
        self.register_buffer("_one_hot_weight", weight, persistent=False)

        def block(in_ch, out_ch, k):
            return nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.MaxPool1d(2),
            )

        self.conv = nn.Sequential(
            block(4, num_filters1, kernel_size1),
            block(num_filters1, num_filters2, kernel_size2),
            block(num_filters2, num_filters3, kernel_size3),
            block(num_filters3, num_filters4, kernel_size4),
        )
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.mlp = nn.Sequential(
            nn.Linear(num_filters4, dense_neurons1),
            nn.BatchNorm1d(dense_neurons1),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(dense_neurons1, dense_neurons2),
            nn.BatchNorm1d(dense_neurons2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
        )
        self.classifier = nn.Linear(dense_neurons2, num_labels)
        self.sigmoid = nn.Sigmoid()
        self.loss_fn = nn.BCELoss()

    def _tokens_to_one_hot(self, input_ids: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.embedding(input_ids, self._one_hot_weight).permute(
            0, 2, 1
        )

    def forward(self, **inputs):
        """Forward pass returning probabilities and last hidden state.

        Returns
        -------
        dict
            Keys: ``logits`` (probabilities), ``last_hidden_state`` (pooled), optional ``labels``.
        """
        labels = inputs.pop("labels", None)
        x = self._tokens_to_one_hot(inputs["input_ids"])  # [B,4,L]
        x = self.conv(x)
        x = self.global_avg_pool(x).squeeze(-1)
        feats = self.mlp(x)
        logits = self.sigmoid(self.classifier(feats))
        return {"logits": logits, "last_hidden_state": feats, "labels": labels}

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
            "n_filters": getattr(self.config, "n_filters", None),
            "n_dilated_layers": getattr(self.config, "n_dilated_layers", None),
            "conv1_kernel_size": getattr(self.config, "conv1_kernel_size", None),
            "dil_kernel_size": getattr(self.config, "dil_kernel_size", None),
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
            n_filters=cfg_dict.get("n_filters", 64),
            n_dilated_layers=cfg_dict.get("n_dilated_layers", 9),
            conv1_kernel_size=cfg_dict.get("conv1_kernel_size", 25),
            dil_kernel_size=cfg_dict.get("dil_kernel_size", 3),
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

    def forward(self, **inputs):
        """Forward pass returning probabilities and last hidden state.

        Returns
        -------
        dict
            Keys: ``logits`` (probabilities), ``last_hidden_state`` (pooled), optional ``labels``.
        """
        labels = inputs.pop("labels", None)
        x = self._tokens_to_one_hot(inputs["input_ids"])  # [B,4,L]
        x = torch.relu(self.conv1(x))
        for layer in self.dilated_convs:
            residual = x
            x = torch.relu(layer(x))
            x = x + residual
        x = self.global_avg_pool(x).squeeze(-1)
        x = self.dropout_layer(x)
        logits = self.sigmoid(self.classifier(x))
        return {"logits": logits, "last_hidden_state": x, "labels": labels}

    def predict(self, sequence_or_inputs, **kwargs):
        """Return probabilities for each label."""
        out = self._forward_from_raw_input(sequence_or_inputs, **kwargs)
        return {
            "predictions": out["logits"],
            "logits": out["logits"],
            "last_hidden_state": out["last_hidden_state"],
        }

    def inference(self, sequence_or_inputs, threshold: float = 0.5, **kwargs):
        """Return thresholded predictions along with confidence scores."""
        out = self._forward_from_raw_input(sequence_or_inputs, **kwargs)
        logits = out["logits"]
        preds = (logits >= threshold).to(torch.int)
        if not isinstance(sequence_or_inputs, list):
            return {
                "predictions": preds[0].cpu(),
                "logits": logits[0].cpu(),
                "confidence": torch.max(logits[0]).cpu(),
                "last_hidden_state": out["last_hidden_state"][0].cpu(),
            }
        return {
            "predictions": preds.cpu(),
            "logits": logits.cpu(),
            "confidence": torch.max(logits, dim=-1)[0].cpu(),
            "last_hidden_state": out["last_hidden_state"],
        }

    def loss_function(self, logits, labels):
        return self.loss_fn(logits.view(-1), labels.view(-1).to(torch.float32))


# ---------------- Generic Multi-Task Framework -----------------
@dataclass
class BaselineConfig:
    """Configuration for baseline backbones and heads.

    Attributes
    ----------
    backbone_type : {"cnn", "rnn", "bpnet", "deepstarr", "basenji"}
        Backbone to instantiate.
    task_name : {"multilabel_classification", "classification", "regression", "token_classification", "token_regression", "multilabel_token_classification"}
        Head task type.
    vocab_size : int
        Vocabulary size for embedding layers.
    hidden_size : int
        Feature size exposed by the backbone for the head.
    num_labels : int
        Number of outputs for the head.
    loss_type : str
        Placeholder for future customization; currently auto-selected per task.
    label2id, id2label : dict
        Label mapping dictionaries.
    pad_token_id : int
        Padding token id used by embeddings and loss functions.
    embed_dim, num_filters, kernel_sizes, dropout, hidden_dim, num_layers, bidirectional :
        Hyperparameters for CNN and RNN backbones.
    n_filters, n_dilated_layers, conv1_kernel_size, dil_kernel_size :
        Hyperparameters for BPNet-style backbone.
    """

    backbone_type: str = "cnn"
    task_name: str = "multilabel_classification"
    vocab_size: int = 0
    hidden_size: int = 128
    num_labels: int = 2
    loss_type: str = "auto"
    label2id: dict = field(default_factory=dict)
    id2label: dict = field(default_factory=dict)
    pad_token_id: int = -100
    embed_dim: int = 128
    num_filters: int = 128
    kernel_sizes: tuple = (3, 5, 7)
    dropout: float = 0.1
    hidden_dim: int = 256
    num_layers: int = 1
    bidirectional: bool = True
    n_filters: int = 64
    n_dilated_layers: int = 9
    conv1_kernel_size: int = 25
    dil_kernel_size: int = 3


class BackboneBase(nn.Module):
    """Abstract base for simple backbones returning both token-level and pooled states."""

    def forward(self, input_ids, attention_mask=None):
        """Compute forward features.

        Parameters
        ----------
        input_ids : torch.LongTensor
            Shape ``(batch_size, seq_len)``.
        attention_mask : torch.LongTensor, optional
            Shape ``(batch_size, seq_len)``; 1 valid, 0 padding.

        Returns
        -------
        dict
            ``{"sequence_output": (B, L, H), "hidden_state": (B, H)}``.
        """
        raise NotImplementedError


class CNNBackbone(BackboneBase):
    """Embedding + multi-kernel 1D-CNN with masked global max-pooling."""

    def __init__(self, cfg: BaselineConfig):
        """Build the CNN backbone.

        Parameters
        ----------
        cfg : BaselineConfig
            Configuration with embedding and convolution hyperparameters.
        """
        super().__init__()
        self.embedding = nn.Embedding(
            cfg.vocab_size, cfg.embed_dim, padding_idx=cfg.pad_token_id
        )
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(cfg.embed_dim, cfg.num_filters, k, padding=k // 2),
                    nn.ReLU(),
                )
                for k in cfg.kernel_sizes
            ]
        )
        self.dropout = nn.Dropout(cfg.dropout)
        self.pool = _MaskedGlobalMaxPool1d()
        self.out_dim = cfg.num_filters * len(cfg.kernel_sizes)

    def forward(self, input_ids, attention_mask=None):
        """Return token-level and pooled features.

        Returns
        -------
        dict
            ``sequence_output``: ``(B, L, H)``, ``hidden_state``: ``(B, H)``.
        """
        x = self.embedding(input_ids)
        x = self.dropout(x)
        feats = [m(x.transpose(1, 2)) for m in self.convs]
        feats = torch.cat(feats, dim=1).transpose(1, 2)
        pooled = self.pool(feats, attention_mask)
        return {"sequence_output": feats, "hidden_state": pooled}


class RNNBackbone(BackboneBase):
    """Embedding + LSTM (optionally bidirectional) with masked mean pooling."""

    def __init__(self, cfg: BaselineConfig):
        """Build the RNN backbone with LSTM."""
        super().__init__()
        self.embedding = nn.Embedding(
            cfg.vocab_size, cfg.embed_dim, padding_idx=cfg.pad_token_id
        )
        self.lstm = nn.LSTM(
            cfg.embed_dim,
            cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=0.0 if cfg.num_layers == 1 else cfg.dropout,
            bidirectional=cfg.bidirectional,
        )
        self.dropout = nn.Dropout(cfg.dropout)
        self.out_dim = cfg.hidden_dim * (2 if cfg.bidirectional else 1)

    def forward(self, input_ids, attention_mask=None):
        """Return token-level and pooled features using mean-pooling over mask."""
        x = self.embedding(input_ids)
        x = self.dropout(x)
        seq_out, _ = self.lstm(x)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).to(seq_out.dtype)
            pooled = (seq_out * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-6)
        else:
            pooled = seq_out.mean(dim=1)
        pooled = self.dropout(pooled)
        return {"sequence_output": seq_out, "hidden_state": pooled}


class BPNetBackbone(BackboneBase):
    """Dilated-convolution backbone with residual connections (BPNet-style)."""

    def __init__(self, cfg: BaselineConfig):
        """Initialize convolutional stack and one-hot projection buffer."""
        super().__init__()
        self.register_buffer(
            "_one_hot_weight", torch.zeros(cfg.vocab_size, 4), persistent=False
        )
        self.conv1 = nn.Conv1d(4, cfg.n_filters, cfg.conv1_kernel_size, padding="same")
        self.dilated_convs = nn.ModuleList(
            [
                nn.Conv1d(
                    cfg.n_filters,
                    cfg.n_filters,
                    cfg.dil_kernel_size,
                    padding="same",
                    dilation=2**i,
                )
                for i in range(cfg.n_dilated_layers)
            ]
        )
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.out_dim = cfg.n_filters
        self.dropout = nn.Dropout(cfg.dropout)

    def set_one_hot(self, weight):
        """Set the token-id-to-one-hot projection matrix.

        Parameters
        ----------
        weight : torch.Tensor
            Shape ``(vocab_size, 4)`` mapping token ids to A,C,G,T one-hots.
        """
        if weight.shape == self._one_hot_weight.shape:
            self._one_hot_weight = weight

    def _tokens_to_one_hot(self, ids):
        """Project token ids to nucleotide one-hot channels.

        Parameters
        ----------
        ids : torch.LongTensor
            Shape ``(batch_size, seq_len)``.

        Returns
        -------
        torch.Tensor
            One-hot tensor with shape ``(batch_size, 4, seq_len)``.
        """
        return torch.nn.functional.embedding(ids, self._one_hot_weight).permute(0, 2, 1)

    def forward(self, input_ids, attention_mask=None):
        """Return sequence and pooled outputs from dilated conv stack."""
        x = self._tokens_to_one_hot(input_ids)
        x = torch.relu(self.conv1(x))
        for layer in self.dilated_convs:
            residual = x
            x = torch.relu(layer(x))
            x = x + residual
        seq_out = x.transpose(1, 2)
        pooled = self.global_avg_pool(x).squeeze(-1)
        pooled = self.dropout(pooled)
        return {"sequence_output": seq_out, "hidden_state": pooled}


class DeepSTARRBackbone(BackboneBase):
    """DeepSTARR-style convolutional backbone with global average pooling.

    Converts token ids to A/C/G/T one-hots, applies stacked conv+BN+ReLU+pool
    blocks, then projects pooled features to a configurable hidden size.
    """

    def __init__(self, cfg: BaselineConfig):
        super().__init__()
        self.register_buffer(
            "_one_hot_weight", torch.zeros(cfg.vocab_size, 4), persistent=False
        )
        # Defaults aligned with DeepSTARR baseline
        num_filters1, k1 = 256, 7
        num_filters2, k2 = 60, 3
        num_filters3, k3 = 60, 5
        num_filters4, k4 = 120, 3

        def block(in_ch, out_ch, k):
            return nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.MaxPool1d(2),
            )

        self.conv = nn.Sequential(
            block(4, num_filters1, k1),
            block(num_filters1, num_filters2, k2),
            block(num_filters2, num_filters3, k3),
            block(num_filters3, num_filters4, k4),
        )
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(cfg.dropout)
        # Project to backbone hidden size for heads
        self.proj = nn.Linear(num_filters4, cfg.hidden_size)
        self.out_dim = cfg.hidden_size

    def set_one_hot(self, weight: torch.Tensor):
        if weight.shape == self._one_hot_weight.shape:
            self._one_hot_weight = weight

    def _tokens_to_one_hot(self, ids: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.embedding(ids, self._one_hot_weight).permute(0, 2, 1)

    def forward(self, input_ids, attention_mask=None):
        x = self._tokens_to_one_hot(input_ids)
        x = self.conv(x)
        seq_out = x.transpose(1, 2)
        pooled = self.global_avg_pool(x).squeeze(-1)
        pooled = self.dropout(pooled)
        pooled = self.proj(pooled)
        return {"sequence_output": seq_out, "hidden_state": pooled}


class BasenjiBackbone(BackboneBase):
    """Basenji-like convolutional backbone with dilations and global pooling."""

    def __init__(self, cfg: BaselineConfig):
        super().__init__()
        self.register_buffer(
            "_one_hot_weight", torch.zeros(cfg.vocab_size, 4), persistent=False
        )
        self.act = nn.GELU()
        # Defaults aligned with Basenji baseline
        conv1kc, conv1ks, pool1ks = 64, 15, 8
        conv2kc, conv2ks, pool2ks = 64, 5, 4
        conv3kc, conv3ks, pool3ks = round(64 * 1.125), 5, 4
        convdc = 6

        self.conv_block_1 = nn.Sequential(
            self.act,
            nn.Conv1d(4, conv1kc, kernel_size=conv1ks, padding=conv1ks // 2, bias=False),
            nn.BatchNorm1d(conv1kc, momentum=0.9, affine=True),
            nn.MaxPool1d(kernel_size=pool1ks, ceil_mode=True),
            nn.Dropout(p=0.2),
        )
        self.conv_block_2 = nn.Sequential(
            self.act,
            nn.Conv1d(conv1kc, conv2kc, kernel_size=conv2ks, padding=conv2ks // 2, bias=False),
            nn.BatchNorm1d(conv2kc, momentum=0.9, affine=True),
            nn.MaxPool1d(kernel_size=pool2ks, ceil_mode=True),
            nn.Dropout(p=0.2),
        )
        self.conv_block_3 = nn.Sequential(
            self.act,
            nn.Conv1d(conv2kc, conv3kc, kernel_size=conv3ks, padding=conv3ks // 2, bias=False),
            nn.BatchNorm1d(conv3kc, momentum=0.9, affine=True),
            nn.MaxPool1d(kernel_size=pool3ks, ceil_mode=True),
            nn.Dropout(p=0.2),
        )
        self.dilations = nn.ModuleList(
            [
                nn.Sequential(
                    self.act,
                    nn.Conv1d(conv3kc, 32, kernel_size=3, padding=2 ** i, dilation=2 ** i, bias=False),
                    nn.BatchNorm1d(32, momentum=0.9, affine=True),
                    self.act,
                    nn.Conv1d(32, 72, kernel_size=1, padding=0, bias=False),
                    nn.BatchNorm1d(72, momentum=0.9, affine=True),
                    nn.Dropout(p=0.25),
                )
                for i in range(convdc)
            ]
        )
        self.conv_block_4 = nn.Sequential(
            self.act,
            nn.Conv1d(72, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(64, momentum=0.9, affine=True),
            nn.Dropout(p=0.1),
        )
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(cfg.dropout)
        self.proj = nn.Linear(64, cfg.hidden_size)
        self.out_dim = cfg.hidden_size

    def set_one_hot(self, weight: torch.Tensor):
        if weight.shape == self._one_hot_weight.shape:
            self._one_hot_weight = weight

    def _tokens_to_one_hot(self, ids: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.embedding(ids, self._one_hot_weight).permute(0, 2, 1)

    def forward(self, input_ids, attention_mask=None):
        x = self._tokens_to_one_hot(input_ids)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        for layer in self.dilations:
            x = layer(x)
        x = self.conv_block_4(x)
        seq_out = x.transpose(1, 2)
        pooled = self.global_avg_pool(x).squeeze(-1)
        pooled = self.dropout(pooled)
        pooled = self.proj(pooled)
        return {"sequence_output": seq_out, "hidden_state": pooled}


BACKBONE_REGISTRY = {
    "cnn": CNNBackbone,
    "rnn": RNNBackbone,
    "bpnet": BPNetBackbone,
    "deepstarr": DeepSTARRBackbone,
    "basenji": BasenjiBackbone,
}


class HeadBase(nn.Module):
    """Abstract prediction head consuming backbone features."""

    def forward(self, features: dict, labels=None):
        """Compute head logits and optional loss.

        Parameters
        ----------
        features : dict
            Must contain ``hidden_state`` or ``sequence_output`` depending on head type.
        labels : torch.Tensor, optional
            Supervision tensor; shape depends on task.

        Returns
        -------
        dict
            Keys: ``logits`` and optionally ``loss`` if labels are provided.
        """
        raise NotImplementedError

    def postprocess(self, logits):
        """Default identity post-processing (override in subclasses)."""
        return logits


class SequenceClassificationHead(HeadBase):
    """Single-label classification head over pooled features using CrossEntropy."""

    def __init__(self, cfg: BaselineConfig):
        super().__init__()
        self.classifier = nn.Linear(cfg.hidden_size, cfg.num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, features, labels=None):
        """Compute logits ``(B, num_labels)`` and optional loss from pooled features."""
        logits = self.classifier(features["hidden_state"])
        loss = self.loss_fn(logits, labels.long()) if labels is not None else None
        return {"logits": logits, "loss": loss}

    def postprocess(self, logits):
        """Return probabilities via softmax."""
        return torch.argmax(torch.softmax(logits, dim=-1), dim=-1)


class MultiLabelClassificationHead(HeadBase):
    """Multi-label head over pooled features using BCEWithLogitsLoss."""

    def __init__(self, cfg: BaselineConfig):
        super().__init__()
        self.classifier = nn.Linear(cfg.hidden_size, cfg.num_labels)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, features, labels=None):
        """Compute raw logits and optional BCE-with-logits loss."""
        logits = self.classifier(features["hidden_state"])
        loss = self.loss_fn(logits, labels.float()) if labels is not None else None
        return {"logits": logits, "loss": loss}

    def postprocess(self, logits):
        """Return multi-label probabilities via sigmoid."""
        return torch.sigmoid(logits)


class RegressionHead(HeadBase):
    """Regression head over pooled features using MSELoss."""

    def __init__(self, cfg: BaselineConfig):
        super().__init__()
        self.regressor = nn.Linear(cfg.hidden_size, cfg.num_labels)
        self.loss_fn = nn.MSELoss()

    def forward(self, features, labels=None):
        """Return continuous outputs and optional MSE loss."""
        logits = self.regressor(features["hidden_state"])
        loss = (
            self.loss_fn(logits.view(-1), labels.view(-1).float())
            if labels is not None
            else None
        )
        return {"logits": logits, "loss": loss}

class TokenClassificationHead(HeadBase):
    """Per-token classification head over ``sequence_output`` using CrossEntropy."""

    def __init__(self, cfg: BaselineConfig):
        super().__init__()
        self.classifier = nn.Linear(cfg.hidden_size, cfg.num_labels)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=cfg.pad_token_id)

    def forward(self, features, labels=None):
        """Return per-token logits ``(B, L, num_labels)`` and optional loss."""
        seq_out = features["sequence_output"]
        logits = self.classifier(seq_out)
        loss = (
            self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1).long())
            if labels is not None
            else None
        )
        return {"logits": logits, "loss": loss}

    def postprocess(self, logits):
        """Return probabilities via softmax."""
        return torch.argmax(torch.softmax(logits, dim=-1), dim=-1)

class TokenRegressionHead(HeadBase):
    """Per-token regression head over ``sequence_output`` using MSELoss."""

    def __init__(self, cfg: BaselineConfig):
        super().__init__()
        self.regressor = nn.Linear(cfg.hidden_size, cfg.num_labels)
        self.loss_fn = nn.MSELoss()

    def forward(self, features, labels=None):
        """Return per-token continuous outputs and optional loss."""
        seq_out = features["sequence_output"]
        logits = self.regressor(seq_out)
        loss = (
            self.loss_fn(logits.view(-1), labels.view(-1).float())
            if labels is not None
            else None
        )
        return {"logits": logits, "loss": loss}


HEAD_REGISTRY = {
    "classification": SequenceClassificationHead,
    "multilabel_classification": MultiLabelClassificationHead,
    "regression": RegressionHead,
    "token_classification": TokenClassificationHead,
    "token_regression": TokenRegressionHead,
}


class OmniGenericBaseline(OmniModel):
    """Generic baseline model wiring a simple backbone to a selected head.

    This class provides a flexible way to create small baselines for different
    tasks by choosing among CNN/LSTM/BPNet-style backbones and head types.

    Parameters
    ----------
    tokenizer : Any
        Tokenizer instance used for vocabulary size and padding index. Must not be None.
    backbone_type : str
        One of {"cnn", "rnn", "bpnet", "deepstarr", "basenji"}.
    task_name : str
        One of {"multilabel_classification", "classification", "regression", "token_classification", "token_regression"}.
    num_labels : int
        Number of outputs for the head.
    label2id : dict, optional
        Mapping from label string to id; default builds {"0":0, ...}.

    Other keyword arguments are forwarded to the chosen backbone configuration
    (e.g., ``embed_dim``, ``hidden_dim``, ``n_filters``, etc.).
    """

    def __init__(self, tokenizer, *args, **kwargs):
        # Enforce required arguments with helpful errors
        if tokenizer is None:
            raise ValueError(
                "tokenizer is required for OmniGenericBaseline (got None). Provide a tokenizer with vocab_size/get_vocab and pad_token_id."
            )
        backbone_type = kwargs.pop("backbone_type", None)
        task_name = kwargs.pop("task_name", None)
        if backbone_type is None:
            raise ValueError(
                f"backbone_type must be provided explicitly. Choices: {sorted(list(BACKBONE_REGISTRY.keys()))}"
            )
        if task_name is None:
            raise ValueError(
                f"task_name must be provided explicitly. Choices: {sorted(list(HEAD_REGISTRY.keys()))}"
            )
        num_labels = kwargs.pop("num_labels")
        label2id = kwargs.pop("label2id", {str(i): i for i in range(num_labels)})
        pad_token_id = int(getattr(tokenizer, "pad_token_id", -100) or -100)

        class Cfg: ...

        cfg = Cfg()
        cfg.hidden_size = kwargs.get("hidden_size", 128)
        cfg.num_labels = num_labels
        cfg.label2id = label2id
        cfg.id2label = {v: k for k, v in label2id.items()}
        cfg.name_or_path = f"GenericBaseline-{backbone_type}-{task_name}"
        cfg.model_type = "baseline"
        cfg.architectures = ["OmniGenericBaseline"]
        cfg.pad_token_id = pad_token_id

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

        super().__init__(
            _Stub(cfg),
            tokenizer,
            num_labels=num_labels,
            label2id=label2id,
            *args,
            **kwargs,
        )
        vocab_size = getattr(self.tokenizer, "vocab_size", None) or len(
            self.tokenizer.get_vocab()
        )
        self.baseline_cfg = BaselineConfig(
            backbone_type=backbone_type,
            task_name=task_name,
            vocab_size=vocab_size,
            hidden_size=cfg.hidden_size,
            num_labels=num_labels,
            pad_token_id=pad_token_id,
            embed_dim=kwargs.get("embed_dim", 128),
            num_filters=kwargs.get("num_filters", 128),
            kernel_sizes=kwargs.get("kernel_sizes", (3, 5, 7)),
            dropout=kwargs.get("dropout", 0.1),
            hidden_dim=kwargs.get("hidden_dim", 256),
            num_layers=kwargs.get("num_layers", 1),
            bidirectional=kwargs.get("bidirectional", True),
            n_filters=kwargs.get("n_filters", 64),
            n_dilated_layers=kwargs.get("n_dilated_layers", 9),
            conv1_kernel_size=kwargs.get("conv1_kernel_size", 25),
            dil_kernel_size=kwargs.get("dil_kernel_size", 3),
            label2id=label2id,
            id2label={v: k for k, v in label2id.items()},
        )
        if backbone_type not in BACKBONE_REGISTRY:
            raise ValueError(
                f"Unknown backbone_type='{backbone_type}'. Valid options: {sorted(list(BACKBONE_REGISTRY.keys()))}"
            )
        backbone_cls = BACKBONE_REGISTRY[backbone_type]
        self.backbone = backbone_cls(self.baseline_cfg)
        # Setup nucleotide one-hot projection for backbones that support it
        weight = torch.zeros(vocab_size, 4)
        for i, toks in enumerate([("A", "a"), ("C", "c"), ("G", "g"), ("T", "t")]):
            for tok in toks:
                try:
                    tid = self.tokenizer.convert_tokens_to_ids(tok)
                    if tid is not None and tid >= 0 and tid < weight.size(0):
                        weight[tid, i] = 1.0
                except Exception:
                    pass
        if hasattr(self.backbone, "set_one_hot"):
            try:
                self.backbone.set_one_hot(weight)
            except Exception:
                pass
        self.baseline_cfg.hidden_size = getattr(
            self.backbone, "out_dim", self.baseline_cfg.hidden_size
        )
        self.config.hidden_size = self.baseline_cfg.hidden_size
        if task_name not in HEAD_REGISTRY:
            raise ValueError(
                f"Unknown task_name='{task_name}'. Valid options: {sorted(list(HEAD_REGISTRY.keys()))}"
            )
        head_cls = HEAD_REGISTRY[task_name]
        self.head = head_cls(self.baseline_cfg)

    def _build_config_dict(self):
        """Return a serializable dictionary for ``save_pretrained``.

        The dictionary includes both backbone hyperparameters and head/task info.
        """
        return {
            "model_type": self.config.model_type,
            "architectures": self.config.architectures,
            "num_labels": self.config.num_labels,
            "label2id": self.config.label2id,
            "id2label": self.config.id2label,
            "pad_token_id": self.config.pad_token_id,
            "hidden_size": self.config.hidden_size,
            "backbone_type": self.baseline_cfg.backbone_type,
            "task_name": self.baseline_cfg.task_name,
            "vocab_size": self.baseline_cfg.vocab_size,
            "embed_dim": self.baseline_cfg.embed_dim,
            "num_filters": self.baseline_cfg.num_filters,
            "kernel_sizes": list(self.baseline_cfg.kernel_sizes),
            "dropout": self.baseline_cfg.dropout,
            "hidden_dim": self.baseline_cfg.hidden_dim,
            "num_layers": self.baseline_cfg.num_layers,
            "bidirectional": self.baseline_cfg.bidirectional,
            "n_filters": self.baseline_cfg.n_filters,
            "n_dilated_layers": self.baseline_cfg.n_dilated_layers,
            "conv1_kernel_size": self.baseline_cfg.conv1_kernel_size,
            "dil_kernel_size": self.baseline_cfg.dil_kernel_size,
            "model_cls": self.__class__.__name__,
            "library_name": "OMNIGENBENCH",
        }

    def save_pretrained(self, save_directory: str, overwrite: bool = True):
        """Save model weights, config, tokenizer and metadata to a directory.

        Files
        -----
        - ``config.json``: Backbone/head configuration.
        - ``pytorch_model.bin``: Model weights.
        - ``tokenizer``: Saved via tokenizer's ``save_pretrained`` when available.
        - ``metadata.json``: Lightweight metadata (class and library).
        """
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
    def from_pretrained(
        cls, save_directory: str, tokenizer=None, map_location=None, **kwargs
    ):
        """Load model, config, and tokenizer from a directory."""
        with open(
            os.path.join(save_directory, "config.json"), "r", encoding="utf8"
        ) as f:
            cfg_dict = json.load(f)
        if tokenizer is None:
            if os.path.exists(
                os.path.join(save_directory, "tokenizer_config.json")
            ) or os.path.exists(os.path.join(save_directory, "vocab.json")):
                try:
                    from transformers import AutoTokenizer

                    tokenizer = AutoTokenizer.from_pretrained(save_directory)
                except Exception:
                    tokenizer = None
            if tokenizer is None and os.path.exists(
                os.path.join(save_directory, "tokenizer.bin")
            ):
                with open(os.path.join(save_directory, "tokenizer.bin"), "rb") as f:
                    tokenizer = torch.load(f, map_location=map_location)
            if tokenizer is None:
                raise ValueError("Tokenizer could not be loaded; please provide one.")
        # Backward-compat: warn if backbone/task missing; fall back to defaults
        if "backbone_type" not in cfg_dict:
            warnings.warn(
                "Missing 'backbone_type' in saved config; defaulting to 'cnn'. Please resave the model to include this field.",
                RuntimeWarning,
            )
        if "task_name" not in cfg_dict:
            warnings.warn(
                "Missing 'task_name' in saved config; defaulting to 'multilabel_classification'. Please resave the model to include this field.",
                RuntimeWarning,
            )
        model = cls(
            tokenizer,
            backbone_type=cfg_dict.get("backbone_type", "cnn"),
            task_name=cfg_dict.get("task_name", "multilabel_classification"),
            num_labels=cfg_dict["num_labels"],
            embed_dim=cfg_dict.get("embed_dim", 128),
            num_filters=cfg_dict.get("num_filters", 128),
            kernel_sizes=tuple(cfg_dict.get("kernel_sizes", (3, 5, 7))),
            dropout=cfg_dict.get("dropout", 0.1),
            hidden_dim=cfg_dict.get("hidden_dim", 256),
            num_layers=cfg_dict.get("num_layers", 1),
            bidirectional=cfg_dict.get("bidirectional", True),
            n_filters=cfg_dict.get("n_filters", 64),
            n_dilated_layers=cfg_dict.get("n_dilated_layers", 9),
            conv1_kernel_size=cfg_dict.get("conv1_kernel_size", 25),
            dil_kernel_size=cfg_dict.get("dil_kernel_size", 3),
            label2id=cfg_dict.get("label2id"),
        )
        state_dict = torch.load(
            os.path.join(save_directory, "pytorch_model.bin"),
            map_location=map_location or "cpu",
        )
        model.load_state_dict(state_dict, strict=False)
        meta_path = os.path.join(save_directory, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf8") as f:
                model.metadata = json.load(f)
        return model

    def forward(self, **inputs):
        """Forward pass through backbone and task head.

        Returns
        -------
        dict
            Always includes ``logits`` and passthrough ``labels`` if present.
            Includes ``last_hidden_state`` and optionally ``sequence_output`` when
            the backbone provides it (e.g., for token-level heads).
        """
        labels = inputs.get("labels")
        feats = self.backbone(inputs["input_ids"], inputs.get("attention_mask"))
        head_out = self.head(feats, labels=labels)
        out = {
            "logits": head_out["logits"],
            "labels": labels,
            "last_hidden_state": feats.get("hidden_state"),
        }
        if "sequence_output" in feats:
            out["sequence_output"] = feats["sequence_output"]
        if head_out.get("loss") is not None:
            out["loss"] = head_out["loss"]
        return out

    def predict(self, sequence_or_inputs, **kwargs):
        """Return task-appropriate probabilities from head postprocess."""
        out = self._forward_from_raw_input(sequence_or_inputs, **kwargs)
        logits = out["logits"]
        probs = self.head.postprocess(logits)
        return {
            "predictions": probs,
            "logits": logits,
            "last_hidden_state": out.get("last_hidden_state"),
        }

    def inference(self, sequence_or_inputs, threshold: float = 0.5, **kwargs):
        """Convenience inference wrapper producing final predictions.

        Behavior depends on ``task_name``:
        - ``multilabel_classification``: thresholded sigmoid probabilities.
        - ``classification``: argmax over softmax probabilities.
        - others: returns post-processed outputs.
        """
        out = self._forward_from_raw_input(sequence_or_inputs, **kwargs)
        logits = out["logits"]
        probs = self.head.postprocess(logits)
        if self.baseline_cfg.task_name == "multilabel_classification":
            preds = (probs >= threshold).to(torch.int)
        elif self.baseline_cfg.task_name == "classification":
            preds = probs.argmax(dim=-1)
        else:
            preds = probs
        if not isinstance(sequence_or_inputs, list):
            return {
                "predictions": preds[0].cpu(),
                "logits": logits[0].cpu(),
                "probabilities": probs[0].cpu(),
                "last_hidden_state": out.get("last_hidden_state")[0].cpu()
                if out.get("last_hidden_state") is not None
                else None,
            }
        return {
            "predictions": preds.cpu(),
            "logits": logits.cpu(),
            "probabilities": probs.cpu(),
            "last_hidden_state": out.get("last_hidden_state"),
        }

    def loss_function(self, logits, labels):
        """Compute task-appropriate training loss."""
        if self.baseline_cfg.task_name == "multilabel_classification":
            return nn.BCEWithLogitsLoss()(logits, labels.float())
        if self.baseline_cfg.task_name == "classification":
            return nn.CrossEntropyLoss()(logits, labels.long())
        if self.baseline_cfg.task_name == "regression":
            return nn.MSELoss()(logits, labels.view(-1).float())
        # Token-level tasks default
        return nn.MSELoss()(logits.view(-1), labels.view(-1).float())


def create_baseline(
    tokenizer,
    *,
    backbone_type: str,
    task_name: str,
    num_labels: int,
    label2id: Optional[dict] = None,
    **kwargs,
):
    """Factory for building baselines via OmniGenericBaseline.

    Example:
        model = create_baseline(
            tokenizer,
            backbone_type="deepstarr",
            task_name="multilabel_classification",
            num_labels=8,
        )
    """
    return OmniGenericBaseline(
        tokenizer,
        backbone_type=backbone_type,
        task_name=task_name,
        num_labels=num_labels,
        label2id=label2id,
        **kwargs,
    )


# ---------------- Base Backbones for OmniModel Wrappers -----------------
class _BaseConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def _build_label_maps(num_labels, label2id=None):
    """Build default label mappings when not provided.

    Returns
    -------
    tuple(dict, dict)
        ``(label2id, id2label)`` consistent pair.
    """
    if label2id is None:
        label2id = {str(i): i for i in range(num_labels)}
        id2label = {v: k for k, v in label2id.items()}
        return label2id, id2label
    return label2id, {v: k for k, v in label2id.items()}


class OmniCNNBaseModel(nn.Module):
    """Lightweight CNN backbone returning per-token hidden states, to be wrapped by OmniModelFor* wrappers.

    Forward returns a dictionary with ``last_hidden_state`` of shape ``(B, L, H)``.
    """

    def __init__(
        self,
        vocab_size,
        embed_dim=128,
        num_filters=128,
        kernel_sizes=(3, 5, 7),
        dropout=0.1,
        pad_token_id=0,
        num_labels=2,
        label2id=None,
        name_or_path="cnn-backbone",
    ):
        super().__init__()
        label2id, id2label = _build_label_maps(num_labels, label2id)
        # device tracker buffer for OmniModel compatibility
        self.register_buffer("_dev_tracker", torch.empty(0), persistent=False)
        self.config = _BaseConfig(
            hidden_size=num_filters * len(kernel_sizes),
            embed_dim=embed_dim,
            num_filters=num_filters,
            kernel_sizes=list(kernel_sizes),
            dropout=dropout,
            pad_token_id=pad_token_id,
            vocab_size=vocab_size,
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label,
            name_or_path=name_or_path,
            model_type="cnn",
            architectures=["OmniCNNBaseModel"],
        )
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(embed_dim, num_filters, k, padding=k // 2), nn.ReLU()
                )
                for k in kernel_sizes
            ]
        )
        self.dropout = nn.Dropout(dropout)

    @property
    def device(self):
        return self._dev_tracker.device

    @property
    def dtype(self):
        return self._dev_tracker.dtype

    def forward(self, input_ids, attention_mask=None, **kwargs):
        """Return token-level features only.

        Parameters
        ----------
        input_ids : torch.LongTensor
            Shape ``(B, L)``.
        attention_mask : torch.LongTensor, optional
            Unused here; kept for compatibility.
        """
        x = self.embedding(input_ids)
        x = self.dropout(x)
        feats = [conv(x.transpose(1, 2)) for conv in self.convs]
        feats = torch.cat(feats, dim=1).transpose(1, 2)
        return {"last_hidden_state": feats}

    def save_pretrained(self, save_directory, **kwargs):
        """Save backbone weights and config to ``save_directory``."""
        os.makedirs(save_directory, exist_ok=True)
        cfg_dict = {
            k: getattr(self.config, k)
            for k in self.config.__dict__
            if not k.startswith("_")
        }
        cfg_dict.update(
            {"model_cls": self.__class__.__name__, "library_name": "OMNIGENBENCH"}
        )
        with open(
            os.path.join(save_directory, "config.json"), "w", encoding="utf8"
        ) as f:
            json.dump(cfg_dict, f, ensure_ascii=False, indent=2)
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        with open(
            os.path.join(save_directory, "metadata.json"), "w", encoding="utf8"
        ) as f:
            json.dump(
                {"model_cls": self.__class__.__name__, "library_name": "OMNIGENBENCH"},
                f,
                indent=2,
            )

    @classmethod
    def from_pretrained(cls, save_directory, map_location=None, **kwargs):
        """Load backbone from a directory containing ``config.json`` and weights."""
        with open(
            os.path.join(save_directory, "config.json"), "r", encoding="utf8"
        ) as f:
            cfg = json.load(f)
        model = cls(
            vocab_size=cfg["vocab_size"],
            embed_dim=cfg.get("embed_dim", 128),
            num_filters=cfg.get("num_filters", 128),
            kernel_sizes=tuple(cfg.get("kernel_sizes", (3, 5, 7))),
            dropout=cfg.get("dropout", 0.1),
            pad_token_id=cfg.get("pad_token_id", 0),
            num_labels=cfg.get("num_labels", 2),
            label2id=cfg.get("label2id"),
            name_or_path=cfg.get("name_or_path", "cnn-backbone"),
        )
        state = torch.load(
            os.path.join(save_directory, "pytorch_model.bin"),
            map_location=map_location or "cpu",
        )
        model.load_state_dict(state, strict=False)
        return model


class OmniRNNBaseModel(nn.Module):
    """BiLSTM backbone returning per-token hidden states for OmniModel wrappers."""

    def __init__(
        self,
        vocab_size,
        embed_dim=128,
        hidden_dim=256,
        num_layers=1,
        bidirectional=True,
        dropout=0.1,
        pad_token_id=0,
        num_labels=2,
        label2id=None,
        name_or_path="rnn-backbone",
    ):
        super().__init__()
        label2id, id2label = _build_label_maps(num_labels, label2id)
        self.register_buffer("_dev_tracker", torch.empty(0), persistent=False)
        self.config = _BaseConfig(
            hidden_size=hidden_dim * (2 if bidirectional else 1),
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            pad_token_id=pad_token_id,
            vocab_size=vocab_size,
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label,
            name_or_path=name_or_path,
            model_type="rnn",
            architectures=["OmniRNNBaseModel"],
        )
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0 if num_layers == 1 else dropout,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(dropout)

    @property
    def device(self):
        return self._dev_tracker.device

    @property
    def dtype(self):
        return self._dev_tracker.dtype

    def forward(self, input_ids, attention_mask=None, **kwargs):
        """Return token-level hidden states only: ``{"last_hidden_state": (B, L, H)}``."""
        x = self.embedding(input_ids)
        x = self.dropout(x)
        seq_out, _ = self.lstm(x)
        return {"last_hidden_state": seq_out}

    def save_pretrained(self, save_directory, **kwargs):
        """Save backbone weights and config to a directory."""
        os.makedirs(save_directory, exist_ok=True)
        cfg_dict = {
            k: getattr(self.config, k)
            for k in self.config.__dict__
            if not k.startswith("_")
        }
        cfg_dict.update(
            {"model_cls": self.__class__.__name__, "library_name": "OMNIGENBENCH"}
        )
        with open(
            os.path.join(save_directory, "config.json"), "w", encoding="utf8"
        ) as f:
            json.dump(cfg_dict, f, ensure_ascii=False, indent=2)
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        with open(
            os.path.join(save_directory, "metadata.json"), "w", encoding="utf8"
        ) as f:
            json.dump(
                {"model_cls": self.__class__.__name__, "library_name": "OMNIGENBENCH"},
                f,
                indent=2,
            )

    @classmethod
    def from_pretrained(cls, save_directory, map_location=None, **kwargs):
        """Load backbone from a directory containing ``config.json`` and weights."""
        with open(
            os.path.join(save_directory, "config.json"), "r", encoding="utf8"
        ) as f:
            cfg = json.load(f)
        model = cls(
            vocab_size=cfg["vocab_size"],
            embed_dim=cfg.get("embed_dim", 128),
            hidden_dim=cfg.get("hidden_dim", 256),
            num_layers=cfg.get("num_layers", 1),
            bidirectional=cfg.get("bidirectional", True),
            dropout=cfg.get("dropout", 0.1),
            pad_token_id=cfg.get("pad_token_id", 0),
            num_labels=cfg.get("num_labels", 2),
            label2id=cfg.get("label2id"),
            name_or_path=cfg.get("name_or_path", "rnn-backbone"),
        )
        state = torch.load(
            os.path.join(save_directory, "pytorch_model.bin"),
            map_location=map_location or "cpu",
        )
        model.load_state_dict(state, strict=False)
        return model


class OmniBPNetBaseModel(nn.Module):
    """BPNet-style dilated CNN backbone that returns per-token hidden states.

    This lightweight base model mirrors OmniCNNBaseModel/OmniRNNBaseModel patterns and
    is suitable for wrapping by OmniModelFor* tasks that expect
    ``{"last_hidden_state": (B, L, H)}``.

    Parameters
    ----------
    vocab_size : int
        Vocabulary size for the token-to-one-hot projection.
    n_filters : int, optional
        Number of channels in convolution blocks, by default 64.
    n_dilated_layers : int, optional
        Number of dilated residual layers, by default 9.
    conv1_kernel_size : int, optional
        Kernel size for the first conv, by default 25.
    dil_kernel_size : int, optional
        Kernel size for dilated convs, by default 3.
    dropout : float, optional
        Dropout prob applied to features, by default 0.1.
    pad_token_id : int, optional
        Padding token id, by default 0.
    num_labels : int, optional
        Unused here; included for config parity, by default 2.
    label2id : dict, optional
        Optional label map stored in config.
    name_or_path : str, optional
        Model name for config metadata, by default "bpnet-backbone".
    """

    def __init__(
        self,
        vocab_size,
        n_filters=64,
        n_dilated_layers=9,
        conv1_kernel_size=25,
        dil_kernel_size=3,
        dropout=0.1,
        pad_token_id=0,
        num_labels=2,
        label2id=None,
        name_or_path="bpnet-backbone",
    ):
        super().__init__()
        label2id, id2label = _build_label_maps(num_labels, label2id)
        self.register_buffer("_dev_tracker", torch.empty(0), persistent=False)
        self.config = _BaseConfig(
            hidden_size=n_filters,
            n_filters=n_filters,
            n_dilated_layers=n_dilated_layers,
            conv1_kernel_size=conv1_kernel_size,
            dil_kernel_size=dil_kernel_size,
            dropout=dropout,
            pad_token_id=pad_token_id,
            vocab_size=vocab_size,
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label,
            name_or_path=name_or_path,
            model_type="bpnet",
            architectures=["OmniBPNetBaseModel"],
        )
        self.register_buffer(
            "_one_hot_weight", torch.zeros(vocab_size, 4), persistent=False
        )
        self.conv1 = nn.Conv1d(4, n_filters, conv1_kernel_size, padding="same")
        self.dilated_convs = nn.ModuleList(
            [
                nn.Conv1d(
                    n_filters, n_filters, dil_kernel_size, padding="same", dilation=2**i
                )
                for i in range(n_dilated_layers)
            ]
        )
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.out_dim = n_filters
        self.dropout = nn.Dropout(dropout)

    @property
    def device(self):
        return self._dev_tracker.device

    @property
    def dtype(self):
        return self._dev_tracker.dtype

    def set_one_hot(self, weight: torch.Tensor):
        """Optionally set the id->one-hot projection matrix of shape (vocab_size, 4)."""
        if (
            isinstance(weight, torch.Tensor)
            and weight.shape == self._one_hot_weight.shape
        ):
            self._one_hot_weight = weight

    def _tokens_to_one_hot(self, ids: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.embedding(ids, self._one_hot_weight).permute(0, 2, 1)

    def forward(self, input_ids, attention_mask=None, **inputs):
        """Return per-token hidden states only.

        Returns
        -------
        dict
            ``{"last_hidden_state": (batch, seq_len, hidden_size)}``.
        """
        x = self._tokens_to_one_hot(input_ids)  # [B,4,L]
        x = torch.relu(self.conv1(x))
        for layer in self.dilated_convs:
            residual = x
            x = torch.relu(layer(x))
            x = x + residual
        seq_out = x.transpose(1, 2)  # (B, L, C)
        seq_out = self.dropout(seq_out)
        return {"last_hidden_state": seq_out}

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
            "n_filters": getattr(self.config, "n_filters", None),
            "n_dilated_layers": getattr(self.config, "n_dilated_layers", None),
            "conv1_kernel_size": getattr(self.config, "conv1_kernel_size", None),
            "dil_kernel_size": getattr(self.config, "dil_kernel_size", None),
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
            n_filters=cfg_dict.get("n_filters", 64),
            n_dilated_layers=cfg_dict.get("n_dilated_layers", 9),
            conv1_kernel_size=cfg_dict.get("conv1_kernel_size", 25),
            dil_kernel_size=cfg_dict.get("dil_kernel_size", 3),
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
