# -*- coding: utf-8 -*-
# file: abstract_model.py
# time: 18:36 06/04/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2025. All Rights Reserved.
import json
import os
import shutil
import time
import warnings
import inspect
from importlib import import_module

import dill
import findfile
import torch
from transformers import AutoModel, AutoConfig, AutoTokenizer, BatchEncoding

from ..misc.utils import fprint, env_meta_info
from .embedding_mixin import EmbeddingMixin

warnings.filterwarnings("once")


def count_parameters(model):
    """
    This function iterates through all parameters of a PyTorch model and counts
    only those that require gradients (i.e., trainable parameters).

    Args:
        model (torch.nn.Module): A PyTorch model.

    Returns:
        int: The total number of trainable parameters.

    Example:
        >>> model = OmniModelForSequenceClassification(config, tokenizer)
        >>> num_params = count_parameters(model)
        >>> print(f"Model has {num_params} trainable parameters")
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class OmniModel(EmbeddingMixin, torch.nn.Module):
    """
    Abstract base class providing a unified interface for genomic foundation models in the
    OmniGenBench framework. This class handles model initialization, forward passes, loss
    computation, prediction interfaces, and model persistence while maintaining compatibility
    with HuggingFace's ecosystem.

    **Architectural Pattern**: This class follows the Template Method pattern, providing
    common infrastructure while delegating task-specific behavior to subclasses
    (OmniModelForSequenceClassification, OmniModelForTokenClassification, etc.).

    **Inherited Capabilities** (via EmbeddingMixin):

    - **Embedding Generation**: ``batch_encode()``, ``encode()``, ``encode_tokens()``
      for extracting fixed-length sequence representations from genomic sequences
    - **Attention Extraction**: ``extract_attention_scores()``, ``batch_extract_attention_scores()``
      for model interpretability and attention weight visualization
    - **Similarity Computation**: ``compute_similarity()`` for sequence comparison and
      relatedness analysis
    - **Visualization Tools**: ``visualize_attention_pattern()`` for generating attention heatmaps
      and understanding model focus

    All task-specific OmniModel subclasses automatically inherit these capabilities,
    enabling representation learning and interpretability without additional implementation.

    **Design Philosophy**: By inheriting from both EmbeddingMixin and torch.nn.Module,
    this class seamlessly integrates sequence embedding capabilities with PyTorch's
    standard training infrastructure, making it compatible with native PyTorch training
    loops, HuggingFace Trainer, and Accelerate-based distributed training.

    **Task-Specific Subclasses**: Users should instantiate concrete implementations rather
    than this abstract class directly:

    - ``OmniModelForSequenceClassification``: Sequence-level classification tasks
      (e.g., promoter identification, functional annotation)
    - ``OmniModelForMultiLabelSequenceClassification``: Multi-label classification
      (e.g., transcription factor binding site prediction with 919 TFs)
    - ``OmniModelForTokenClassification``: Per-nucleotide predictions
      (e.g., splice site detection, secondary structure annotation)
    - ``OmniModelForSequenceRegression``: Sequence-level continuous predictions
      (e.g., gene expression levels, binding affinity scores)
    - ``OmniModelForTokenRegression``: Per-nucleotide continuous predictions
      (e.g., chromatin accessibility profiles, conservation scores)
    - ``OmniModelForRNADesign``: Structure-guided RNA sequence generation
      (genetic algorithm + masked language model)
    - ``OmniModelForEmbedding``: Representation learning and feature extraction
    """

    def __init__(self, config_or_model, tokenizer, *args, **kwargs):
        """
        Initializes the genomic foundation model with flexible input types.

        This method handles three initialization patterns:

        1. **From pre-trained path** (recommended): Loads model from HuggingFace Hub or local
           directory. The architecture is automatically detected via ``config.json`` using
           the ``auto_map`` or ``architectures`` fields.

        2. **From PyTorch module**: Wraps an existing ``nn.Module`` with OmniModel interface,
           useful for integrating custom architectures or models loaded via other means.

        3. **From configuration**: Initializes a new model from AutoConfig specification,
           typically used for training models from scratch.

        The initialization process automatically detects the underlying architecture via
        HuggingFace's ``config.json`` (using ``auto_map`` or ``architectures`` fields),
        eliminating manual architecture specification for standard models.

        Args:
            config_or_model: One of the following:
                - **str**: Path or HuggingFace Hub identifier (e.g., "yangheng/OmniGenome-186M").
                  Can be a local path to a model directory or a Hub model ID.
                - **torch.nn.Module**: Pre-instantiated PyTorch model to wrap with OmniModel
                  interface, enabling use of custom architectures within the framework.
                - **AutoConfig**: Configuration object for new model initialization, used when
                  training models from scratch or with custom configurations.

            tokenizer: Tokenizer instance compatible with the model architecture.
                Used for sequence preprocessing during inference. Should implement either
                OmniTokenizer interface or HuggingFace tokenizer protocol.

            *args: Additional positional arguments passed to torch.nn.Module.__init__

            **kwargs: Additional keyword arguments:
                - **label2id** (dict, optional): Mapping from class labels to integer IDs.
                  Required for classification tasks. Example: {"negative": 0, "positive": 1}.
                  Either this or num_labels must be provided.

                - **num_labels** (int, optional): Number of output classes. Alternative to label2id
                  for when label names are not available. If both provided, they must be consistent
                  (len(label2id) must equal num_labels).

                - **trust_remote_code** (bool, optional): Whether to trust remote code when loading
                  from HuggingFace Hub. Defaults to True. Set to False for security-critical
                  environments where only vetted models should be loaded.

                - **ignore_mismatched_sizes** (bool, optional): Whether to ignore size mismatches
                  when loading pre-trained weights (e.g., different classifier head dimensions).
                  Defaults to False. Set to True when fine-tuning for a different number of
                  labels than the pre-trained model.

                - **dropout** (float, optional): Dropout probability for regularization in
                  classification/regression heads. Defaults to 0.0. Typical values: 0.1-0.5.

                - **dataset_class** (type, optional): Dataset class used for preprocessing.
                  Enables models to use the dataset's ``prepare_input`` method during inference,
                  allowing custom field handling beyond basic tokenization. Useful when inference
                  requires the same complex preprocessing as training.

                - **problem_type** (str, optional): Type of prediction problem. Common values:
                  "single_label_classification", "multi_label_classification", "regression".
                  Affects loss calculation and output interpretation.

        Raises:
            ValueError: If neither label2id nor num_labels is provided, or if they are
                inconsistent (len(label2id) != num_labels). Also raised if config_or_model
                is an unsupported type (not str, nn.Module, or AutoConfig).

            RuntimeError: If the hidden size cannot be determined from the config (model must
                define one of: hidden_size, n_embd, or d_model), or if the model architecture
                cannot be auto-detected from config.json (missing both architectures and auto_map).

            FileNotFoundError: If the specified model path does not exist locally and cannot be
                found on HuggingFace Hub. Check model path/ID spelling and internet connectivity.

        Example:
            >>> # Pattern 1: Initialize from pre-trained model (recommended)
            >>> from omnigenbench import OmniModelForSequenceClassification, OmniTokenizer
            >>> tokenizer = OmniTokenizer.from_pretrained("yangheng/OmniGenome-186M")
            >>> model = OmniModelForSequenceClassification(
            ...     "yangheng/OmniGenome-186M",
            ...     tokenizer=tokenizer,
            ...     num_labels=2,
            ...     problem_type="single_label_classification"
            ... )
            >>> print(f"Model has {count_parameters(model):,} trainable parameters")

            >>> # Pattern 2: Initialize with label2id mapping
            >>> label2id = {"background": 0, "promoter": 1, "enhancer": 2}
            >>> model = OmniModelForSequenceClassification(
            ...     "yangheng/OmniGenome-186M",
            ...     tokenizer=tokenizer,
            ...     label2id=label2id  # num_labels inferred automatically as 3
            ... )

            >>> # Pattern 3: Initialize from configuration (for custom models)
            >>> from transformers import AutoConfig
            >>> config = AutoConfig.from_pretrained("yangheng/OmniGenome-186M")
            >>> config.num_labels = 10
            >>> model = OmniModelForSequenceClassification(config, tokenizer)

            >>> # Pattern 4: Wrap existing PyTorch module
            >>> from transformers import AutoModel
            >>> base_model = AutoModel.from_pretrained("yangheng/OmniGenome-186M")
            >>> model = OmniModelForSequenceClassification(
            ...     base_model, tokenizer, num_labels=2
            ... )

            >>> # Pattern 5: Initialize with dataset class for complex preprocessing
            >>> from omnigenbench import OmniDatasetForSequenceClassification
            >>> model = OmniModelForSequenceClassification(
            ...     "yangheng/OmniGenome-186M",
            ...     tokenizer=tokenizer,
            ...     num_labels=2,
            ...     dataset_class=OmniDatasetForSequenceClassification
            ... )
            >>> # Now model.inference() can use dataset's prepare_input method
        """
        self.loss_fn = None

        label2id = kwargs.pop("label2id", None)
        trust_remote_code = kwargs.pop("trust_remote_code", True)
        num_labels = kwargs.pop("num_labels", len(label2id) if label2id else None)
        ignore_mismatched_sizes = kwargs.pop("ignore_mismatched_sizes", False)
        dataset_class = kwargs.pop("dataset_class", None)

        if label2id and not num_labels:
            num_labels = len(label2id)
        elif num_labels and not label2id:
            label2id = {str(i): i for i in range(num_labels)}
        elif not label2id and not num_labels:
            raise ValueError(
                "Either label2id or num_labels must be provided to initialize the model."
            )
        else:
            if len(label2id) != num_labels:
                raise ValueError(
                    "The length of label2id does not match num_labels. "
                    f"Expected {num_labels}, but got {len(label2id)}."
                )

        # do not change the order of the following lines
        super().__init__(*args, **kwargs)

        if isinstance(config_or_model, str):
            config = AutoConfig.from_pretrained(
                config_or_model,
                num_labels=num_labels,
                label2id=label2id,
                trust_remote_code=trust_remote_code,
            )
            # Load the model from either `architectures` or `auto_map`
            if hasattr(config, "auto_map") and config.auto_map:
                architectures = list(set(config.auto_map.keys()) - set(["AutoConfig"]))
                if architectures:
                    model_cls_name = (
                        "AutoModel"
                        if "AutoModel" in architectures
                        else architectures[-1]
                    )
                    if "multimolecule" in config_or_model.__repr__().lower():
                        model_cls = getattr(
                            import_module(f"multimolecule"), model_cls_name
                        )
                    else:
                        model_cls = getattr(
                            import_module(f"transformers"), model_cls_name
                        )

                    model = model_cls.from_pretrained(
                        config_or_model,
                        config=config,
                        trust_remote_code=trust_remote_code,
                        ignore_mismatched_sizes=ignore_mismatched_sizes,
                    ).base_model
                else:
                    raise ValueError(
                        f"Model cannot be instantiated from '{config_or_model}'. "
                        f"The configuration must contain either 'architectures' or 'auto_map' field. "
                        f"Please verify the model path/ID is correct and config.json is properly formatted."
                    )
            elif hasattr(config, "architectures") and config.architectures:
                model_cls_name = (
                    AutoModel
                    if "AutoModel" in config.architectures
                    else config.architectures[-1]
                )
                if hasattr(import_module(f"multimolecule"), model_cls_name):
                    model_cls = getattr(import_module(f"multimolecule"), model_cls_name)
                elif hasattr(import_module(f"transformers"), model_cls_name):
                    model_cls = getattr(import_module(f"transformers"), model_cls_name)
                else:
                    raise ValueError(
                        f"Model class '{model_cls_name}' not found in transformers or multimolecule libraries."
                    )
                model = model_cls.from_pretrained(
                    config_or_model,
                    config=config,
                    trust_remote_code=trust_remote_code,
                    ignore_mismatched_sizes=ignore_mismatched_sizes,
                ).base_model
            else:
                raise ValueError(
                    f"Model configuration from '{config_or_model}' is missing both 'architectures' and 'auto_map' fields. "
                    f"Cannot determine the model architecture. Please ensure the model has a valid config.json file."
                )
            self.model = model
            self.model.config = config
            del model_cls
        elif isinstance(config_or_model, torch.nn.Module):
            self.model = config_or_model
            self.model.config.num_labels = (
                num_labels if len(label2id) == num_labels else len(label2id)
            )
            self.model.config.label2id = label2id
        elif isinstance(config_or_model, AutoConfig):
            config = config_or_model
            config.num_labels = (
                num_labels if len(label2id) == num_labels else len(label2id)
            )
            config.label2id = label2id
            self.model = AutoModel.from_config(config)
            self.model.config = config
        else:
            raise ValueError(
                f"Invalid type for config_or_model: {type(config_or_model).__name__}. "
                f"Expected one of: str (model path/ID), torch.nn.Module (model instance), "
                f"or AutoConfig (configuration object)."
            )

        # Update the config
        self.config = self.model.config
        if isinstance(label2id, dict):
            self.config.label2id = label2id
            self.config.id2label = {v: k for k, v in label2id.items()}
        if (
            not hasattr(self.config, "num_labels")
            or len(self.config.id2label) != self.config.num_labels
        ):
            fprint(
                "Warning: The number of labels in the config is not equal to the number of labels in the label2id dictionary. "
            )
            fprint(
                "Please check the label2id dictionary and the num_labels parameter in the config."
            )
            self.config.num_labels = len(self.config.id2label)

        assert (
            len(self.config.label2id) == num_labels
        ), f"Expected {num_labels} labels, but got {len(self.config.label2id)} in label2id dictionary."

        # The metadata of the model
        self.metadata = env_meta_info()
        self.metadata["model_cls"] = self.__class__.__name__

        # Store dataset class for data preprocessing during inference
        if dataset_class is not None:
            self.dataset_class = dataset_class
            self.metadata["dataset_cls"] = dataset_class.__name__
            self.metadata["dataset_module"] = dataset_class.__module__
        elif hasattr(self, "dataset_class"):
            dataset_class = self.dataset_class
            self.metadata["dataset_cls"] = dataset_class.__name__
            self.metadata["dataset_module"] = dataset_class.__module__
        else:
            self.dataset_class = None
            self.metadata["dataset_cls"] = None
            self.metadata["dataset_module"] = None
            fprint(
                "Warning: No dataset_class is provided for the model, please set 'dataset_class=...' "
                "when initializing the model if you want to use the dataset's prepare_input method during inference."
            )

        # The config of the model
        if hasattr(self.config, "n_embd") and self.config.n_embd:
            self.config.hidden_size = self.config.n_embd
        elif hasattr(self.config, "d_model") and self.config.d_model:
            self.config.hidden_size = self.config.d_model
        elif hasattr(self.config, "hidden_size") and self.config.hidden_size:
            self.config.hidden_size = self.config.hidden_size
        else:
            raise RuntimeError(
                "The hidden size of the model is not found in the config."
            )

        # The tokenizer of the model
        self.tokenizer = tokenizer
        self.metadata["tokenizer_cls"] = self.tokenizer.__class__.__name__
        if hasattr(self.tokenizer, "base_tokenizer"):
            self.pad_token_id = self.tokenizer.base_tokenizer.pad_token_id
        else:
            self.pad_token_id = self.tokenizer.pad_token_id

        self.dropout = torch.nn.Dropout(kwargs.get("dropout", 0.0))
        self.activation = torch.nn.Tanh()

        # Device management: track device but don't move yet (subclass layers not created)
        # Let device movement happen explicitly via .to() or automatically during forward pass
        try:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        except Exception:
            self._device = torch.device("cpu")

    def last_hidden_state_forward(self, **inputs):
        """
        Performs a forward pass to get the last hidden state from the base model. It also handles compatibility with different
        model architectures by mapping input parameters appropriately.

        Args:
            **inputs: The inputs to the model, compatible with the base model's
                     forward method. Typically includes 'input_ids', 'attention_mask',
                     and other model-specific parameters.

        Returns:
            torch.Tensor: The last hidden state tensor.

        Example:
            >>> inputs = {
            ...     'input_ids': torch.tensor([[1, 2, 3, 4]]),
            ...     'attention_mask': torch.tensor([[1, 1, 1, 1]])
            ... }
            >>> hidden_states = model.last_hidden_state_forward(**inputs)
        """
        model = self.model
        input_mapping = {}
        inputs["output_hidden_states"] = True

        if "strippedhyena" in model.__class__.__name__.lower():
            inputs["x"] = inputs["input_ids"]  # For compatibility with Evo models
        if isinstance(inputs, BatchEncoding) or isinstance(inputs, dict):
            # Determine the input parameter names of the model's forward method
            forward_params = inspect.signature(model.forward).parameters
            # Map the inputs to the forward method parameters
            for param in forward_params:
                if param in inputs:
                    input_mapping[param] = inputs[param]
            # 对于未在模型签名中声明的关键参数，可以给出警告或日志
            ignored_keys = set(inputs.keys()) - set(input_mapping.keys())
            if ignored_keys:
                warnings.warn(f"Warning: Ignored keys in inputs: {ignored_keys}")

            inputs = input_mapping
        elif isinstance(inputs, tuple):
            input_ids = inputs[0]
            attention_mask = inputs[1] if len(inputs) > 1 else None
            inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        elif isinstance(inputs, torch.Tensor):
            shape = inputs.shape
            try:
                if len(shape) == 3:
                    if shape[1] == 2:
                        input_ids = inputs[:, 0]
                        attention_mask = inputs[:, 1]
                    else:
                        input_ids = inputs[0]
                        attention_mask = inputs[1] if len(inputs) > 1 else None
                elif len(shape) == 2:
                    input_ids = inputs
                    attention_mask = None
                else:
                    raise ValueError(
                        f"Failed to get the input_ids and attention_mask from the inputs, got shape {shape}."
                    )
            except:
                raise ValueError(
                    f"Failed to get the input_ids and attention_mask from the inputs, got shape {shape}."
                )
            inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        else:
            raise ValueError(
                f"The inputs should be a tuple, BatchEncoding or a dictionary-like object, got {type(inputs)}."
            )

        # 执行模型
        outputs = model(**inputs)

        if not hasattr(outputs, "last_hidden_state"):
            warnings.warn(
                f"last_hidden_state not found in the outputs from the {model.__class__.__name__} model."
            )

        if hasattr(outputs, "last_hidden_state"):
            last_hidden_state = outputs.last_hidden_state
        elif isinstance(outputs, dict) and "last_hidden_state" in outputs:
            last_hidden_state = outputs["last_hidden_state"]
        elif hasattr(outputs, "hidden_states"):
            last_hidden_state = outputs.hidden_states[-1]
        elif isinstance(outputs, (list, tuple, torch.Tensor)):
            if len(outputs) <= 2:
                # For Evo models that return a tuple of (last_hidden_state, logits)
                last_hidden_state = outputs[0]
            elif len(outputs) >= 3:
                last_hidden_state = outputs[-1]
        else:
            raise ValueError(
                f"Cannot find the last hidden state in the outputs from the {model.__class__.__name__} model, "
                f"please check the model architecture."
            )

        return last_hidden_state

    def loss_function(self, logits, labels):
        """
        Calculates the loss. This method should be implemented by concrete model classes to define
        how the loss is calculated for their specific task (classification,
        regression, etc.).

        Args:
            logits (torch.Tensor): The model's output logits.
            labels (torch.Tensor): The ground truth labels.

        Returns:
            torch.Tensor: The calculated loss.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.

        Example:
            >>> # In a classification model
            >>> loss = model.loss_function(logits, labels)
        """
        raise NotImplementedError(
            "The loss_function() function should be implemented for your model."
        )

    def set_loss_fn(self, loss_function):
        """
        Sets a custom loss function for the model. The loss function should be compatible with the
        model's output format.

        Args:
            loss_function (callable): A callable loss function that takes
                                    logits and labels as arguments.

        Example:
            >>> import torch.nn as nn
            >>> model.set_loss_fn(nn.CrossEntropyLoss())
        """
        self.loss_fn = loss_function
        try:
            self.loss_fn.weight.to(self.model.device)
        except AttributeError:
            # If the loss function does not have a weight attribute, we assume it's not weighted
            pass

    def predict(self, sequence_or_inputs, **kwargs):
        """
        This method takes raw sequences or tokenized inputs and returns
        the raw model outputs (logits, hidden states, etc.) without
        post-processing. It's useful for getting the model's direct
        predictions for further processing.

        If the model has a dataset_class, this method will use the dataset's
        prepare_input method for data preprocessing, allowing for more complex
        data preparation including custom field handling.

        Args:
            sequence_or_inputs: Can be one of:
                - str: A single sequence (e.g., "ATCGATCG")
                - list: A list of sequences (e.g., ["ATCGATCG", "GCTAGCTA"])
                - dict: A dictionary with 'sequence'/'seq' and optionally 'label'/'labels'
                        and other custom fields (e.g., {"sequence": "ATCG", "label": 1})
                - BatchEncoding/dict with 'input_ids': Already tokenized inputs
            **kwargs: Additional arguments for tokenization or dataset preparation.
                     Common options include 'max_length', 'padding', 'truncation', etc.

        Returns:
            dict: A dictionary containing the raw model outputs, typically including
                 `logits`, `last_hidden_state`, and other model-specific outputs.

        Example:
            >>> # Predict on a single sequence
            >>> outputs = model.predict("ATCGATCG")
            >>>
            >>> # Predict on multiple sequences
            >>> outputs = model.predict(["ATCGATCG", "GCTAGCTA"])
            >>>
            >>> # Predict with dict input (if dataset_class is set)
            >>> outputs = model.predict({"sequence": "ATCGATCG", "label": 1})
        """
        # Please implement the predict() function for your model
        raw_outputs = self._forward_from_raw_input(sequence_or_inputs, **kwargs)
        return raw_outputs

    def inference(self, sequence_or_inputs, **kwargs):
        """
        This method takes raw sequences or tokenized inputs and returns
        processed predictions that are ready for human consumption. It
        typically includes post-processing steps like converting logits
        to class labels or probabilities.

        If the model has a dataset_class, this method will use the dataset's
        prepare_input method for data preprocessing, allowing for more complex
        data preparation including custom field handling.

        Args:
            sequence_or_inputs: Can be one of:
                - str: A single sequence (e.g., "ATCGATCG")
                - list: A list of sequences (e.g., ["ATCGATCG", "GCTAGCTA"])
                - dict: A dictionary with 'sequence'/'seq' and optionally 'label'/'labels'
                        and other custom fields (e.g., {"sequence": "ATCG", "label": 1})
                - BatchEncoding/dict with 'input_ids': Already tokenized inputs
            **kwargs: Additional arguments for tokenization, dataset preparation, or inference.
                     Common options include 'max_length', 'padding', 'truncation', etc.

        Returns:
            dict: A dictionary containing the processed predictions, typically including
                 'predictions', 'confidence', and other human-readable outputs.

        Example:
            >>> # Inference on a single sequence
            >>> results = model.inference("ATCGATCG")
            >>> print(results['predictions'])  # Class labels
            >>>
            >>> # Inference on multiple sequences
            >>> results = model.inference(["ATCGATCG", "GCTAGCTA"])
            >>>
            >>> # Inference with dict input (if dataset_class is set)
            >>> results = model.inference({"sequence": "ATCGATCG", "label": 1})
            >>> print(results['predictions'])
        """
        # Please implement the predict() function for your model
        raw_outputs = self._forward_from_raw_input(sequence_or_inputs, **kwargs)
        return raw_outputs

    def __call__(self, **inputs):
        """
        The main forward pass of the model, suitable for training loops.

        This method is the primary interface for model forward passes during
        training. It handles both tokenized inputs and raw sequences,
        calculates loss if labels are provided, and returns a comprehensive
        output dictionary.

        Args:
            **inputs: A dictionary of tokenized inputs, potentially including
                     labels. Can also handle raw sequences that will be
                     tokenized automatically.

        Returns:
            dict: A dictionary containing logits, last_hidden_state, labels,
                  and loss (if labels were provided).

        Example:
            >>> # Training forward pass
            >>> outputs = model(
            ...     input_ids=torch.tensor([[1, 2, 3, 4]]),
            ...     attention_mask=torch.tensor([[1, 1, 1, 1]]),
            ...     labels=torch.tensor([0])
            ... )
            >>> loss = outputs['loss']
        """
        # For transformer trainer integration, we need to pop the "inputs" to be a tokenized inputs object.
        # For native trainer, the inputs are already tokenized inputs object
        labels = inputs.pop("labels", None)
        inputs = inputs.pop("inputs", inputs)
        inputs["labels"] = labels
        if isinstance(inputs, dict):
            labels = inputs.get("labels", None)
            label = inputs.get("label", None)
            labels = labels if labels is not None else label
            # if labels is None:
            #     warnings.warn(
            #         "No labels are provided in the inputs, the model will not calculate the loss."
            #     )
        elif isinstance(inputs, tuple):
            labels = inputs[1]
            inputs = inputs[0]
        elif labels is not None:
            labels = labels
        outputs = self.forward(**inputs)

        if labels is not None:
            outputs["loss"] = self._calculate_loss(outputs, labels)
        else:
            outputs["loss"] = None
        return outputs

    def _calculate_loss(self, outputs, labels):
        """
        Internal method to calculate loss if not already present in outputs.

        :param outputs: The dictionary of model outputs.
        :param labels: The ground truth labels.
        :return: The calculated loss.
        """
        loss = outputs.get("loss", None)
        if loss is not None:
            return loss

        logits = outputs["logits"]
        if logits is not None or labels is not None:
            loss = self.loss_function(logits, labels)
            return loss
        else:
            raise RuntimeError(
                "The output of the forward() function should be a dictionary-like objective"
                " and have either 'loss', or 'logits' and 'labels' attribute."
            )

    # ==================== Save Helper Methods ====================

    def _save_base_files(self, path):
        """
        Copy base model configuration files to save directory.

        This method copies essential configuration files from the original model
        directory to the target save directory, excluding weight files to avoid
        duplication. Files with extensions .bin, .json, .txt, and .py are copied.

        Args:
            path (str): Target directory path where files will be copied.

        Note:
            - pytorch_model.bin and model.safetensors are excluded as they
              will be saved separately in _save_weights()
            - Only files from the original model path are copied
        """
        for file in findfile.find_files(
            self.config.name_or_path,
            or_key=["bin", "json", "txt", "py"],
            exclude_key=["pytorch_model.bin", "model.safetensors"],
            return_relative_path=False,
        ):
            shutil.copyfile(file, f"{path}/{os.path.basename(file)}")

    def _save_custom_model_class(self, path, metadata):
        """
        Save custom model class source file if it's user-defined.

        This method detects and saves the source code of custom model classes
        that are defined outside the omnigenbench/omnigenome packages. This
        enables loading models without requiring the original source code.

        Args:
            path (str): Target directory path where the custom model file will be saved.
            metadata (dict): Metadata dictionary that will be updated with custom
                           model file information.

        Side Effects:
            - Creates 'custom_model.py' in the target directory if model is custom
            - Updates metadata with 'custom_model_file' key
            - Prints confirmation message on success
            - Prints warning on failure (non-fatal)

        Note:
            - Only saves source code for models NOT in omnigenbench/omnigenome packages
            - Failures are logged but don't interrupt the save process
        """
        try:
            model_class = self.__class__
            model_source_file = inspect.getfile(model_class)
            # Check if it's a user-defined model (not from omnigenbench/omnigenome packages)
            if (
                "omnigenbench" not in model_source_file
                and "omnigenome" not in model_source_file
            ):
                custom_model_path = os.path.join(path, "custom_model.py")
                shutil.copyfile(model_source_file, custom_model_path)
                metadata["custom_model_file"] = "custom_model.py"
                fprint(f"Saved custom model class source to: {custom_model_path}")
        except (TypeError, OSError) as e:
            fprint(f"Could not save custom model source file: {e}")

    def _save_custom_dataset_class(self, path, metadata):
        """
        Save custom dataset class source file if available.

        This method saves the source code of custom dataset classes associated
        with the model. This ensures that data preprocessing logic is preserved
        and can be used during inference without requiring the original code.

        Args:
            path (str): Target directory path where the custom dataset file will be saved.
            metadata (dict): Metadata dictionary that will be updated with custom
                           dataset file information.

        Side Effects:
            - Creates 'custom_dataset.py' in the target directory if dataset is custom
            - Updates metadata with 'custom_dataset_file' and 'custom_dataset_class' keys
            - Prints confirmation message on success
            - Prints warning on failure (non-fatal)

        Note:
            - Requires model to have 'dataset_class' attribute
            - Only saves source code for datasets NOT in omnigenbench/omnigenome packages
            - Silently returns if model has no dataset_class attribute
            - Failures are logged but don't interrupt the save process
        """
        if not hasattr(self, "dataset_class"):
            return

        try:
            dataset_class = self.dataset_class
            dataset_source_file = inspect.getfile(dataset_class)
            # Check if it's a user-defined dataset (not from omnigenbench/omnigenome)
            if (
                "omnigenbench" not in dataset_source_file
                and "omnigenome" not in dataset_source_file
            ):
                custom_dataset_path = os.path.join(path, "custom_dataset.py")
                shutil.copyfile(dataset_source_file, custom_dataset_path)
                metadata["custom_dataset_file"] = "custom_dataset.py"
                metadata["custom_dataset_class"] = dataset_class.__name__
                fprint(f"Saved custom dataset class source to: {custom_dataset_path}")
        except (TypeError, OSError, AttributeError) as e:
            fprint(f"Could not save custom dataset source file: {e}")

    def _collect_metadata(self):
        """
        Collect all metadata to be saved with the model.

        This method gathers comprehensive metadata about the model, including:
        - Loss function information (class name and module)
        - Model class information (name and module)
        - Custom attributes (num_labels, num_classes, label mappings, etc.)
        - Dataset metadata (if present)

        Returns:
            dict: A dictionary containing all model metadata with the following structure:
                {
                    'model_cls': str,           # Model class name
                    'model_module': str,        # Model module path
                    'loss_fn_class': str,       # Loss function class name (optional)
                    'loss_fn_module': str,      # Loss function module path (optional)
                    'custom_attrs': dict,       # Custom model attributes (optional)
                    'dataset_metadata': dict,   # Dataset metadata (optional)
                    ... (other metadata from self.metadata)
                }

        Note:
            - Only serializable attributes (int, float, str, bool, list, dict) are saved
            - Custom attributes checked: num_labels, num_classes, threshold,
              label2idx, idx2label, tissue_names, tissue_columns
            - Base metadata is copied from self.metadata
        """
        metadata = self.metadata.copy()

        # Loss function metadata
        if self.loss_fn is not None:
            metadata["loss_fn_class"] = self.loss_fn.__class__.__name__
            metadata["loss_fn_module"] = self.loss_fn.__class__.__module__

        # Model class metadata
        model_class = self.__class__
        metadata["model_cls"] = model_class.__name__
        metadata["model_module"] = model_class.__module__

        # Custom attributes
        custom_attrs = {}
        for attr_name in [
            "num_labels",
            "num_classes",
            "threshold",
            "label2idx",
            "idx2label",
            "tissue_names",
            "tissue_columns",
        ]:
            if hasattr(self, attr_name):
                attr_value = getattr(self, attr_name)
                if isinstance(attr_value, (int, float, str, bool, list, dict)):
                    custom_attrs[attr_name] = attr_value
        if custom_attrs:
            metadata["custom_attrs"] = custom_attrs

        # Dataset metadata
        if hasattr(self, "metadata") and "dataset_metadata" in self.metadata:
            metadata["dataset_metadata"] = self.metadata["dataset_metadata"]

        return metadata

    def _save_weights(self, path):
        """
        Save model weights and tokenizer to disk.

        This method handles the serialization of model weights and tokenizer.
        It attempts multiple saving strategies to ensure compatibility:
        1. Save tokenizer using dill serialization
        2. Try to save base model using save_pretrained() (HuggingFace style)
        3. Save complete state dict as fallback

        Args:
            path (str): Target directory path where weights and tokenizer will be saved.

        Side Effects:
            - Creates 'tokenizer.bin' in the target directory
            - Creates model weight files (via save_pretrained if available)
            - Creates 'pytorch_model.bin' containing complete state dict

        Note:
            - Tokenizer is serialized using dill to preserve all attributes
            - Base model save_pretrained() may fail for custom models (non-fatal)
            - Complete state dict is always saved as backup
        """
        # Save tokenizer
        with open(f"{path}/tokenizer.bin", "wb") as f:
            dill.dump(self.tokenizer, f)

        # Try to save the underlying base model
        try:
            self.model.save_pretrained(f"{path}", safe_serialization=False)
        except AttributeError:
            # Fallback: if the OmniModel subclass provides its own `save_pretrained`, use it
            if hasattr(self, "save_pretrained"):
                try:
                    self.save_pretrained(path, overwrite=True)
                except Exception:
                    pass

        # Save complete state dict including all components
        with open(f"{path}/pytorch_model.bin", "wb") as f:
            torch.save(self.state_dict(), f)

    def save(self, path, overwrite=False, dtype=torch.float16, **kwargs):
        """
        Save the complete model, tokenizer, and metadata to a directory.

        This method performs a comprehensive save operation that includes:
        - Base model configuration files
        - Custom model and dataset class source code (if applicable)
        - Complete metadata including loss function and custom attributes
        - Model weights and tokenizer

        The save process follows a 6-step workflow:
        1. Save base configuration files
        2. Collect comprehensive metadata
        3. Save custom model class source (if user-defined)
        4. Save custom dataset class source (if available)
        5. Save metadata to JSON file
        6. Save model weights and tokenizer

        Args:
            path (str): Target directory path for saving the model.
            overwrite (bool, optional): Whether to overwrite existing directory.
                                       If False, a timestamp will be appended to path.
                                       Defaults to False.
            dtype (torch.dtype, optional): Data type for saving model weights.
                                          Model is temporarily converted to this dtype
                                          during saving. Defaults to torch.float16.
            **kwargs: Additional keyword arguments (reserved for future use).

        Side Effects:
            - Creates target directory if it doesn't exist
            - Temporarily moves model to CPU and converts dtype
            - Restores original device and dtype after saving
            - Prints confirmation message on completion

        Example:
            >>> # Basic save
            >>> model.save("my_model")

            >>> # Save with overwrite
            >>> model.save("my_model", overwrite=True)

            >>> # Save with specific dtype
            >>> model.save("my_model", dtype=torch.float32)

        Note:
            - Model is set to eval mode before saving
            - Original device and dtype are preserved
            - Custom classes are only saved if defined outside framework packages
            - Failures in non-critical steps (e.g., custom class saving) are logged but don't stop the process
        """
        self.eval()

        # Handle path conflicts
        if os.path.exists(path) and not overwrite:
            fprint(
                f"The path {path} already exists, please set overwrite=True to overwrite it. "
                f"Rename the path to {path}_{time.strftime('%Y%m%d_%H%M%S')} to save it with a timestamp."
            )
            path = f"{path}_{time.strftime('%Y%m%d_%H%M%S')}"
        if not os.path.exists(path):
            os.makedirs(path)

        # Store original device and dtype
        _device = self.model.device
        _dtype = self.model.dtype
        self.model.to(dtype).to("cpu")

        # Save tokenizer config
        self.tokenizer.save_pretrained(path)

        # Step 1: Save base files
        self._save_base_files(path)

        # Step 2: Collect metadata
        metadata = self._collect_metadata()

        # Step 3: Save custom model class
        self._save_custom_model_class(path, metadata)

        # Step 4: Save custom dataset class
        self._save_custom_dataset_class(path, metadata)

        # Step 5: Save metadata to JSON
        with open(f"{path}/metadata.json", "w", encoding="utf8") as f:
            json.dump(metadata, f, indent=2)

        # Step 6: Save weights and tokenizer
        self._save_weights(path)

        # Restore original device and dtype
        self.model.to(_dtype).to(_device)
        fprint(f"The model is saved to {path}.")

    # ==================== Load Helper Methods ====================

    def _load_metadata(self, path):
        """
        Load and validate metadata from saved model directory.

        This method reads the metadata.json file and performs validation
        to ensure the saved model matches the current model class.

        Args:
            path (str): Directory path containing the saved model and metadata.json.

        Returns:
            dict: Loaded metadata dictionary containing model information.

        Raises:
            ValueError: If the saved model class doesn't match the current model class.
            FileNotFoundError: If metadata.json is not found in the directory.
            json.JSONDecodeError: If metadata.json is malformed.

        Example:
            >>> metadata = model._load_metadata("checkpoint")
            >>> print(metadata['model_cls'])  # 'OmniModelForSequenceClassification'

        Note:
            - Validates that saved model class matches current class
            - This check ensures type safety when loading models
        """
        with open(f"{path}/metadata.json", "r", encoding="utf8") as f:
            metadata = json.load(f)

        if metadata["model_cls"] != self.__class__.__name__:
            raise ValueError(
                f"The model class in the loaded model is {metadata['model_cls']}, "
                f"but the current model class is {self.__class__.__name__}."
            )

        return metadata

    def _load_config(self, path, **kwargs):
        """
        Load model configuration and check for differences with current config.

        This method loads the saved configuration and compares it with the
        current model's configuration, warning about any differences found.

        Args:
            path (str): Directory path containing the saved model configuration.
            **kwargs: Additional arguments passed to AutoConfig.from_pretrained().

        Returns:
            AutoConfig: Loaded configuration object.

        Side Effects:
            - Prints warnings for any configuration differences found
            - Warnings include the key name and both values (saved vs current)

        Example:
            >>> config = model._load_config("checkpoint", trust_remote_code=True)
            Warning: The value of the key num_labels in the loaded model is 10,
            but the current value is 5.

        Note:
            - trust_remote_code is set to True by default for custom models
            - Configuration differences don't prevent loading but are logged
            - Useful for detecting model version mismatches
        """
        config = AutoConfig.from_pretrained(path, trust_remote_code=True, **kwargs)

        for key, value in config.__dict__.items():
            if key not in self.config.__dict__ or self.config.__dict__[key] != value:
                fprint(
                    f"Warning: The value of the key {key} in the loaded model is {value}, "
                    f"but the current value is {self.config.__dict__.get(key, None)}."
                )

        return config

    def _load_dataset_class(self, path, metadata):
        """
        Restore the dataset class from metadata.

        This method attempts to dynamically import the dataset class that was
        saved with the model. This allows the model to use the same data
        preprocessing logic during inference as was used during training.

        Args:
            path (str): Directory path containing the saved model.
            metadata (dict): Metadata dictionary containing dataset class information.
                           Expected keys: 'dataset_cls', 'dataset_module', 'custom_dataset_file', 'custom_dataset_class'

        Side Effects:
            - Sets self.dataset_class to the restored dataset class
            - Prints confirmation message on success
            - Prints warning on failure (non-fatal)

        Note:
            - Silently returns if dataset class info is not in metadata
            - Tries multiple loading strategies: built-in modules, custom files
            - Import or attribute errors are caught and logged as warnings
            - Custom dataset classes are loaded from custom_dataset.py if available
        """
        if "dataset_cls" not in metadata and "custom_dataset_class" not in metadata:
            return

        dataset_cls = None

        # Method 1: Try to load from custom_dataset.py file
        if "custom_dataset_file" in metadata and "custom_dataset_class" in metadata:
            try:
                custom_dataset_path = os.path.join(
                    path, metadata["custom_dataset_file"]
                )
                if os.path.exists(custom_dataset_path):
                    import importlib.util

                    spec = importlib.util.spec_from_file_location(
                        "custom_dataset_module", custom_dataset_path
                    )
                    custom_dataset_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(custom_dataset_module)
                    dataset_cls = getattr(
                        custom_dataset_module, metadata["custom_dataset_class"]
                    )
                    fprint(
                        f"Restored custom dataset class: {metadata['custom_dataset_class']} from {metadata['custom_dataset_file']}"
                    )
            except (ImportError, AttributeError, OSError) as e:
                warnings.warn(f"Could not restore custom dataset class from file: {e}")

        # Method 2: Try to load from built-in omnigenbench/omnigenome modules
        if (
            dataset_cls is None
            and "dataset_module" in metadata
            and "dataset_cls" in metadata
        ):
            try:
                dataset_module = import_module(metadata["dataset_module"])
                dataset_cls = getattr(dataset_module, metadata["dataset_cls"])
                fprint(
                    f"Restored dataset class: {metadata['dataset_cls']} from {metadata['dataset_module']}"
                )
            except (ImportError, AttributeError) as e:
                warnings.warn(f"Could not restore dataset class from module: {e}")

        if dataset_cls is not None:
            self.dataset_class = dataset_cls

    def _load_loss_function(self, metadata):
        """
        Restore saved loss function from metadata.

        This method attempts to restore the loss function that was used
        during training by dynamically importing it based on saved metadata.

        Args:
            metadata (dict): Metadata dictionary containing loss function information.
                           Expected keys: 'loss_fn_class', 'loss_fn_module'

        Side Effects:
            - Sets self.loss_fn to the restored loss function instance
            - Prints confirmation message on success
            - Prints warning on failure (non-fatal)

        Example:
            >>> # metadata contains: {'loss_fn_class': 'CrossEntropyLoss', 'loss_fn_module': 'torch.nn.modules.loss'}
            >>> model._load_loss_function(metadata)
            Restored loss function: CrossEntropyLoss from torch.nn.modules.loss

        Note:
            - Silently returns if loss function info is not in metadata
            - Import or attribute errors are caught and logged as warnings
            - Loss function is instantiated with default parameters
            - Custom loss functions must be importable at load time
        """
        if "loss_fn_class" not in metadata or "loss_fn_module" not in metadata:
            return

        try:
            loss_module = import_module(metadata["loss_fn_module"])
            loss_class = getattr(loss_module, metadata["loss_fn_class"])
            self.loss_fn = loss_class()
            fprint(
                f"Restored loss function: {metadata['loss_fn_class']} from {metadata['loss_fn_module']}"
            )
        except (ImportError, AttributeError) as e:
            warnings.warn(f"Could not restore loss function: {e}")

    def _load_weights(self, path, **kwargs):
        """
        Load model weights with validation and compatibility checks.

        This method loads the saved state dictionary and performs thorough
        validation by comparing saved weights with the current model structure.
        It reports any missing or unexpected keys.

        Args:
            path (str): Directory path containing pytorch_model.bin file.
            **kwargs: Additional arguments, may include:
                     - device (str): Device to map loaded tensors to (e.g., 'cpu', 'cuda:0')

        Side Effects:
            - Loads weights into current model's state dict
            - Prints warnings for missing or unexpected keys
            - Uses strict=False to allow partial loading

        Warnings:
            - Missing keys: Parameters in current model not found in saved weights
            - Unexpected keys: Parameters in saved weights not found in current model

        Example:
            >>> model._load_weights("checkpoint", device="cuda:0")
            Warning: Missing keys in loaded weights: {'classifier.bias'}
            Warning: Unexpected keys in loaded weights: {'old_layer.weight'}

        Note:
            - strict=False allows loading with architecture mismatches
            - Missing keys will be randomly initialized
            - Unexpected keys are ignored
            - Device mapping prevents CUDA OOM when loading on different device
        """
        with open(f"{path}/pytorch_model.bin", "rb") as f:
            loaded_state_dict = torch.load(f, map_location=kwargs.get("device", "cpu"))

            # Check if keys match between current and loaded state dict
            current_keys = set(self.state_dict().keys())
            loaded_keys = set(loaded_state_dict.keys())
            missing_keys = current_keys - loaded_keys
            unexpected_keys = loaded_keys - current_keys

            if missing_keys:
                warnings.warn(f"Missing keys in loaded weights: {missing_keys}")
            if unexpected_keys:
                warnings.warn(f"Unexpected keys in loaded weights: {unexpected_keys}")

            self.load_state_dict(loaded_state_dict, strict=False)

    def _load_tokenizer(self, path):
        """
        Load saved tokenizer from binary file.

        This method deserializes the tokenizer that was saved using dill.
        The tokenizer is essential for proper text preprocessing during inference.

        Args:
            path (str): Directory path containing tokenizer.bin file.

        Side Effects:
            - Sets self.tokenizer to the loaded tokenizer instance
            - Silently returns if tokenizer.bin doesn't exist

        Example:
            >>> model._load_tokenizer("checkpoint")
            >>> print(type(model.tokenizer))  # <class 'omnigenbench.OmniTokenizer'>

        Note:
            - Tokenizer is saved/loaded using dill for complete serialization
            - If tokenizer.bin doesn't exist, current tokenizer is preserved
            - Dill is used instead of pickle to handle complex tokenizer objects
        """
        tokenizer_path = f"{path}/tokenizer.bin"
        if os.path.exists(tokenizer_path):
            with open(tokenizer_path, "rb") as f:
                self.tokenizer = dill.load(f)

    def load(self, path, **kwargs):
        """
        Load a complete model from a saved directory.

        This method performs a comprehensive load operation that restores:
        - Model metadata and configuration
        - Loss function (if saved)
        - Model weights with validation
        - Tokenizer

        The load process follows a 6-step workflow:
        1. Load and validate metadata
        2. Load model configuration
        3. Restore dataset class (if available)
        4. Restore loss function (if available)
        5. Load model weights with validation
        6. Load tokenizer

        Args:
            path (str): Directory path containing the saved model.
            **kwargs: Additional keyword arguments passed to loading functions.
                     Common options include:
                     - device (str): Device to load model to (e.g., 'cpu', 'cuda:0')
                     - trust_remote_code (bool): Whether to trust custom code

        Returns:
            OmniModel: The loaded model instance (self).

        Raises:
            ValueError: If saved model class doesn't match current class.
            FileNotFoundError: If required files (metadata.json, pytorch_model.bin) are missing.

        Example:
            >>> # Basic load
            >>> loaded_model = model.load("checkpoint")

            >>> # Load to specific device
            >>> loaded_model = model.load("checkpoint", device="cuda:0")

            >>> # Load with custom code trust
            >>> loaded_model = model.load("checkpoint", trust_remote_code=True)

        Side Effects:
            - Updates all model attributes (weights, config, tokenizer, loss_fn)
            - Prints warnings for configuration differences
            - Prints warnings for weight loading issues (missing/unexpected keys)
            - Prints confirmation messages for restored components

        Note:
            - Model class must match the saved model class
            - Partial loading is supported (missing keys will be randomly initialized)
            - Loss function restoration is optional (warnings only if fails)
            - Custom models require trust_remote_code=True (enabled by default)
        """
        # Step 1: Load metadata
        metadata = self._load_metadata(path)

        # Step 2: Load configuration
        config = self._load_config(path, **kwargs)

        # Step 3: Restore dataset class
        self._load_dataset_class(path, metadata)

        # Step 4: Restore loss function
        self._load_loss_function(metadata)

        # Step 5: Load weights
        self._load_weights(path, **kwargs)

        # Step 6: Load tokenizer
        self._load_tokenizer(path)

        return self

    def _forward_from_raw_input(self, sequence_or_inputs, **kwargs):
        """
        Tokenizes raw input and performs a forward pass in no_grad mode.

        This method supports two preprocessing strategies:
        1. If dataset_class is available, use dataset's prepare_input method for preprocessing
        2. Otherwise, use direct tokenizer (backward compatible)

        :param sequence_or_inputs: A sequence (str), list of sequences, dict with 'sequence' key,
                                  or tokenized inputs (BatchEncoding/dict).
        :param kwargs: Additional arguments for tokenization or dataset preparation.
        :return: A dictionary containing the raw model outputs and the tokenized inputs.
        """
        # Check if inputs are already tokenized
        if isinstance(sequence_or_inputs, (BatchEncoding, dict)):
            # If it's a dict, check if it contains 'input_ids' (tokenized) or raw data
            if (
                isinstance(sequence_or_inputs, dict)
                and "input_ids" in sequence_or_inputs
            ):
                inputs = sequence_or_inputs
            # If it's a dict without 'input_ids', it might be raw data for dataset.prepare_input
            elif (isinstance(sequence_or_inputs, dict)
                  and hasattr( self, "dataset_class")
                  and self.dataset_class is not None):
                # Use dataset's prepare_input method
                try:
                    # Create a temporary dataset instance for using prepare_input
                    max_length = kwargs.pop("max_length", 1024)
                    dataset_instance = self.dataset_class(
                        dataset_name_or_path=None,
                        tokenizer=self.tokenizer,
                        max_length=max_length,
                        **kwargs,
                    )
                    inputs = dataset_instance.prepare_input(
                        sequence_or_inputs, **kwargs
                    )
                    # Remove batch dimension if present (prepare_input may add it)
                    for key, value in inputs.items():
                        if (
                            isinstance(value, torch.Tensor)
                            and value.dim() > 1
                            and value.size(0) == 1
                        ):
                            inputs[key] = value.squeeze(0)
                except Exception as e:
                    warnings.warn(
                        f"Failed to use dataset.prepare_input: {e}. Falling back to tokenizer."
                    )
                    # Fallback to tokenizer for dict input
                    if "sequence" in sequence_or_inputs or "seq" in sequence_or_inputs:
                        seq = sequence_or_inputs.get(
                            "sequence", sequence_or_inputs.get("seq")
                        )
                        inputs = self.tokenizer(
                            seq,
                            padding=kwargs.pop("padding", True),
                            max_length=kwargs.pop("max_length", 1024),
                            truncation=kwargs.pop("truncation", True),
                            return_tensors=kwargs.pop("return_tensors", "pt"),
                            **kwargs,
                        )
                    else:
                        inputs = sequence_or_inputs
            else:
                inputs = sequence_or_inputs
        # Handle string or list of strings
        elif isinstance(sequence_or_inputs, (str, list)):
            # If dataset_class is available, try to use its prepare_input method
            if hasattr(self, "dataset_class") and self.dataset_class is not None:
                try:
                    # Prepare instance(s) for dataset.prepare_input
                    if isinstance(sequence_or_inputs, str):
                        instance = sequence_or_inputs
                    else:
                        # For list of sequences, we'll process them one by one
                        instance = sequence_or_inputs

                    max_length = kwargs.pop("max_length", 1024)
                    dataset_instance = self.dataset_class(
                        dataset_name_or_path=None,
                        tokenizer=self.tokenizer,
                        max_length=max_length,
                        **kwargs,
                    )

                    if isinstance(instance, list):
                        # Process list of sequences
                        batch_inputs = []
                        for seq in instance:
                            inp = dataset_instance.prepare_input(seq, **kwargs)
                            batch_inputs.append(inp)
                        # Stack all inputs
                        inputs = {
                            key: torch.stack(
                                [
                                    (
                                        inp[key].squeeze(0)
                                        if inp[key].dim() > 0
                                        else inp[key]
                                    )
                                    for inp in batch_inputs
                                ]
                            )
                            for key in batch_inputs[0].keys()
                        }
                    else:
                        inputs = dataset_instance.prepare_input(instance, **kwargs)
                        # Remove batch dimension if present
                        for key, value in inputs.items():
                            if (
                                isinstance(value, torch.Tensor)
                                and value.dim() > 1
                                and value.size(0) == 1
                            ):
                                inputs[key] = value.squeeze(0)
                except Exception as e:
                    warnings.warn(
                        f"Failed to use dataset.prepare_input: {e}. Falling back to tokenizer."
                    )
                    # Fallback to tokenizer
                    inputs = self.tokenizer(
                        sequence_or_inputs,
                        padding=kwargs.pop("padding", True),
                        max_length=kwargs.pop("max_length", 1024),
                        truncation=kwargs.pop("truncation", True),
                        return_tensors=kwargs.pop("return_tensors", "pt"),
                        **kwargs,
                    )
            else:
                # No dataset_class, use tokenizer directly (backward compatible)
                inputs = self.tokenizer(
                    sequence_or_inputs,
                    padding=kwargs.pop("padding", True),
                    max_length=kwargs.pop("max_length", 1024),
                    truncation=kwargs.pop("truncation", True),
                    return_tensors=kwargs.pop("return_tensors", "pt"),
                    **kwargs,
                )
        else:
            raise ValueError(f"Unsupported input type: {type(sequence_or_inputs)}")

        # Ensure inputs are on the correct device and add batch dimension if needed
        if not isinstance(inputs, (BatchEncoding, dict)):
            raise ValueError(f"Processed inputs must be a dict, got {type(inputs)}")

        # Add batch dimension if missing
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor) and value.dim() == 1:
                inputs[key] = value.unsqueeze(0)

        # Move tensors to the model's device
        target_device = self.device
        inputs = {
            k: v.to(target_device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        with torch.no_grad():
            raw_outputs = self(**inputs)
            raw_outputs["inputs"] = inputs
        return raw_outputs

    @property
    def device(self):
        """Return the actual device of model parameters, not cached value."""
        # Always infer from actual parameters to handle subclass layers correctly
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def to(self, *args, **kwargs):
        """Move model to specified device/dtype and keep device tracking in sync."""
        super().to(*args, **kwargs)
        self.model.to(*args, **kwargs)

        # Update internal device tracking if a device/dtype is specified
        # Try to derive device from args/kwargs or model parameters
        updated_device = None
        for arg in args:
            if isinstance(arg, torch.device):
                updated_device = arg
            elif isinstance(arg, str) and ("cuda" in arg or "cpu" in arg):
                updated_device = torch.device(arg)
        if "device" in kwargs:
            dev = kwargs.get("device")
            updated_device = dev if isinstance(dev, torch.device) else torch.device(dev)
        if updated_device is None:
            try:
                updated_device = next(self.model.parameters()).device
            except StopIteration:
                updated_device = self.device

        self._device = updated_device
        # Mirror `.device` for compatibility with existing call sites
        try:
            self.model.device = updated_device
            for module in self.model.modules():
                try:
                    module.device = updated_device
                except Exception:
                    pass
        except Exception:
            pass

        return self

    @staticmethod
    def from_pretrained(config_or_model, tokenizer, *args, **kwargs):
        """
        Loads a pre-trained model and tokenizer.

        :param config_or_model: The name or path of the pre-trained model.
        :param tokenizer: The tokenizer to use.
        :param args: Additional positional arguments.
        :param kwargs: Additional keyword arguments.
        :return: An instance of `OmniModel`.
        """
        config = kwargs.pop("config", None)
        if config is None:
            config = AutoConfig.from_pretrained(config_or_model, **kwargs)
        base_model = AutoModel.from_pretrained(config_or_model, **kwargs)
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(base_model, **kwargs)
        return OmniModel(config, base_model, tokenizer, *args, **kwargs)

    def model_info(self):
        """
        Prints and returns detailed information about the model.

        :return: A string containing the model information.
        """
        info = f"Model Name: {self.__class__.__name__}\n"
        info += f"Model Metadata: {self.metadata}\n"
        info += f"Base Model Name: {self.config.name_or_path}\n"
        info += f"Model Type: {self.config.model_type}\n"
        info += f"Model Architecture: {self.config.architectures}\n"
        info += f"Model Parameters: {count_parameters(self.model) / 1e6} M\n"
        info += f"Model Config: {self.config}\n"
        fprint(info)
        return info
