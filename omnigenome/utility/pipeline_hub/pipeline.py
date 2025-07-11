# -*- coding: utf-8 -*-
# file: pipeline.py
# time: 18:38 12/04/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.
"""
Pipeline Module

This module provides the Pipeline class for creating and managing complete
machine learning workflows that combine models, tokenizers, datasets, and
trainers. Pipelines provide a unified interface for training, inference,
and model management.
"""

import json
import os

import autocuda
from transformers import AutoConfig, AutoTokenizer

from ..hub_utils import download_pipeline
from ..model_hub.model_hub import ModelHub
from ...src.abc.abstract_model import OmniModel
from ...src.misc.utils import env_meta_info, fprint
from ...src.trainer.trainer import Trainer


class Pipeline:
    """
    Complete machine learning pipeline combining model, tokenizer, datasets, and trainer.
    
    The Pipeline class provides a unified interface for managing complete machine
    learning workflows. It handles model initialization, training, inference, and
    persistence. Pipelines can be loaded from pre-built configurations or created
    from scratch with custom components.
    
    Attributes:
        model (OmniModel): The underlying model for the pipeline.
        tokenizer: Tokenizer for preprocessing input sequences.
        dataset (dict): Dictionary containing train/validation/test datasets.
        metadata (dict): Environment and pipeline metadata.
        trainer (Trainer): Trainer instance for model training.
        device (str): Target device for model execution (CPU/GPU).
        name (str): Name identifier for the pipeline.
    
    Example:
        >>> from omnigenome import Pipeline, OmniModelForSequenceClassification
        >>> # Create pipeline from model
        >>> model = OmniModelForSequenceClassification("model_path", tokenizer)
        >>> pipeline = Pipeline("my_pipeline", model_name_or_path=model)
        >>> # Use for inference
        >>> predictions = pipeline("ATCGATCG")
        >>> # Train the model
        >>> pipeline.train(datasets)
        >>> # Save pipeline
        >>> pipeline.save("./saved_pipeline")
    
    Note:
        - Pipelines automatically handle device placement and model optimization
        - Environment metadata is collected for reproducibility
        - Pipelines can be saved and loaded for easy deployment
        - Supports both local models and hub-based model loading
    """
    
    model: OmniModel = None
    tokenizer = None
    dataset: dict = None
    metadata: dict = None

    def __init__(
        self,
        name,
        *,
        model_name_or_path,
        tokenizer=None,
        datasets=None,
        trainer=None,
        **kwargs,
    ):
        """
        Initialize a Pipeline instance.
        
        Args:
            name (str): Name identifier for the pipeline.
            model_name_or_path (Union[str, OmniModel]): Model to use in the pipeline.
                Can be a string path/identifier or an OmniModel instance.
            tokenizer (optional): Tokenizer for preprocessing. If None, will be
                loaded from the model or model path. Defaults to None.
            datasets (dict, optional): Dictionary containing train/validation/test
                datasets. Keys should be 'train', 'valid', 'test'. Defaults to None.
            trainer (Trainer, optional): Trainer instance for model training.
                If None, a default trainer will be created. Defaults to None.
            **kwargs: Additional keyword arguments including:
                - device (str): Target device for model execution
                - trust_remote_code (bool): Whether to trust remote code in tokenizers
                - Other model-specific configuration parameters
        
        Raises:
            ValueError: If model initialization fails.
            ImportError: If required dependencies are not available.
            FileNotFoundError: If model path is invalid.
        
        Example:
            >>> # Create from model path
            >>> pipeline = Pipeline("rna_classification", 
            ...                    model_name_or_path="yangheng/OmniGenome-186M")
            >>> # Create from model instance
            >>> model = OmniModelForSequenceClassification("model_path", tokenizer)
            >>> pipeline = Pipeline("custom_pipeline", model_name_or_path=model)
        
        Note:
            - The pipeline automatically handles model loading and device placement
            - Environment metadata is collected for tracking system information
            - If a model instance is provided, its tokenizer and metadata are used
        """
        self.metadata = env_meta_info()
        self.name = name
        self.tokenizer = tokenizer
        self.datasets = datasets
        self.trainer = trainer
        self.device = (
            autocuda.auto_cuda()
            if kwargs.get("device") is None
            else kwargs.get("device")
        )
        if not isinstance(model_name_or_path, str):
            self.model = model_name_or_path
            self.tokenizer = self.model.tokenizer
            self.metadata = self.model.metadata
        else:
            self.init_pipeline(
                model_name_or_path=model_name_or_path, tokenizer=tokenizer, **kwargs
            )

        self.model.to(self.device)

    def __call__(self, inputs, *args, **kwargs):
        """
        Call the pipeline for inference.
        
        This method provides a convenient interface for running inference
        through the pipeline. It delegates to the model's inference method.
        
        Args:
            inputs: Input data for inference (can be string, list, or tensor).
            *args: Additional positional arguments passed to model inference.
            **kwargs: Additional keyword arguments passed to model inference.
        
        Returns:
            dict: Inference results including predictions and confidence scores.
        
        Example:
            >>> pipeline = Pipeline("my_pipeline", model_name_or_path=model)
            >>> results = pipeline("ATCGATCG")
            >>> print(results['predictions'])
        """
        return self.model.inference(inputs, **kwargs)

    def to(self, device):
        """
        Move the pipeline to a specific device.
        
        Args:
            device (str): Target device ('cpu', 'cuda', 'cuda:0', etc.).
        
        Returns:
            Pipeline: Self for method chaining.
        
        Example:
            >>> pipeline = Pipeline("my_pipeline", model_name_or_path=model)
            >>> pipeline.to("cuda:0")  # Move to GPU
            >>> pipeline.to("cpu")     # Move to CPU
        """
        self.model.to(device)
        self.device = device
        return self

    def init_pipeline(self, *, model_name_or_path, tokenizer=None, **kwargs):
        """
        Initialize the pipeline components from a model path.
        
        This method handles loading the model, tokenizer, and configuration
        from a model path or identifier. It tries to load from the ModelHub
        first, then falls back to HuggingFace transformers.
        
        Args:
            model_name_or_path (str): Path or identifier of the model to load.
            tokenizer (optional): Tokenizer instance. If None, will be loaded
                from the model path. Defaults to None.
            **kwargs: Additional keyword arguments for model loading including:
                - trust_remote_code (bool): Whether to trust remote code
                - device (str): Target device for the model
                - Other model-specific parameters
        
        Returns:
            Pipeline: Self for method chaining.
        
        Raises:
            ValueError: If model loading fails.
            ImportError: If required dependencies are not available.
        
        Example:
            >>> pipeline = Pipeline("my_pipeline")
            >>> pipeline.init_pipeline(model_name_or_path="yangheng/OmniGenome-186M")
        
        Note:
            - First attempts to load from OmniGenome ModelHub
            - Falls back to HuggingFace transformers if ModelHub fails
            - Automatically handles tokenizer loading and configuration
        """
        trust_remote_code = kwargs.get("trust_remote_code", True)
        try:  # for the models saved by OmniGenome and served by the model hub
            self.model = ModelHub.load(model_name_or_path, **kwargs)
            self.tokenizer = self.model.tokenizer
            self.metadata.update(self.model.metadata)
        except Exception as e:
            fprint(f"Fail to load the model from the model hub, the error is: {e}")

            config = AutoConfig.from_pretrained(
                model_name_or_path, trust_remote_code=trust_remote_code
            )
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name_or_path, trust_remote_code=trust_remote_code
                )
            self.model = OmniModel.from_pretrained(
                model_name_or_path,
                config=config,
                tokenizer=tokenizer,
                trust_remote_code=trust_remote_code,
                **kwargs,
            )
            self.tokenizer = self.model.tokenizer
            self.metadata.update(self.model.metadata)
        fprint(f"The pipeline has been initialized from {model_name_or_path}.")
        return self

    def train(self, datasets: dict = None, trainer=None, **kwargs):
        """
        Train the model in the pipeline.
        
        This method initiates training of the model using the provided datasets
        and trainer configuration. If no trainer is provided, the pipeline's
        existing trainer will be used.
        
        Args:
            datasets (dict, optional): Dictionary containing train/validation/test
                datasets. If None, uses the pipeline's existing datasets.
                Keys should be 'train', 'valid', 'test'. Defaults to None.
            trainer (Trainer, optional): Trainer instance to use for training.
                If None, uses the pipeline's existing trainer. Defaults to None.
            **kwargs: Additional keyword arguments passed to the trainer.
        
        Raises:
            ValueError: If no trainer is available or datasets are invalid.
            RuntimeError: If training fails.
        
        Example:
            >>> pipeline = Pipeline("my_pipeline", model_name_or_path=model)
            >>> # Train with existing datasets
            >>> pipeline.train()
            >>> # Train with custom datasets
            >>> custom_datasets = {'train': train_data, 'valid': valid_data}
            >>> pipeline.train(datasets=custom_datasets)
            >>> # Train with custom trainer
            >>> from omnigenome import Trainer
            >>> custom_trainer = Trainer(model, train_dataset=train_data)
            >>> pipeline.train(trainer=custom_trainer)
        
        Note:
            - Training uses the pipeline's current model and device
            - Progress and metrics are logged during training
            - The trained model is automatically saved in the pipeline
        """
        if trainer is not None:
            assert isinstance(trainer, Trainer)
            self.trainer = trainer

        self.trainer.train()

    def predict(self, inputs, **kwargs):
        """
        Generate predictions for input data.
        
        This method provides a high-level interface for generating predictions
        from the pipeline's model. It handles preprocessing and postprocessing
        automatically.
        
        Args:
            inputs: Input data for prediction. Can be:
                - str: Single sequence string
                - list: List of sequence strings
                - tensor: Preprocessed input tensors
            **kwargs: Additional keyword arguments passed to model prediction.
        
        Returns:
            dict: Prediction results including:
                - predictions: Predicted labels or values
                - confidence: Confidence scores (if available)
                - logits: Raw model outputs (if requested)
        
        Example:
            >>> pipeline = Pipeline("my_pipeline", model_name_or_path=model)
            >>> # Single prediction
            >>> result = pipeline.predict("ATCGATCG")
            >>> print(result['predictions'])
            >>> # Batch prediction
            >>> results = pipeline.predict(["ATCGATCG", "GCTAGCTA"])
            >>> print(results['predictions'])
        
        Note:
            - Input preprocessing is handled automatically
            - Results are formatted consistently across different model types
            - Confidence scores are included when available
        """
        return self.model.predict(inputs, **kwargs)

    def inference(self, inputs, **kwargs):
        """
        Run full inference pipeline on input data.
        
        This method provides the complete inference pipeline including
        preprocessing, model forward pass, and postprocessing. It's the
        recommended method for production inference.
        
        Args:
            inputs: Input data for inference. Can be:
                - str: Single sequence string
                - list: List of sequence strings
                - tensor: Preprocessed input tensors
            **kwargs: Additional keyword arguments for inference including:
                - return_attention: Whether to return attention weights
                - return_hidden_states: Whether to return hidden states
                - temperature: Temperature for sampling (if applicable)
        
        Returns:
            dict: Complete inference results including:
                - predictions: Final predictions
                - confidence: Confidence scores
                - attention: Attention weights (if requested)
                - hidden_states: Hidden states (if requested)
        
        Example:
            >>> pipeline = Pipeline("my_pipeline", model_name_or_path=model)
            >>> # Basic inference
            >>> results = pipeline.inference("ATCGATCG")
            >>> print(results['predictions'])
            >>> # Inference with attention
            >>> results = pipeline.inference("ATCGATCG", return_attention=True)
            >>> print(results['attention'].shape)
        
        Note:
            - This is the most comprehensive inference method
            - Handles all preprocessing and postprocessing automatically
            - Returns rich information about the model's internal states
        """
        return self.model.inference(inputs, **kwargs)

    @staticmethod
    def load(pipeline_name_or_path, local_only=False, **kwargs):
        """
        Load a pipeline from disk or hub.
        
        This static method loads a complete pipeline including model, tokenizer,
        datasets, and trainer from a saved pipeline directory or hub identifier.
        
        Args:
            pipeline_name_or_path (str): Path to saved pipeline directory or
                hub identifier for downloading.
            local_only (bool, optional): If True, only load from local paths.
                If False, download from hub if not found locally. Defaults to False.
            **kwargs: Additional keyword arguments for pipeline initialization:
                - device: Target device for the model
                - name: Custom name for the pipeline
                - trust_remote_code: Whether to trust remote code
        
        Returns:
            Pipeline: Loaded pipeline instance ready for use.
        
        Raises:
            FileNotFoundError: If pipeline cannot be found locally and
                local_only is True.
            ValueError: If pipeline files are corrupted or invalid.
            ImportError: If required dependencies are not available.
        
        Example:
            >>> # Load from local path
            >>> pipeline = Pipeline.load("./saved_pipeline")
            >>> # Load from hub
            >>> pipeline = Pipeline.load("yangheng/OmniGenome-RNA-Classification")
            >>> # Use loaded pipeline
            >>> results = pipeline("ATCGATCG")
        
        Note:
            - Loads all pipeline components (model, tokenizer, datasets, trainer)
            - Automatically handles device placement
            - Preserves all training configurations and metadata
        """
        import dill

        if os.path.exists(pipeline_name_or_path):
            path = pipeline_name_or_path
        else:
            path = download_pipeline(
                pipeline_name_or_path, local_only=local_only, **kwargs
            )
        with open(f"{path}/datasets.pkl", "rb") as f:
            datasets = dill.load(f)
        with open(f"{path}/trainer.pkl", "rb") as f:
            trainer = dill.load(f)
        model = ModelHub.load(path, local_only=local_only, **kwargs)
        tokenizer = model.tokenizer
        pipeline = Pipeline(
            name=(
                pipeline_name_or_path
                if kwargs.get("name") is None
                else kwargs.get("name")
            ),
            model_name_or_path=model,
            tokenizer=tokenizer,
            datasets=datasets,
            trainer=trainer,
            **kwargs,
        )
        return pipeline

    def save(self, path, overwrite=False, **kwargs):
        """
        Save the pipeline to disk.
        
        This method saves the complete pipeline including model, tokenizer,
        datasets, trainer, and metadata to a directory. The saved pipeline
        can be loaded later using Pipeline.load().
        
        Args:
            path (str): Directory path where to save the pipeline.
            overwrite (bool, optional): If True, overwrite existing directory.
                If False, raise error if directory exists. Defaults to False.
            **kwargs: Additional keyword arguments for model saving.
        
        Raises:
            FileExistsError: If path exists and overwrite is False.
            OSError: If there are issues creating the directory or writing files.
            RuntimeError: If saving fails due to model or data issues.
        
        Example:
            >>> pipeline = Pipeline("my_pipeline", model_name_or_path=model)
            >>> # Train the pipeline
            >>> pipeline.train(datasets)
            >>> # Save the trained pipeline
            >>> pipeline.save("./trained_pipeline", overwrite=True)
            >>> # Load the saved pipeline later
            >>> loaded_pipeline = Pipeline.load("./trained_pipeline")
        
        Note:
            - Saves all pipeline components (model, tokenizer, datasets, trainer)
            - Preserves training configurations and metadata
            - Model is temporarily moved to CPU during saving to avoid GPU memory issues
            - Creates a complete, self-contained pipeline directory
        """
        import dill

        if os.path.exists(path) and not overwrite:
            raise FileExistsError(
                f"The path {path} already exists, please set overwrite=True to overwrite it."
            )
        if not os.path.exists(path):
            os.makedirs(path)
        device = self.model.model.device
        self.model.model.to("cpu")
        with open(f"{path}/datasets.pkl", "wb") as f:
            dill.dump(self.datasets, f)
        with open(f"{path}/metadata.json", "w") as f:
            json.dump(self.metadata, f)
        with open(f"{path}/tokenizer.pkl", "wb") as f:
            dill.dump(self.tokenizer, f)
        with open(f"{path}/trainer.pkl", "wb") as f:
            dill.dump(self.trainer, f)
        self.model.save(path, overwrite=overwrite, **kwargs)
        self.model.model.to(device)
