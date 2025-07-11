# -*- coding: utf-8 -*-
# file: pipeline_hub.py
# time: 22:26 08/04/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.
"""
Pipeline Hub Module

This module provides the PipelineHub class for managing and loading pre-built
pipelines from the OmniGenome hub. Pipelines combine models, tokenizers,
datasets, and trainers into ready-to-use workflows.
"""

from .pipeline import Pipeline
from ...src.misc.utils import env_meta_info


class PipelineHub:
    """
    Hub for managing and loading pre-built OmniGenome pipelines.
    
    The PipelineHub provides a centralized interface for accessing pre-built
    pipelines that combine models, tokenizers, datasets, and training
    configurations. It handles automatic downloading and loading of pipelines
    from the OmniGenome hub.
    
    Attributes:
        metadata (dict): Environment metadata including system information,
            package versions, and hardware details.
    
    Example:
        >>> from omnigenome import PipelineHub
        >>> hub = PipelineHub()
        >>> pipeline = hub.load("yangheng/OmniGenome-RNA-Classification")
        >>> predictions = pipeline("ATCGATCG")
        >>> print(predictions['predictions'])
    
    Note:
        - Pipelines can be loaded from local paths or downloaded from the hub
        - The hub automatically handles model, tokenizer, and dataset loading
        - Environment metadata is collected for reproducibility
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize the PipelineHub.
        
        Args:
            *args: Variable length argument list (currently unused).
            **kwargs: Arbitrary keyword arguments (currently unused).
        
        Note:
            The constructor initializes environment metadata for tracking
            system information and package versions.
        """
        super(PipelineHub, self).__init__(*args, **kwargs)
        self.metadata = env_meta_info()

    @staticmethod
    def load(pipeline_name_or_path, local_only=False, **kwargs):
        """
        Load a pipeline from the hub or local path.
        
        This method loads a complete pipeline including the model, tokenizer,
        datasets, and trainer configuration. If the pipeline doesn't exist
        locally and local_only is False, it will be downloaded from the hub.
        
        Args:
            pipeline_name_or_path (str): Name or path of the pipeline to load.
                Can be a local directory path or a hub identifier.
            local_only (bool, optional): If True, only load from local paths.
                If False, download from hub if not found locally. Defaults to False.
            **kwargs: Additional keyword arguments passed to the Pipeline constructor.
                Common options include:
                - device: Target device for the model
                - trust_remote_code: Whether to trust remote code in tokenizers
                - name: Custom name for the pipeline
        
        Returns:
            Pipeline: Loaded pipeline instance with model, tokenizer, datasets,
                and trainer ready for use.
        
        Raises:
            FileNotFoundError: If the pipeline cannot be found locally and
                local_only is True.
            ValueError: If the pipeline configuration is invalid.
            ImportError: If required dependencies are not available.
        
        Example:
            >>> hub = PipelineHub()
            >>> # Load from hub
            >>> pipeline = hub.load("yangheng/OmniGenome-RNA-Classification")
            >>> # Load from local path
            >>> pipeline = hub.load("./my_pipeline", local_only=True)
            >>> # Use pipeline for inference
            >>> results = pipeline("ATCGATCG")
        
        Note:
            - The pipeline includes all necessary components for training and inference
            - Model weights, tokenizer, and datasets are automatically loaded
            - The pipeline can be used immediately for inference or fine-tuning
        """
        return Pipeline.load(pipeline_name_or_path, local_only=local_only, **kwargs)

    def push(self, pipeline, **kwargs):
        """
        Push a pipeline to the hub (not yet implemented).
        
        This method is intended to upload custom pipelines to the OmniGenome hub
        for sharing and distribution. Currently not implemented.
        
        Args:
            pipeline (Pipeline): Pipeline instance to upload to the hub.
            **kwargs: Additional keyword arguments for the upload process.
        
        Raises:
            NotImplementedError: This method has not been implemented yet.
        
        Note:
            Future implementation will support:
            - Pipeline metadata and documentation
            - Model weights and configuration
            - Tokenizer and dataset specifications
            - Training configurations and results
        """
        raise NotImplementedError("This method has not implemented yet.")
