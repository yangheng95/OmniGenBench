# -*- coding: utf-8 -*-
# file: __init__.py
# time: 14:53 06/04/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.

"""
OmniGenome - Alias package for omnigenbench
===========================================

This package provides the same functionality as omnigenbench but with the omnigenome name.
All imports are redirected to the omnigenbench package.

For backward compatibility, this package maintains the same API as omnigenbench.
"""

# Package metadata (define locally to avoid circular imports)
__name__ = "omnigenome"
__version__ = "1.0.0beta"
__author__ = "Yang, Heng"
__email__ = "yangheng2021@gmail.com"
__license__ = "Apache-2.0"

import warnings

warnings.warn(
    "The 'omnigenome' package is deprecated, please use omnigenbench package instead. "
    "e.g., from omnigenome import *  ->  from omnigenbench import *\n"
    "All imports from omnigenome will be redirected to omnigenbench. ",
    DeprecationWarning,
)

# Import strategy: Try to import from omnigenbench with proper error handling
try:
    # Import core auto components
    from omnigenbench.auto.auto_bench.auto_bench import AutoBench
    from omnigenbench.auto.config.auto_config import AutoConfig
    from omnigenbench.auto.bench_hub.bench_hub import BenchHub
    from omnigenbench.auto.auto_train.auto_train import AutoTrain
    from omnigenbench.auto.auto_bench.auto_bench_cli import run_bench, bench_command
    from omnigenbench.auto.auto_train.auto_train_cli import run_train, train_command

    # Import source modules
    from omnigenbench.src import dataset, metric, model, tokenizer

    # Import abstract base classes
    from omnigenbench.src.abc.abstract_dataset import OmniDataset
    from omnigenbench.src.abc.abstract_metric import OmniMetric
    from omnigenbench.src.abc.abstract_model import OmniModel
    from omnigenbench.src.abc.abstract_tokenizer import OmniTokenizer

    # Import dataset classes
    from omnigenbench.src.dataset.omni_dataset import (
        OmniDatasetForSequenceClassification,
        OmniDatasetForSequenceRegression,
        OmniDatasetForTokenClassification,
        OmniDatasetForTokenRegression,
    )

    # Import metric classes
    from omnigenbench.src.metric import (
        ClassificationMetric,
        RegressionMetric,
        RankingMetric,
    )

    # Import utility functions
    from omnigenbench.src.misc.utils import (
        clean_temp_dir_pt_files,
        fprint,
        seed_everything,
        save_args,
        naive_secondary_structure_repair,
        check_bench_version,
        clean_temp_checkpoint,
        print_args,
        env_meta_info,
        RNA2StructureCache,
    )

    # Import model classes
    from omnigenbench.src.model import (
        OmniModelForSequenceClassification,
        OmniModelForMultiLabelSequenceClassification,
        OmniModelForTokenClassification,
        OmniModelForSequenceRegression,
        OmniModelForTokenRegression,
        OmniModelForStructuralImputation,
        OmniModelForMatrixRegression,
        OmniModelForMatrixClassification,
        OmniModelForMLM,
        OmniModelForSeq2Seq,
        OmniModelForRNADesign,
        OmniModelForEmbedding,
        OmniModelForAugmentation,
    )

    # Import LoRA model
    from omnigenbench.src.lora.lora_model import OmniLoraModel

    # Import tokenizer classes
    from omnigenbench.src.tokenizer import (
        OmniBPETokenizer,
        OmniKmersTokenizer,
        OmniSingleNucleotideTokenizer,
    )

    # Import trainer classes
    from omnigenbench.src.trainer.hf_trainer import HFTrainer
    from omnigenbench.src.trainer.trainer import Trainer
    from omnigenbench.src.trainer.accelerate_trainer import AccelerateTrainer

    # Import hub utilities
    from omnigenbench.utility.hub_utils import (
        download_benchmark,
        download_model,
        download_pipeline,
        query_models_info,
    )
    from omnigenbench.utility import hub_utils

    # Import hub classes
    from omnigenbench.utility.model_hub.model_hub import ModelHub
    from omnigenbench.utility.dataset_hub.dataset_hub import load_benchmark_datasets
    from omnigenbench.utility.pipeline_hub.pipeline import Pipeline
    from omnigenbench.utility.pipeline_hub.pipeline_hub import PipelineHub

    # Import module utilities
    from omnigenbench.src.model.module_utils import OmniPooling
    from omnigenbench.utility.ensemble import VoteEnsemblePredictor

    # For backward compatibility version 0.2.7alpha and earlier
    from omnigenbench.auto.config.auto_config import AutoBenchConfig

    # Create backward compatibility aliases
    OmniGenomeTokenizer = OmniTokenizer
    OmniGenomeKmersTokenizer = OmniKmersTokenizer
    OmniGenomeSingleNucleotideTokenizer = OmniSingleNucleotideTokenizer
    OmniGenomeBPETokenizer = OmniBPETokenizer
    OmniGenomeDataset = OmniDataset
    OmniGenomeMetric = OmniMetric
    OmniGenomeModel = OmniModel
    OmniGenomeDatasetForSequenceClassification = OmniDatasetForSequenceClassification
    OmniGenomeDatasetForSequenceRegression = OmniDatasetForSequenceRegression
    OmniGenomeDatasetForTokenClassification = OmniDatasetForTokenClassification
    OmniGenomeDatasetForTokenRegression = OmniDatasetForTokenRegression
    OmniGenomeLoraModel = OmniLoraModel
    OmniGenomeModelForSequenceClassification = OmniModelForSequenceClassification
    OmniGenomeModelForMultiLabelSequenceClassification = (
        OmniModelForMultiLabelSequenceClassification
    )
    OmniGenomeModelForTokenClassification = OmniModelForTokenClassification
    OmniGenomeModelForSequenceRegression = OmniModelForSequenceRegression
    OmniGenomeModelForTokenRegression = OmniModelForTokenRegression
    OmniGenomeModelForStructuralImputation = OmniModelForStructuralImputation
    OmniGenomeModelForMatrixRegression = OmniModelForMatrixRegression
    OmniGenomeModelForMatrixClassification = OmniModelForMatrixClassification
    OmniGenomeModelForMLM = OmniModelForMLM
    OmniGenomeModelForSeq2Seq = OmniModelForSeq2Seq
    OmniGenomeModelForRNADesign = OmniModelForRNADesign
    OmniGenomeModelForEmbedding = OmniModelForEmbedding
    OmniGenomeModelForAugmentation = OmniModelForAugmentation

    # Define __all__ for explicit exports
    __all__ = [
        "load_benchmark_datasets",
        "OmniDataset",
        "OmniModel",
        "OmniMetric",
        "OmniTokenizer",
        "OmniKmersTokenizer",
        "OmniSingleNucleotideTokenizer",
        "OmniBPETokenizer",
        "ModelHub",
        "Pipeline",
        "PipelineHub",
        "BenchHub",
        "AutoBench",
        "AutoTrain",
        "AutoConfig",
        "ClassificationMetric",
        "RegressionMetric",
        "RankingMetric",
        "Trainer",
        "HFTrainer",
        "AccelerateTrainer",
        "AutoBenchConfig",
        "download_benchmark",
        "download_model",
        "download_pipeline",
        "VoteEnsemblePredictor",
        "clean_temp_dir_pt_files",
        "fprint",
        "seed_everything",
        "save_args",
        "naive_secondary_structure_repair",
        "check_bench_version",
        "clean_temp_checkpoint",
        "print_args",
        "env_meta_info",
        "RNA2StructureCache",
        # OmniGenome* aliases for backward compatibility
        "OmniGenomeTokenizer",
        "OmniGenomeKmersTokenizer",
        "OmniGenomeSingleNucleotideTokenizer",
        "OmniGenomeBPETokenizer",
        "OmniGenomeDataset",
        "OmniGenomeMetric",
        "OmniGenomeModel",
        "OmniGenomeDatasetForSequenceClassification",
        "OmniGenomeDatasetForSequenceRegression",
        "OmniGenomeDatasetForTokenClassification",
        "OmniGenomeDatasetForTokenRegression",
        "OmniGenomeLoraModel",
        "OmniGenomeModelForSequenceClassification",
        "OmniGenomeModelForMultiLabelSequenceClassification",
        "OmniGenomeModelForTokenClassification",
        "OmniGenomeModelForSequenceRegression",
        "OmniGenomeModelForTokenRegression",
        "OmniGenomeModelForStructuralImputation",
        "OmniGenomeModelForMatrixRegression",
        "OmniGenomeModelForMatrixClassification",
        "OmniGenomeModelForMLM",
        "OmniGenomeModelForSeq2Seq",
        "OmniGenomeModelForRNADesign",
        "OmniGenomeModelForEmbedding",
        "OmniGenomeModelForAugmentation",
        # Command line functions
        "run_bench",
        "bench_command",
        "run_train",
        "train_command",
    ]

except ImportError as e:
    import warnings

    warnings.warn(
        f"Failed to import omnigenbench modules: {e}. "
        "Please ensure omnigenbench is properly installed.\n"
        "You can install it with: pip install omnigenbench\n"
        "and replace all 'omnigenome' with 'omnigenbench' in your code.\n"
        "e.g., from omnigenome import *  ->  from omnigenbench import *",
        ImportWarning,
    )

    # Minimal fallback to prevent complete failure
    __all__ = []
