# -*- coding: utf-8 -*-
# file: __init__.py
# time: 14:53 06/04/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.

"""
This __init__.py file exposes the Key API Entries of the library for easy access.
Use dir(omnigenome) to see all available APIs.

Key API Entries:
----------------
- AutoBench: Automated benchmarking of genomic models
- AutoTrain: Automated training of genomic models
- BenchHub: Hub for accessing benchmarks
- ModelHub: Hub for accessing pre-trained models
- PipelineHub: Hub for accessing pipelines
- Various model classes for different genomic tasks
- Dataset classes for different data formats
- Tokenizer classes for different sequence representations
- Metric classes for evaluation
- Trainer classes for model training
"""

__name__ = "omnigenbench"
__version__ = "0.3.1alpha"

__author__ = "YANG, HENG"
__email__ = "yangheng2021@gmail.com"
__license__ = "Apache-2.0"

# Import core auto components
from .auto.auto_bench.auto_bench import AutoBench
from .auto.auto_bench.auto_bench_config import AutoBenchConfig
from .auto.bench_hub.bench_hub import BenchHub
from .auto.auto_train.auto_train import AutoTrain
from .auto.auto_bench.auto_bench_cli import run_bench, bench_command
from .auto.auto_train.auto_train_cli import run_train, train_command

# Import source modules
from .src import dataset as dataset
from .src import metric as metric
from .src import model as model
from .src import tokenizer as tokenizer

# Import abstract base classes
from .src.abc.abstract_dataset import OmniDataset
from .src.abc.abstract_metric import OmniMetric
from .src.abc.abstract_model import OmniModel
from .src.abc.abstract_tokenizer import OmniTokenizer
from .src.abc.abstract_tokenizer import OmniTokenizer as AutoTokenizer

# Import dataset classes
from .src.dataset.omni_dataset import OmniDatasetForSequenceClassification
from .src.dataset.omni_dataset import OmniDatasetForSequenceRegression
from .src.dataset.omni_dataset import OmniDatasetForTokenClassification
from .src.dataset.omni_dataset import OmniDatasetForTokenRegression

# Import metric classes
from .src.metric import ClassificationMetric, RegressionMetric, RankingMetric

# Import utility functions
from .src.misc import utils as utils
from .src.misc.utils import clean_temp_dir_pt_files

# Import model classes
from .src.model import (
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
from .src.lora.lora_model import OmniLoraModel

# Import tokenizer classes
from .src.tokenizer import OmniBPETokenizer
from .src.tokenizer import OmniKmersTokenizer
from .src.tokenizer import OmniSingleNucleotideTokenizer

# Import trainer classes
from .src.trainer.hf_trainer import HFTrainer
from .src.trainer.trainer import Trainer
from .src.trainer.accelerate_trainer import AccelerateTrainer

# Import hub utilities
from .utility.hub_utils import download_benchmark
from .utility.hub_utils import download_model
from .utility.hub_utils import download_pipeline
from .utility import hub_utils as hub_utils

# Import hub classes
from .utility.model_hub.model_hub import ModelHub
from .utility.dataset_hub.dataset_hub import load_benchmark_datasets
from .utility.pipeline_hub.pipeline import Pipeline
from .utility.pipeline_hub.pipeline_hub import PipelineHub

# Import module utilities
from .src.model.module_utils import OmniPooling

# --------------------------------------------------------------------------------
# For backward compatibility version 0.2.7alpha and earlier
from .src.abc.abstract_tokenizer import OmniTokenizer as OmniGenomeTokenizer
from .src.abc.abstract_dataset import OmniDataset as OmniGenomeDataset
from .src.abc.abstract_metric import OmniMetric as OmniGenomeMetric
from .src.abc.abstract_model import OmniModel as OmniGenomeModel
from .src.dataset.omni_dataset import (
    OmniDatasetForSequenceClassification as OmniGenomeDatasetForSequenceClassification,
)
from .src.dataset.omni_dataset import (
    OmniDatasetForSequenceRegression as OmniGenomeDatasetForSequenceRegression,
)
from .src.dataset.omni_dataset import (
    OmniDatasetForTokenClassification as OmniGenomeDatasetForTokenClassification,
)
from .src.dataset.omni_dataset import (
    OmniDatasetForTokenRegression as OmniGenomeDatasetForTokenRegression,
)
from .src.lora.lora_model import OmniLoraModel as OmniGenomeLoraModel
from .src.model import (
    OmniModelForSequenceClassification as OmniGenomeModelForSequenceClassification,
    OmniModelForMultiLabelSequenceClassification as OmniGenomeModelForMultiLabelSequenceClassification,
    OmniModelForTokenClassification as OmniGenomeModelForTokenClassification,
    OmniModelForSequenceRegression as OmniGenomeModelForSequenceRegression,
    OmniModelForTokenRegression as OmniGenomeModelForTokenRegression,
    OmniModelForStructuralImputation as OmniGenomeModelForStructuralImputation,
    OmniModelForMatrixRegression as OmniGenomeModelForMatrixRegression,
    OmniModelForMatrixClassification as OmniGenomeModelForMatrixClassification,
    OmniModelForMLM as OmniGenomeModelForMLM,
    OmniModelForSeq2Seq as OmniGenomeModelForSeq2Seq,
    OmniModelForRNADesign as OmniGenomeModelForRNADesign,
    OmniModelForEmbedding as OmniGenomeModelForEmbedding,
    OmniModelForAugmentation as OmniGenomeModelForAugmentation,
)

from .utility.ensemble import VoteEnsemblePredictor

# ------------------------------------------------------------------------------


__all__ = [
    "load_benchmark_datasets",
    "OmniDataset",
    "OmniModel",
    "OmniMetric",
    "AutoTokenizer",
    "OmniTokenizer",
    "OmniKmersTokenizer",
    "OmniSingleNucleotideTokenizer",
    "OmniBPETokenizer",
    "ModelHub",
    "Pipeline",
    "PipelineHub",
    "BenchHub",
    "AutoBench",
    "AutoBenchConfig",
    "utils",
    "model",
    "tokenizer",
    "dataset",
    "OmniModelForSequenceClassification",
    "OmniModelForMultiLabelSequenceClassification",
    "OmniModelForTokenClassification",
    "OmniModelForSequenceRegression",
    "OmniModelForTokenRegression",
    "OmniModelForRNADesign",
    "OmniModelForEmbedding",
    "OmniModelForAugmentation",
    "OmniModelForStructuralImputation",
    "OmniModelForMatrixRegression",
    "OmniModelForMatrixClassification",
    "OmniModelForMLM",
    "OmniModelForSeq2Seq",
    "OmniDatasetForTokenClassification",
    "OmniDatasetForTokenRegression",
    "OmniDatasetForSequenceClassification",
    "OmniDatasetForSequenceRegression",
    "OmniLoraModel",
    "ClassificationMetric",
    "RegressionMetric",
    "RankingMetric",
    "Trainer",
    "HFTrainer",
    "AccelerateTrainer",
    "AutoBenchConfig",
    "AutoBench",
    "download_benchmark",
    "download_model",
    "download_pipeline",
    "VoteEnsemblePredictor",
]


LOGO1 = r"""                       
    **@@ #========= @@**            ___                     _ 
      **@@ +----- @@**             / _ \  _ __ ___   _ __  (_)
        **@@ = @@**               | | | || '_ ` _ \ | '_ \ | |
           **@@                   | |_| || | | | | || | | || |
        @@** = **@@                \___/ |_| |_| |_||_| |_||_|
     @@** ------+ **@@                
   @@** =========# **@@            ____  
  @@ ---------------+ @@          / ___|  ___  _ __   
 @@ ================== @@        | |  _  / _ \| '_ \
  @@ +--------------- @@         | |_| ||  __/| | | |
   @@** #========= **@@           \____| \___||_| |_| 
    @@** +------ **@@          
       @@** = **@@           
          @@**                    ____                      _   
       **@@ = @@**               | __ )   ___  _ __    ___ | |__  
    **@@ -----+  @@**            |  _ \  / _ \| '_ \  / __|| '_ \ 
  **@@ ==========# @@**          | |_) ||  __/| | | || (__ | | | |
  @@ --------------+ @@**        |____/  \___||_| |_| \___||_| |_|
"""

LOGO2 = r"""

   **  +----------- **           ___                     _ 
  @@                 @@         / _ \  _ __ ___   _ __  (_)
 @@* #============== *@@       | | | || '_ ` _ \ | '_ \ | |
 @@*                 *@@       | |_| || | | | | || | | || |
 *@@  +------------ *@@         \___/ |_| |_| |_||_| |_||_|
  *@*               @@*       
   *@@  #========= @@*        
    *@@*         *@@*          
      *@@  +---@@@*              ____  
        *@@*   **               / ___|  ___  _ __  
          **@**                | |  _  / _ \| '_ \ 
        *@@* *@@*              | |_| ||  __/| | | |
      *@@ ---+  @@*             \____| \___||_| |_|
    *@@*         *@@*          
   *@@ =========#  @@*         
  *@@               @@*        
 *@@ -------------+  @@*        ____                      _   
 @@                   @@       | __ )   ___  _ __    ___ | |__ 
 @@ ===============#  @@       |  _ \  / _ \| '_ \  / __|| '_ \ 
  @@                 @@        | |_) ||  __/| | | || (__ | | | |
   ** -----------+  **         |____/  \___||_| |_| \___||_| |_|
"""

art_dna_color_map = {
    "*": "blue",  # Bases represented by '*'
    "@": "white",  # Bases represented by '@'
    "-": "yellow",  # Hydrogen bonds, assuming '-' represents a bond
    "=": "light_cyan",  # Hydrogen bonds, assuming '=' represents a bond
    "+": "yellow",  # '+' symbols in cyan
    " ": "black",  # Use red for undefined characters
}
import random

LOGO = random.choice([LOGO1, LOGO2])
print(LOGO)

clean_temp_dir_pt_files()
