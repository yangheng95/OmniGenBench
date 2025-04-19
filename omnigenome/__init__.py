# -*- coding: utf-8 -*-
# file: __init__.py
# time: 14:53 06/04/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.

__name__ = "OmniGenome"
__version__ = "0.2.4alpha3"
__author__ = "YANG, HENG"
__email__ = "yangheng2021@gmail.com"
__license__ = "MIT"

from .auto.auto_bench.auto_bench import AutoBench
from .auto.auto_bench.auto_bench_config import AutoBenchConfig
from .auto.bench_hub.bench_hub import BenchHub
from .auto.auto_train.auto_train import AutoTrain
from .auto.auto_bench.auto_bench_cli import run_bench, bench_command
from .auto.auto_train.auto_train_cli import run_train, train_command

from .src import dataset as dataset
from .src import metric as metric
from .src import model as model
from .src import tokenizer as tokenizer
from .src.abc.abstract_dataset import OmniGenomeDataset
from .src.abc.abstract_metric import OmniGenomeMetric
from .src.abc.abstract_model import OmniGenomeModel
from .src.abc.abstract_tokenizer import OmniGenomeTokenizer
from .src.abc.abstract_tokenizer import OmniGenomeTokenizer as AutoTokenizer
from .src.dataset.omnigenome_dataset import OmniGenomeDatasetForSequenceClassification
from .src.dataset.omnigenome_dataset import OmniGenomeDatasetForSequenceRegression
from .src.dataset.omnigenome_dataset import OmniGenomeDatasetForTokenClassification
from .src.dataset.omnigenome_dataset import OmniGenomeDatasetForTokenRegression
from .src.metric import ClassificationMetric, RegressionMetric, RankingMetric
from .src.misc import utils as utils
from .src.model import (
    OmniGenomeModelForSequenceClassification,
    OmniGenomeModelForMultiLabelSequenceClassification,
    OmniGenomeModelForTokenClassification,
    OmniGenomeModelForSequenceRegression,
    OmniGenomeModelForTokenRegression,
    OmniGenomeModelForStructuralImputation,
    OmniGenomeModelForMatrixRegression,
    OmniGenomeModelForMatrixClassification,
    OmniGenomeModelForMLM,
    OmniGenomeModelForSeq2Seq,
    OmniGenomeModelForRNADesign,
    OmniGenomeModelForEmbedding,
    OmniGenomeModelForAugmentation,
)
from .src.tokenizer import OmniBPETokenizer
from .src.tokenizer import OmniKmersTokenizer
from .src.tokenizer import OmniSingleNucleotideTokenizer
from .src.trainer.hf_trainer import HFTrainer
from .src.trainer.trainer import Trainer
from .src.trainer.accelerate_trainer import AccelerateTrainer

from .utility.hub_utils import download_benchmark
from .utility.hub_utils import download_model
from .utility.hub_utils import download_pipeline

from .utility import hub_utils as hub_utils
from .utility.model_hub.model_hub import ModelHub
from .utility.pipeline_hub.pipeline import Pipeline
from .utility.pipeline_hub.pipeline_hub import PipelineHub

from .src.model.module_utils import OmniGenomePooling


__all__ = [
    "OmniGenomeDataset",
    "OmniGenomeModel",
    "OmniGenomeMetric",
    "AutoTokenizer",
    "OmniGenomeTokenizer",
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
    "OmniGenomeDatasetForTokenClassification",
    "OmniGenomeDatasetForTokenRegression",
    "OmniGenomeDatasetForSequenceClassification",
    "OmniGenomeDatasetForSequenceRegression",
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
]


LOGO1 = r"""                       
    **@@ #========= @@**            ___                     _ 
      **@@ +----- @@**             / _ \  _ __ ___   _ __  (_)
        **@@ = @@**               | | | || '_ ` _ \ | '_ \ | |
           **@@                   | |_| || | | | | || | | || |
        @@** = **@@                \___/ |_| |_| |_||_| |_||_|
     @@** ------+ **@@                
   @@** =========# **@@            ____  
  @@ ---------------+ @@          / ___|  ___  _ __    ___   _ __ ___    ___ 
 @@ ================== @@        | |  _  / _ \| '_ \  / _ \ | '_ ` _ \  / _ \
  @@ +--------------- @@         | |_| ||  __/| | | || (_) || | | | | ||  __/ 
   @@** #========= **@@           \____| \___||_| |_| \___/ |_| |_| |_| \___| 
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
        *@@*   **               / ___|  ___  _ __    ___   _ __ ___    ___ 
          **@**                | |  _  / _ \| '_ \  / _ \ | '_ ` _ \  / _ \ 
        *@@* *@@*              | |_| ||  __/| | | || (_) || | | | | ||  __/ 
      *@@ ---+  @@*             \____| \___||_| |_| \___/ |_| |_| |_| \___| 
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
