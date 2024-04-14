# -*- coding: utf-8 -*-
# file: __init__.py
# time: 14:53 06/04/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.

__name__ = "OmniGenome"
__version__ = "0.0.1alpha0"
__author__ = "YANG, HENG"
__email__ = "yangheng2021@gmail.com"
__license__ = "MIT"


from .src.abc.abstract_dataset import OmniGenomeDataset
from .src.abc.abstract_model import OmniGenomeModel
from .src.abc.abstract_metric import OmniGenomeMetric
from .src.abc.abstract_tokenizer import OmniGenomeTokenizer

from .src.tokenizer import OmniKmersTokenizer
from .src.tokenizer import OmniSingleNucleotideTokenizer
from .src.tokenizer import OmniBPETokenizer

# from .src import config as config  # no development yet

from .utility import hub_utils as hub_utils
from .utility.model_hub.model_hub import ModelHub

from .utility.pipeline_hub.pipeline import Pipeline
from .utility.pipeline_hub.pipeline_hub import PipelineHub

from .bench.bench_hub.bench_hub import BenchHub
from .bench.auto_bench.auto_bench import AutoBench
from .bench.auto_bench.auto_bench import AutoBenchConfig

from .src.misc import utils as utils

from .src.trainer.trainer import Trainer
from .src.trainer.hf_trainer import HFTrainer

from .src import config as config
from .src import metric as metric
from .src import model as model
from .src import tokenizer as tokenizer
from .src import dataset as dataset

from .src.model import (
    OmniGenomeEncoderModelForSequenceClassification,
    OmniGenomeEncoderModelForTokenClassification,
    OmniGenomeEncoderModelForSequenceClassificationWith2DStructure,
    OmniGenomeEncoderModelForTokenClassificationWith2DStructure,
    OmniGenomeDecoderModelForSequenceClassification,
    OmniGenomeDecoderModelForTokenClassification,
    OmniGenomeDecoderModelForSequenceClassificationWith2DStructure,
    OmniGenomeDecoderModelForTokenClassificationWith2DStructure,

    OmniGenomeEncoderModelForSequenceRegression,
    OmniGenomeEncoderModelForTokenRegression,
    OmniGenomeEncoderModelForSequenceRegressionWith2DStructure,
    OmniGenomeEncoderModelForTokenRegressionWith2DStructure,
    OmniGenomeDecoderModelForSequenceRegression,
    OmniGenomeDecoderModelForTokenRegression,
    OmniGenomeDecoderModelForSequenceRegressionWith2DStructure,
    OmniGenomeDecoderModelForTokenRegressionWith2DStructure,
    OmniGenomeEncoderModelForMLM,
    OmniGenomeEncoderModelForSeq2Seq,
)

from .src.dataset.omnigenome_dataset import OmniGenomeDatasetForTokenClassification
from .src.dataset.omnigenome_dataset import OmniGenomeDatasetForTokenRegression
from .src.dataset.omnigenome_dataset import OmniGenomeDatasetForSequenceClassification
from .src.dataset.omnigenome_dataset import OmniGenomeDatasetForSequenceRegression

from .src.metric import ClassificationMetric, RegressionMetric, RankingMetric
__all__ = [
    "OmniGenomeDataset",
    "OmniGenomeModel",
    "OmniGenomeMetric",
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
    "config",
    "metric",
    "model",
    "tokenizer",
    "dataset",
    "OmniGenomeEncoderModelForSequenceClassification",
    "OmniGenomeEncoderModelForTokenClassification",
    "OmniGenomeEncoderModelForSequenceClassificationWith2DStructure",
    "OmniGenomeEncoderModelForTokenClassificationWith2DStructure",
    "OmniGenomeDecoderModelForSequenceClassification",
    "OmniGenomeDecoderModelForTokenClassification",
    "OmniGenomeDecoderModelForSequenceClassificationWith2DStructure",
    "OmniGenomeDecoderModelForTokenClassificationWith2DStructure",
    "OmniGenomeEncoderModelForSequenceRegression",
    "OmniGenomeEncoderModelForTokenRegression",
    "OmniGenomeEncoderModelForSequenceRegressionWith2DStructure",
    "OmniGenomeEncoderModelForTokenRegressionWith2DStructure",
    "OmniGenomeDecoderModelForSequenceRegression",
    "OmniGenomeDecoderModelForTokenRegression",
    "OmniGenomeDecoderModelForSequenceRegressionWith2DStructure",
    "OmniGenomeDecoderModelForTokenRegressionWith2DStructure",
    "OmniGenomeEncoderModelForMLM",
    "OmniGenomeEncoderModelForSeq2Seq",

    "OmniGenomeDatasetForTokenClassification",
    "OmniGenomeDatasetForTokenRegression",
    "OmniGenomeDatasetForSequenceClassification",
    "OmniGenomeDatasetForSequenceRegression",

    "ClassificationMetric",
    "RegressionMetric",
    "RankingMetric",

    "Trainer",
    "HFTrainer",
]

