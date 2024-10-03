# -*- coding: utf-8 -*-
# file: pipeline.py
# time: 18:38 12/04/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.
import json
import os

import autocuda
from transformers import AutoConfig, AutoTokenizer

from ..hub_utils import download_pipeline
from ..model_hub.model_hub import ModelHub
from ...src.abc.abstract_model import OmniGenomeModel
from ...src.misc.utils import env_meta_info, fprint
from ...src.trainer.trainer import Trainer


class Pipeline:
    model: OmniGenomeModel = None
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
        return self.model.inference(inputs, **kwargs)

    def to(self, device):
        self.model.to(device)
        self.device = device

    def init_pipeline(self, *, model_name_or_path, tokenizer=None, **kwargs):
        trust_remote_code = kwargs.get("trust_remote_code", True)
        try:  # for the models saved by OmniGenome and served by the model hub
            self.model = ModelHub.load(model_name_or_path, **kwargs)
            self.tokenizer = self.model.tokenizer
            self.metadata.update(self.model.metadata)
        except Exception as e:
            print(f"Fail to load the model from the model hub, the error is: {e}")

            config = AutoConfig.from_pretrained(
                model_name_or_path, trust_remote_code=trust_remote_code
            )
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name_or_path, trust_remote_code=trust_remote_code
                )
            self.model = OmniGenomeModel.from_pretrained(
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
        if trainer is not None:
            assert isinstance(trainer, Trainer)
            self.trainer = trainer

        self.trainer.train()

    def predict(self, inputs, **kwargs):
        return self.model.predict(inputs, **kwargs)

    def inference(self, inputs, **kwargs):
        return self.model.inference(inputs, **kwargs)

    @staticmethod
    def load(pipeline_name_or_path, local_only=False, **kwargs):
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
