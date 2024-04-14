# -*- coding: utf-8 -*-
# file: model_hub.py
# time: 18:13 12/04/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.
import json
import os

import torch
from transformers import AutoConfig, AutoModel

from omnigenome.utility.hub_utils import query_models_info, download_model
from ...src.misc.utils import env_meta_info, fprint


class ModelHub:
    def __init__(self, *args, **kwargs):
        super(ModelHub, self).__init__(*args, **kwargs)

        self.metadata = env_meta_info()

    @staticmethod
    def load_model_and_tokenizer(model_name_or_path, local_only=False, **kwargs):
        model = ModelHub.load(model_name_or_path, local_only=local_only, **kwargs)
        fprint(f"The model and tokenizer has been loaded from {model_name_or_path}.")
        return model, model.tokenizer

    @staticmethod
    def load(model_name_or_path, local_only=False, **kwargs):
        if isinstance(model_name_or_path, str) and os.path.exists(model_name_or_path):
            path = model_name_or_path
        elif isinstance(model_name_or_path, str) and not os.path.exists(
            model_name_or_path
        ):
            path = download_model(model_name_or_path, local_only=local_only, **kwargs)
        else:
            raise ValueError("model_name_or_path must be a string.")
        import dill
        import importlib

        config = AutoConfig.from_pretrained(path, trust_remote_code=True, **kwargs)

        with open(f"{path}/metadata.json", "r", encoding="utf8") as f:
            metadata = json.load(f)

        config.metadata = metadata
        base_model = AutoModel.from_config(config, trust_remote_code=True, **kwargs)
        model_lib = importlib.import_module(metadata["library_name"].lower()).model
        model_cls = getattr(model_lib, metadata["model_cls"])

        with open(f"{path}/tokenizer.pkl", "rb") as f:
            tokenizer = dill.load(f)

        model = model_cls(config, base_model, tokenizer, **kwargs)

        with open(f"{path}/pytorch_model.bin", "rb") as f:
            model.load_state_dict(
                torch.load(f, map_location=kwargs.get("device", "cpu")), strict=False
            )
            model.metadata.update(metadata)
        return model

    @staticmethod
    def save(self, path, overwrite=False, **kwargs):
        import dill

        if os.path.exists(path) and not overwrite:
            raise FileExistsError(
                f"The path {path} already exists, please set overwrite=True to overwrite it."
            )

        if not os.path.exists(path):
            os.makedirs(path)

        device = self.model.device

        self.model.to("cpu")
        with open(f"{path}/model.pkl", "wb") as f:
            dill.dump(self, f)
        with open(f"{path}/tokenizer.pkl", "wb") as f:
            dill.dump(self.tokenizer, f)
        self.config.metadata = self.metadata
        self.config.save_pretrained(path)

        self.model.to(device)

        fprint(f"The model and tokenizer has been saved to {path}.")

    def available_models(
        self, model_name_or_path=None, local_only=False, repo="", **kwargs
    ):
        models_info = query_models_info(
            model_name_or_path, local_only=local_only, repo=repo, **kwargs
        )
        return models_info

    def push(self, model, **kwargs):
        raise NotImplementedError("This method has not implemented yet.")