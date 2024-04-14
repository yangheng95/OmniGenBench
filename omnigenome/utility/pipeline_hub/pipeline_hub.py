# -*- coding: utf-8 -*-
# file: pipeline_hub.py
# time: 22:26 08/04/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.

from .pipeline import Pipeline
from ...src.misc.utils import env_meta_info


class PipelineHub:
    def __init__(self, *args, **kwargs):
        super(PipelineHub, self).__init__(*args, **kwargs)
        self.metadata = env_meta_info()

    @staticmethod
    def load(pipeline_name_or_path, local_only=False, **kwargs):
        return Pipeline.load(pipeline_name_or_path, local_only=local_only, **kwargs)

    def push(self, pipeline, **kwargs):
        raise NotImplementedError("This method has not implemented yet.")
