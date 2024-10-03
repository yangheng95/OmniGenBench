# -*- coding: utf-8 -*-
# file: batch_rgb_benchmark.py
# time: 13:26 04/06/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.

import argparse
import random
import torch
import torch.nn as nn

from omnigenome import AutoBench
from transformers import PreTrainedModel

class EvoModel(PreTrainedModel):
    def __init__(self, model_name='togethercomputer/evo-1-131k-base'):
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True, revision="1.1_fix")
        backbone = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            trust_remote_code=True,
            revision="1.1_fix"
        ).backbone
        super(EvoModel, self).__init__(config)

        self.backbone = backbone

    def forward(self, **kwargs):
        output = self.backbone(**kwargs)
        return {"last_hidden_state": output[0]}


if __name__ == "__main__":
    from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

    model_name = 'togethercomputer/evo-1-131k-base'
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = 'N'
    # tokenizer = AutoTokenizer.from_pretrained("genomic_foundation_models/OmniGenomeV4-186M")
    model = EvoModel(model_name)
    model.to('cuda')

    bench_root = "RGB"
    batch_size = 1
    gradient_accumulation_steps = 8
    max_length = 5
    patience = 3
    seeds = [3407]

    bench = AutoBench(
        autocast=True,
        bench_root=bench_root,
        model_name_or_path=model,
        tokenizer=tokenizer,
        overwrite=True
    )
    bench.run(
        batch_size=batch_size,
        max_length=max_length,
        patience=patience,
        # structure_in=True,
        gradient_accumulation_steps=batch_size//gradient_accumulation_steps,
        seeds=seeds
    )
