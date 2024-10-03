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

from omnigenome import AutoBench

if __name__ == "__main__":
    gfms = [
        # "genomic_foundation_models/OmniGenome-52M-1000",
        "genomic_foundation_models/OmniGenomeV5-186M",
        # "anonymous8/OmniGenome-186M",

        # "multimolecule/rnamsm",
        # "multimolecule/rnafm",
        # "multimolecule/rnabert",
        # "genomic_foundation_models/OmniGenomeV3-186M",

        # "anonymous8/OmniGenome-52M",
        # "genomic_foundation_models/OmniGenome-418M",

        # "genomic_foundation_models/SpliceBERT-510nt",
        # "genomic_foundation_models/DNABERT-2-117M",
        # "genomic_foundation_models/3utrbert",
        # "genomic_foundation_models/hyenadna-large-1m-seqlen-hf",
        # "genomic_foundation_models/nucleotide-transformer-v2-100m-multi-species",
    ]
    bench_root = "RGB+"
    batch_size = 4
    gradient_accumulation_steps = 8
    max_length = 512
    patience = 3
    seeds = [3407]
    for gfm in gfms:
        if 'multimolecule' in gfm:
            from multimolecule import RnaTokenizer, AutoModelForTokenPrediction
            tokenizer = RnaTokenizer.from_pretrained(gfm)
            gfm = AutoModelForTokenPrediction.from_pretrained(gfm, trust_remote_code=True).base_model
        else:
            tokenizer = None
        bench = AutoBench(
            autocast="fp16",
            bench_root=bench_root,
            model_name_or_path=gfm,
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
