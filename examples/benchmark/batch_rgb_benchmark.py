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
        "genomic_foundation_models/OmniGenomeV3-186M",
        # "genomic_foundation_models/OmniGenomeV2-186M",
        # "anonymous8/OmniGenome-186M",
        # "genomic_foundation_models/SpliceBERT-510nt",
        # "genomic_foundation_models/DNABERT-2-117M",
        # "genomic_foundation_models/3utrbert",
        # "genomic_foundation_models/hyenadna-large-1m-seqlen-hf",
        # "genomic_foundation_models/nucleotide-transformer-v2-100m-multi-species",
    ]
    bench_root = "RGB"
    bench_size = 8
    seeds = [3407, 3408, 3409]
    for gfm in gfms:
        bench = AutoBench(
            bench_root=bench_root, model_name_or_path=gfm, overwrite=True
        )
        bench.run(autocast=False, batch_size=bench_size, seeds=seeds)
