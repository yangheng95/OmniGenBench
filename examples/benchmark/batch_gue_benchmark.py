# -*- coding: utf-8 -*-
# file: batch_rgb_benchmark.py
# time: 13:26 04/06/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.


import random

from omnigenome import AutoBench
from transformers import AutoTokenizer

if __name__ == "__main__":
    gfms = [
        "genomic_foundation_models/OmniGenomeV3-186M",
        # "anonymous8/OmniGenome-52M",
        # "kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16",
        # "multimolecule/rnamsm",
        # "multimolecule/rnafm",
        # "multimolecule/rnabert",
        # "genomic_foundation_models/SpliceBERT ",
        # "genomic_foundation_models/DNABERT-2-117M",
        # "genomic_foundation_models/3utrbert",
        # "genomic_foundation_models/hyenadna-large-1m-seqlen-hf",
        # "genomic_foundation_models/nucleotide-transformer-v2-100m-multi-species",
    ]
    bench_root = "GUE"
    batch_size = 8
    target_batch_size = 256
    epoch = 10
    gradient_accumulation_steps = target_batch_size // batch_size
    max_examples = 10000
    max_length = 512
    patience = 10
    seeds = [random.randint(0, 1000000) for _ in range(1)]
    for gfm in gfms:
        if 'multimolecule' in gfm:
            from multimolecule import RnaTokenizer, AutoModelForTokenPrediction
            tokenizer = RnaTokenizer.from_pretrained(gfm)
            gfm = AutoModelForTokenPrediction.from_pretrained(gfm, trust_remote_code=True).base_model
        else:
            tokenizer = AutoTokenizer.from_pretrained(gfm)
        bench = AutoBench(
            autocast="fp16",
            bench_root=bench_root,
            model_name_or_path=gfm,
            tokenizer=tokenizer,
            overwrite=True,
            use_hf_trainer=True,
            # use_hf_trainer=False,
        )
        bench.run(
            batch_size=batch_size,
            epochs=epoch,
            max_length=max_length,
            max_examples=max_examples,
            gradient_accumulation_steps=gradient_accumulation_steps,
            patience=patience,
            seeds=seeds
        )

