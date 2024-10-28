# -*- coding: utf-8 -*-
# file: batch_rgb_benchmark.py
# time: 13:26 04/06/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.

from omnigenome import AutoBench

if __name__ == "__main__":
    gfms = [
        # "genomic_foundation_models/OmniGenomeV3-186M",
        # "kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16",
        # "multimolecule/rnamsm",
        # "multimolecule/rnafm",
        # "multimolecule/rnabert",
        # "InstaDeepAI/agro-nucleotide-transformer-1b",
        "genomic_foundation_models/SpliceBERT-510nt",
        "genomic_foundation_models/DNABERT-2-117M",
        "genomic_foundation_models/3utrbert",
        "genomic_foundation_models/hyenadna-large-1m-seqlen-hf",
        "genomic_foundation_models/nucleotide-transformer-v2-100m-multi-species",
    ]
    bench_root = "GUE"
    batch_size = 4
    max_length = 512
    gradient_accumulation_steps = 64
    max_examples = 100000
    patience = 3
    seeds = [3401]
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
            overwrite=True,
            use_hf_trainer=True,
        )
        bench.run(
            batch_size=batch_size,
            max_length=max_length,
            max_examples=max_examples,
            patience=patience,
            seeds=seeds,
            gradient_accumulation_steps=gradient_accumulation_steps
        )

