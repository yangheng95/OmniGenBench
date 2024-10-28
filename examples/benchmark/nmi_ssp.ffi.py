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
    # gfms = [
    #     # "yangheng/PlantRNA-FM",
    #     # "genomic_foundation_models/OmniGenome-52M-1000",
    #     # "kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16",
    #     # "genomic_foundation_models/OmniGenomeV3-186M",
    #     "genomic_foundation_models/OmniGenomeV5-186M",
    #
    #     # "multimolecule/rnamsm",
    #     # "multimolecule/rnafm",
    #     # "multimolecule/rnabert",
    #     # "facebook/esm2_t12_35M_UR50D",
    #
    #     # "genomic_foundation_models/OmniGenomeV3-186M",
    #
    #     # "anonymous8/OmniGenome-52M",
    #     # "genomic_foundation_models/OmniGenome-418M",
    #
    #     # "genomic_foundation_models/SpliceBERT-510nt",
    #     # "genomic_foundation_models/DNABERT-2-117M",
    #     # "genomic_foundation_models/3utrbert",
    #     # "genomic_foundation_models/hyenadna-large-1m-seqlen-hf",
    #     # "genomic_foundation_models/nucleotide-transformer-v2-100m-multi-species",
    # ]
    gfms = [
        # "multimolecule/rnamsm",
        # "multimolecule/rnafm",
        "yangheng/PlantRNA-FM",
        "multimolecule/rnabert",
        "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
        "facebook/esm2_t12_35M_UR50D",
        # "genomic_foundation_models/esm2_rna_35M",
        # "genomic_foundation_models/esm2_rna_35M_ss",
        # "genomic_foundation_models/splicebert/SpliceBERT-510nt",
        # "genomic_foundation_models/cdsBERT",
        # "genomic_foundation_models/3utrbert",
        # "genomic_foundation_models/hyenadna-large-1m-seqlen-hf",
        "genomic_foundation_models/DNABERT-2-117M",
    ]
    bench_root = "RGB"
    batch_size = 8
    patience = 3
    max_seq_length = 440 if "rnabert" in gfms[0] else 512
    seeds = [3401, 3402, 3403, 3404, 3405]
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
            overwrite=False,
            # use_hf_trainer=True
        )
        bench.run(
            batch_size=batch_size,
            patience=patience,
            seeds=seeds
        )
