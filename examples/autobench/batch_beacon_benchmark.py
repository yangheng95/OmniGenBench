# -*- coding: utf-8 -*-
# file: batch_beacon_benchmark.py
# time: 13:26 04/06/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.
import random

from omnigenome import AutoBench
if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--gfm", type=str, default="genomic_foundation_models/OmniGenome-52M-1000")
    parser.add_argument("--benchmark", type=str, default="BEACON")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seeds", type=int, default=random.randint(0, 1000))
    parser.add_argument("--deepspeed", default="ds_config.json")
    args = parser.parse_args()

    gfms = [
        # "genomic_foundation_models/OmniGenomeV3-186M",
        # "kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16",
        # "multimolecule/rnamsm",
        "multimolecule/rnafm",
        # "multimolecule/rnabert",
        # "InstaDeepAI/agro-nucleotide-transformer-1b",
        # "genomic_foundation_models/OmniGenome-flashattn-186M",
        # "yangheng/OmniGenome-v1.5",
        # "genomic_foundation_models/SpliceBERT-510nt",
        # "genomic_foundation_models/DNABERT-2-117M",
        # "genomic_foundation_models/3utrbert",
        # "genomic_foundation_models/hyenadna-large-1m-seqlen-hf",
        # "genomic_foundation_models/nucleotide-transformer-v2-100m-multi-species",
    ]
    benchmark = "BEACON"
    batch_size = 2
    patience = 5
    epochs = 20
    max_examples=None
    # max_examples=1000
    # max_seq_length = 440 if "rnabert" in gfms[0] else 512
    seeds = random.randint(0, 1000)
    for gfm in gfms:
        if 'multimolecule' in gfm:
            from multimolecule import RnaTokenizer, AutoModelForTokenPrediction
            tokenizer = RnaTokenizer.from_pretrained(gfm)
            gfm = AutoModelForTokenPrediction.from_pretrained(gfm, trust_remote_code=True).base_model
        else:
            tokenizer = None
        bench = AutoBench(
            benchmark=benchmark,
            model_name_or_path=gfm,
            tokenizer=tokenizer,
            overwrite=True,
            # trainer='native',
            # use_hf_trainer=True
        )
        bench.run(
            # The following parameters are optional and will override the default values
            batch_size=batch_size,
            gradient_accumulation_steps=1,
            patience=patience,
            max_examples=max_examples,
            seeds=seeds,
            epochs=epochs,
            # deepspeed=args.deepspeed
        )

