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
if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--gfm", type=str, default="genomic_foundation_models/OmniGenome-52M-1000")
    parser.add_argument("--bench_root", type=str, default="RGB")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seeds", type=int, default=random.randint(0, 1000))
    args = parser.parse_args()

    gfms = [
        # "genomic_foundation_models/MoEOmniGenome",
        "genomic_foundation_models/MoEOmniGenomeV2",
        # "genomic_foundation_models/OmniGenomeV3-186M",
        # "genomic_foundation_models/checkpoint-str_ids",
    ]
    bench_root = "RGB"
    batch_size = 4
    patience = 5
    epochs = 20
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
            autocast="fp16",
            bench_root=bench_root,
            model_name_or_path=gfm,
            tokenizer=tokenizer,
            overwrite=True,
            # use_hf_trainer=True
        )
        bench.run(
            batch_size=batch_size,
            gradient_accumulation_steps=1,
            patience=patience,
            seeds=seeds,
            epochs=epochs,
        )
