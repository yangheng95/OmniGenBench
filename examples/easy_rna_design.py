# -*- coding: utf-8 -*-
# file: easy_rna_design.py
# time: 16:46 28/05/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.


import torch

from transformers import OmniGenomeModelForSeq2SeqLM

if __name__ == "__main__":
    target_structures = []
    sequences = []
    with open("eterna100_vienna2.txt", encoding="utf8", mode="r") as f:
        for line in f.readlines()[1:]:
            parts = line.split("\t")
            target_structures.append(parts[4].strip())
            sequences.append(parts[5].strip())

    structures = target_structures[:]

    model = OmniGenomeModelForSeq2SeqLM.from_pretrained("yangheng/OmniGenome-186M")
    model.to("cuda") if torch.cuda.is_available() else model.to("cpu")

    num_all = 0
    num_acc = 0
    for i, structure in enumerate(structures):
        candidate_sequences = model.design(
            structure, num_population=50, num_generation=30
        )
        if candidate_sequences:
            num_acc += 1
        print(
            f"Puzzle {i + 1}:",
            candidate_sequences,
            "Accuracy:",
            num_acc / (i + 1) * 100,
            "%",
        )
