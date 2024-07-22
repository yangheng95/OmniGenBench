# -*- coding: utf-8 -*-
# file: easy_rna_design.py
# time: 16:46 28/05/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.


import torch
import os
from transformers import OmniGenomeModelForSeq2SeqLM


def predict_contrafold_structure(sequences):
    structures = []
    contra_fold_bin = "../contrafold/src/contrafold"
    if not isinstance(sequences, list):
        sequences = [sequences]

    for i, seq in enumerate(sequences):
        fname = f"temp/temp_{seq[:100]}_{i}.txt"
        with open(fname, "w") as f:
            f.write(f"{seq}\n")
        with os.popen(f"{contra_fold_bin} predict {fname}") as p:
            result = p.read()
            structure = result.split("\n")[-2]
        # print(structure)
        structures.append(structure)
    return structures


if __name__ == "__main__":
    target_structures = []
    sequences = []
    with open("eterna100_vienna2.txt", encoding="utf8", mode="r") as f:
        lines = f.readlines()[1:]
        for line in lines:
            parts = line.split("\t")
            target_structures.append(parts[4].strip())
            sequences.append(parts[5].strip())

    # with open("eterna100_contrafold.txt", encoding="utf8", mode="w") as f:
    #     strctures = predict_contrafold_structure(sequences)
    #     for i, seq in enumerate(sequences):
    #         new_line = '\t'.join(lines[i].split('\t')[:4]) + '\t' + strctures[i] + '\t' + seq + '\n'
    #         f.write(new_line)

    structures = target_structures[:]

    # model = OmniGenomeModelForSeq2SeqLM.from_pretrained("anonymous8/OmniGenome-186M")
    model = OmniGenomeModelForSeq2SeqLM.from_pretrained(
        "benchmark/genomic_foundation_models/OmniGenomeV3-186M"
    )
    model.to("cuda") if torch.cuda.is_available() else model.to("cpu")

    num_all = 0
    num_acc = 0
    for i, structure in enumerate(structures):
        candidate_sequences = model.design(
            structure, mutation_ratio=0.2, num_population=50, num_generation=30
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
