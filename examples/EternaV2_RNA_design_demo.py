# -*- coding: utf-8 -*-
# file: OmniGenomeRNADesign.py
# time: 20:24 08/05/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.
import os
import multiprocessing
import random

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    benchmark_file = "eterna100_vienna2.txt"
    # benchmark_file = "eterna100_contrafold.txt"
    structures = []
    sequences = []
    solved_sequences = {}

    with open(benchmark_file, encoding="utf8", mode="r") as f:
        lines = f.readlines()[1:]
        # lines = sorted(lines, key=lambda x: len(x.split("\t")[4]))
        random.shuffle(lines)
        for line in lines:
            parts = line.split("\t")
            # if len(parts[5].strip()) > 200:
            #     continue
            structures.append(parts[4].strip())
            sequences.append(parts[5].strip())

    outputs = []
    pred_count = 0
    acc_count = 0

    pool = multiprocessing.Pool(20)
    for i, (seq, structure) in enumerate(zip(sequences, structures)):
        cmd = (f'python easy_rna_design.py '
               f'--structure "{structure}" '
               f'--sequence "{seq}" '
               f'--num_population 1000 '
               f'--num_generation 1000 '
               f'--mutation_ratio 0.5 '
               )
        import time

        time.sleep(3)
        pool.apply_async(os.system, args=(cmd,))

    pool.close()
    pool.join()



