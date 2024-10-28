# -*- coding: utf-8 -*-
# file: extract_evolution_history.ffi.py
# time: 00:19 23/09/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.

import json
import time

from ViennaRNA import ViennaRNA

solved_sequences = json.load(open("solved_sequences.json", "r"))


histories = [v['histories'] for k, v in solved_sequences.items()]

history = histories[5]
print(len(history))
for gen in history:
    best_candidate = gen[0][0]
    strucutre = ViennaRNA.fold(best_candidate)[0]
    ViennaRNA.svg_rna_plot(best_candidate, strucutre, f"predicted_structure.svg")
    time.sleep(1)
print(histories)

