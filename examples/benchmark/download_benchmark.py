# -*- coding: utf-8 -*-
# file: easy_rna_design.py
# time: 16:46 28/05/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.

from omnigenome import download_benchmark

if __name__ == "__main__":
    # benchmarks = ["RGB", "GB", "PGB", "GUE"]
    bench = download_benchmark("RGB")
    print(bench)
