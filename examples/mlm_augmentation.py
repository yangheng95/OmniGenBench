# -*- coding: utf-8 -*-
# file: mlm_augmentation.py
# time: 13:05 18/07/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.

import json
import random

import autocuda
import torch
import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

from omnigenome import ClassificationMetric
from omnigenome import (
    OmniGenomeDatasetForTokenClassification,
)

model_name_or_path = "benchmark/genomic_foundation_models/OmniGenomeV3-186M"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForMaskedLM.from_pretrained(model_name_or_path, trust_remote_code=True)
device = autocuda.auto_cuda()
model.to(device)

input_file = 'toy_datasets/test.json'

sequences = []
with open(input_file, 'r') as f:
    for line in f.readlines():
        sequences.append(json.loads(line))

sequences = [sequences[i]['seq'] for i in range(len(sequences))]

# add_noise
for i, seq in enumerate(sequences):
    sequences[i] = list(seq)
    for _ in range(int(len(seq) * 0.15)):
        sequences[i][random.randint(0, len(seq) - 1)] = tokenizer.mask_token
    sequences[i] = ''.join(sequences[i])


for seq in tqdm.tqdm(sequences):
    tokenized_inputs = tokenizer(
        seq,
        padding="do_not_pad",
        truncation=True,
        max_length=1026,
        return_tensors="pt",
    )

    prediction = model(**tokenized_inputs.to(device))["logits"]
    prediction = prediction.argmax(dim=-1).view(-1).to("cpu")

    tokenized_inputs = tokenized_inputs.to("cpu")
    tokenized_inputs['input_ids'][0][tokenized_inputs['input_ids'][0] == tokenizer.mask_token_id] = prediction[tokenized_inputs['input_ids'][0] == tokenizer.mask_token_id]
    aug_seq = tokenizer.decode(tokenized_inputs['input_ids'][0], skip_special_tokens=True)

    print(f"Original sequence: {seq}")
    print(f"Augmented sequence: {aug_seq}")
    print()