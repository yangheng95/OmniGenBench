# -*- coding: utf-8 -*-
# file: rna_modeling_using_omnigenome.py
# time: 16:58 28/05/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.


import random

import autocuda
import torch
from transformers import AutoTokenizer

from omnigenome import (
    OmniGenomeDatasetForTokenClassification,
)
from omnigenome import ClassificationMetric
from omnigenome import (
    OmniGenomeModelForTokenClassification,
)
from omnigenome import OmniGenomeTokenizer, ModelHub
from omnigenome import Trainer


# Step 1: Load the model
model_name_or_path = "anonymous8/OmniGenome-52M"
# SN_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
SN_tokenizer = OmniGenomeTokenizer.from_pretrained(model_name_or_path)

label2id = {"(": 0, ")": 1, ".": 2}
model = OmniGenomeModelForTokenClassification(
    model_name_or_path, tokenizer=SN_tokenizer, label2id=label2id
)

# Step 2: Set up the training environment
epochs = 1
learning_rate = 2e-5
weight_decay = 1e-5
batch_size = 8
seed = random.randint(0, 1000)


# Step 3: Load the dataset
train_file = "toy_datasets/train.json"
test_file = "toy_datasets/test.json"
valid_file = "toy_datasets/valid.json"

# If you have multiple datasets, you can load them like this:
# train_file = [
#     "toy_datasets1/train.json",
#     "toy_datasets2/train.json",
#     ...
# ]
# valid_file = [
#     "toy_datasets1/valid.json",
#     "toy_datasets2/valid.json",
#     ...
# ]
# test_file = [
#     "toy_datasets1/test.json",
#     "toy_datasets2/test.json",
# ]

train_set = OmniGenomeDatasetForTokenClassification(
    data_source=train_file, tokenizer=SN_tokenizer, label2id=label2id, max_length=512
)
test_set = OmniGenomeDatasetForTokenClassification(
    data_source=test_file, tokenizer=SN_tokenizer, label2id=label2id, max_length=512
)
valid_set = OmniGenomeDatasetForTokenClassification(
    data_source=valid_file, tokenizer=SN_tokenizer, label2id=label2id, max_length=512
)

# Step 4: Create the data loaders
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True
)
valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

# Step 5: Define the optimizer, evaluation metric,  loss function (Optional), and the Trainer
compute_metrics = ClassificationMetric(ignore_y=-100, average="macro").f1_score

optimizer = torch.optim.AdamW(
    model.parameters(), lr=learning_rate, weight_decay=weight_decay
)
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    eval_loader=valid_loader,
    test_loader=test_loader,
    batch_size=batch_size,
    epochs=epochs,
    patience=3,
    optimizer=optimizer,
    compute_metrics=compute_metrics,
    seed=seed,
    device=autocuda.auto_cuda(),
)

# metrics = trainer.train()
# Save the model
model.save("OmniGenome-52M", overwrite=True)


# Load the model: Option 1
model.load("OmniGenome-52M")

# Load the model: Option 2
model = ModelHub.load("OmniGenome-52M")

# inference
output = model.inference(
    [
        "GCCCGAAUAGCUCAGCCGGUUAGAGCACUUGACUGUUAAUCAGGGGGUCGUUGGUUCGAGUCCAACUUCGGGCGCCA",
        "GCCCGAAUAGCUCAGCCGGUUAGAGCACUUGACUGUUAAUCAGGGGGUCGUUGGUUCGAGUCCAACUUCGGGCGCCA",
    ]
)
print(output["predictions"])
