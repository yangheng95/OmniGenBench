# -*- coding: utf-8 -*-
# file: zero_shot_secondary_structure_prediction.py
# time: 16:53 28/05/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.
import json

import torch

import autocuda

from transformers import OmniGenomeForTokenClassification, AutoTokenizer

from sklearn import metrics

def zero_shot_rfam_batch_evaluation(model, tokenizer, test_data_path):
    model.eval()
    with open(test_data_path, "r") as f:
        lines = f.readlines()
        sequences = []
        structures = []
        for line in lines:
            parts = json.loads(line)
            if len(parts['seq']) <= 512:
                sequences.append(parts['seq'])
                structures.append(parts['label'])

        batch = 8
        all_predicted_structures = []
        all_true_structures = []
        num_all = 0

        for i in range(0, len(sequences), batch):
            batch_sequences = sequences[i:i+batch]
            batch_structures = structures[i:i+batch]
            inputs = tokenizer(
                batch_sequences, return_tensors="pt", padding="max_length", max_length=512, truncation=True
            ).to(model.device)
            with torch.no_grad():
                outputs = model(**inputs)
            predictions = outputs.logits.argmax(dim=-1)[:, 1:-1]
            predicted_structures = [
                model.config.id2label[prediction.item()] for prediction in predictions[0]
            ]
            batch_structures = [list(structure) for structure in batch_structures]
            all_predicted_structures.extend([pred[:len(true)] for pred, true in zip(predicted_structures, batch_structures)])
            all_true_structures.extend(batch_structures)

        f1 = metrics.f1_score(all_true_structures, all_predicted_structures)

    return f1

def zero_shot_secondary_structure_prediction(model, sequence):
    model.eval()
    inputs = tokenizer(
        sequence, return_tensors="pt", padding="max_length", truncation=True
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = outputs.logits.argmax(dim=-1)[:, 1:-1]
    structure = [
        model.config.id2label[prediction.item()] for prediction in predictions[0]
    ]
    return "".join(structure)


if __name__ == "__main__":


    # anonymous8/OmniGenome-186M is a model trained on the RNA secondary structure prediction task
    # Use it in zero-shot RNA secondary structure prediction
    ssp_model = OmniGenomeForTokenClassification.from_pretrained(
        "anonymous8/OmniGenome-186M"
    ).to(autocuda.auto_cuda())
    tokenizer = AutoTokenizer.from_pretrained("anonymous8/OmniGenome-186M")

    # Batch evaluation on the test data
    test_data_path = "benchmark/__OMNIGENOME_DATA__/benchmarks/RGB/RNA-SSP-Rfam/test.json"
    f1 = zero_shot_rfam_batch_evaluation(ssp_model, tokenizer, test_data_path)



    sequence = "GAAAAAAAAGGGGAGAAAUCCCGCCCGAAAGGGCGCCCAAAGGGC"
    # The following code predicts the secondary structure of the input sequence
    structure = zero_shot_secondary_structure_prediction(ssp_model, sequence)

    # The output should be: "..........((((....))))((((....))))((((...))))"

    # The above code can be simplified as follows
    # Compared to the previous code, the output is a list containing the possible predicted secondary structure
    structure = ssp_model.fold(sequence)
    print(structure)
    # The output should be: ["..........((((....))))((((....))))((((...))))"]

    # For comparison, you can also use ViennaRNA

    # import ViennaRNA
    # print(ViennaRNA.fold(sequence)[0])
    # The output should be: "..........((((....))))((((....))))((((...))))"
