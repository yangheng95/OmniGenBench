# -*- coding: utf-8 -*-
# file: zero_shot_secondary_structure_prediction.py
# time: 16:53 28/05/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.

import torch

from transformers import OmniGenomeForTokenClassification, AutoTokenizer

if __name__ == "__main__":
    sequence = "GAAAAAAAAGGGGAGAAAUCCCGCCCGAAAGGGCGCCCAAAGGGC"

    # anonymous8/OmniGenome-186M is a model trained on the RNA secondary structure prediction task
    # Use it in zero-shot RNA secondary structure prediction
    ssp_model = OmniGenomeForTokenClassification.from_pretrained(
        "anonymous8/OmniGenome-186M"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ssp_model.to(device)

    tokenizer = AutoTokenizer.from_pretrained("anonymous8/OmniGenome-186M")
    inputs = tokenizer(
        sequence, return_tensors="pt", padding="max_length", truncation=True
    ).to(device)
    with torch.no_grad():
        outputs = ssp_model(**inputs)
    predictions = outputs.logits.argmax(dim=-1)[:, 1:-1]
    structure = [
        ssp_model.config.id2label[prediction.item()] for prediction in predictions[0]
    ]
    print("".join(structure))
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
