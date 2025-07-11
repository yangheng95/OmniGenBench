# -*- coding: utf-8 -*-
# file: model.py
# time: 18:37 22/09/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.
"""
Data augmentation model for genomic sequences.

This module provides a data augmentation model that uses masked language modeling
to generate augmented versions of genomic sequences. It's useful for expanding
training datasets and improving model robustness.
"""
import torch
import random
import json
import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer
import autocuda


class OmniModelForAugmentation(torch.nn.Module):
    """
    Data augmentation model for genomic sequences using masked language modeling.
    
    This model uses a pre-trained masked language model to generate augmented
    versions of genomic sequences by randomly masking tokens and predicting
    replacements. It's useful for expanding training datasets and improving
    model generalization.
    
    Attributes:
        tokenizer: Tokenizer for processing genomic sequences
        model: Pre-trained masked language model
        device: Device to run the model on (CPU or GPU)
        noise_ratio: Proportion of tokens to mask for augmentation
        max_length: Maximum sequence length for tokenization
        k: Number of augmented instances to generate per sequence
    """
    
    def __init__(
        self,
        model_name_or_path=None,
        noise_ratio=0.15,
        max_length=1026,
        instance_num=1,
        *args,
        **kwargs
    ):
        """
        Initialize the augmentation model.
        
        Args:
            model_name_or_path (str): Path or model name for loading the pre-trained model
            noise_ratio (float): The proportion of tokens to mask in each sequence for augmentation (default: 0.15)
            max_length (int): The maximum sequence length for tokenization (default: 1026)
            instance_num (int): Number of augmented instances to generate per sequence (default: 1)
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        super().__init__()
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        except Exception as e:
            if "RnaTokenizer" in str(e):
                from multimolecule import RnaTokenizer

                self.tokenizer = RnaTokenizer.from_pretrained(model_name_or_path)

        self.model = AutoModelForMaskedLM.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        self.device = autocuda.auto_cuda()
        self.model.to(self.device)

        # Hyperparameters for augmentation
        self.noise_ratio = noise_ratio
        self.max_length = max_length
        self.k = instance_num

    def load_sequences_from_file(self, input_file):
        """
        Load sequences from a JSON file.
        
        Args:
            input_file (str): Path to the input JSON file containing sequences
            
        Returns:
            list: List of sequences loaded from the file
        """
        sequences = []
        with open(input_file, "r") as f:
            for line in f.readlines():
                sequences.append(json.loads(line)["seq"])
        return sequences

    def apply_noise_to_sequence(self, seq):
        """
        Apply noise to a single sequence by randomly masking tokens.
        
        Args:
            seq (str): Input genomic sequence
            
        Returns:
            str: Sequence with randomly masked tokens
        """
        seq_list = self.tokenizer.tokenize(seq)
        for _ in range(int(len(seq_list) * self.noise_ratio)):
            random_idx = random.randint(0, len(seq_list) - 1)
            seq_list[random_idx] = self.tokenizer.mask_token
        return "".join(seq_list)

    def augment_sequence(self, seq):
        """
        Perform augmentation on a single sequence by predicting masked tokens.
        
        Args:
            seq (str): Input genomic sequence with masked tokens
            
        Returns:
            str: Augmented sequence with predicted tokens replacing masked tokens
        """
        tokenized_inputs = self.tokenizer(
            seq,
            padding="do_not_pad",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        with torch.no_grad():
            predictions = self.model(**tokenized_inputs.to(self.device))["logits"]
            predicted_tokens = predictions.argmax(dim=-1).cpu()

        # Replace masked tokens with predicted tokens
        input_ids = tokenized_inputs["input_ids"][0].cpu()
        input_ids[input_ids == self.tokenizer.mask_token_id] = predicted_tokens[0][
            input_ids == self.tokenizer.mask_token_id
        ]

        augmented_sequence = self.tokenizer.decode(input_ids, skip_special_tokens=True)
        return augmented_sequence

    def augment(self, seq, k=None):
        """
        Generate multiple augmented instances for a single sequence.
        
        Args:
            seq (str): Input genomic sequence
            k (int, optional): Number of augmented instances to generate (default: None, uses self.k)
            
        Returns:
            list: List of augmented sequences
        """
        augmented_sequences = []
        for _ in range(self.k if k is None else k):
            noised_seq = self.apply_noise_to_sequence(seq)
            augmented_seq = self.augment_sequence(noised_seq)
            augmented_sequences.append(augmented_seq)
        return augmented_sequences

    def augment_sequences(self, sequences):
        """
        Augment a list of sequences by applying noise and performing MLM-based predictions.
        
        Args:
            sequences (list): List of genomic sequences to augment
            
        Returns:
            list: List of all augmented sequences
        """
        all_augmented_sequences = []
        for seq in tqdm.tqdm(sequences, desc="Augmenting Sequences"):
            augmented_instances = self.augment(seq)
            all_augmented_sequences.extend(augmented_instances)
        return all_augmented_sequences

    def save_augmented_sequences(self, augmented_sequences, output_file):
        """
        Save augmented sequences to a JSON file.
        
        Args:
            augmented_sequences (list): List of augmented sequences to save
            output_file (str): Path to the output JSON file
        """
        with open(output_file, "w") as f:
            for seq in augmented_sequences:
                f.write(json.dumps({"aug_seq": seq}) + "\n")

    def augment_from_file(self, input_file, output_file):
        """
        Main function to handle the augmentation process from a file input to a file output.
        
        This method loads sequences from an input file, augments them using the MLM model,
        and saves the augmented sequences to an output file.
        
        Args:
            input_file (str): Path to the input file containing sequences
            output_file (str): Path to the output file where augmented sequences will be saved
        """
        sequences = self.load_sequences_from_file(input_file)
        augmented_sequences = self.augment_sequences(sequences)
        self.save_augmented_sequences(augmented_sequences, output_file)


# Example usage
if __name__ == "__main__":
    model = OmniModelForAugmentation(
        model_name_or_path="anonymous8/OmniGenome-186M",
        noise_ratio=0.2,  # Example noise ratio
        max_length=1026,  # Maximum token length
        instance_num=3,  # Number of augmented instances per sequence
    )
    aug = model.augment_sequence("ATCTTGCATTGAAG")
    input_file = "toy_datasets/test.json"
    output_file = "toy_datasets/augmented_sequences.json"

    model.augment_from_file(input_file, output_file)
