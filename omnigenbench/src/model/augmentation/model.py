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
        batch_size=32,
        use_amp=None,
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
            batch_size (int): Batch size used when running the MLM forward pass (default: 32)
            use_amp (bool|None): Whether to use automatic mixed precision for speed on GPU (default: auto-detect)
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
        self.model.eval()

        # Hyperparameters for augmentation
        self.noise_ratio = noise_ratio
        self.max_length = max_length
        self.k = instance_num
        self.batch_size = int(batch_size) if batch_size is not None else 32
        if use_amp is None:
            self.use_amp = torch.cuda.is_available()
        else:
            self.use_amp = bool(use_amp)

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
        if not seq_list:
            return seq
        mask_cnt = (
            max(1, int(len(seq_list) * self.noise_ratio)) if self.noise_ratio > 0 else 0
        )
        for _ in range(mask_cnt):
            random_idx = random.randint(0, len(seq_list) - 1)
            seq_list[random_idx] = self.tokenizer.mask_token
        return "".join(seq_list)

    def _augment_batch(self, seq_list):
        """
        Augment a batch of masked sequences using a single forward pass.

        Args:
            seq_list (List[str]): List of masked sequences

        Returns:
            List[str]: Augmented sequences
        """
        if not seq_list:
            return []

        tokenized_inputs = self.tokenizer(
            seq_list,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        with torch.no_grad():
            if self.use_amp:
                # autocast speeds up FP16/BF16 on GPU while preserving accuracy for decoding
                autocast_ctx = torch.cuda.amp.autocast(
                    dtype=torch.float16 if torch.cuda.is_available() else None
                )
            else:
                # dummy context manager
                class _Noop:
                    def __enter__(self):
                        return None

                    def __exit__(self, exc_type, exc, tb):
                        return False

                autocast_ctx = _Noop()

            with autocast_ctx:
                outputs = self.model(**tokenized_inputs.to(self.device))
                logits = (
                    outputs["logits"] if isinstance(outputs, dict) else outputs.logits
                )
                predicted_tokens = logits.argmax(dim=-1).detach().cpu()

        input_ids = tokenized_inputs["input_ids"].cpu()
        mask_id = self.tokenizer.mask_token_id
        # Replace [MASK] positions with top-1 predictions
        mask_positions = input_ids == mask_id
        input_ids[mask_positions] = predicted_tokens[mask_positions]

        # Decode back to strings
        augmented_sequences = self.tokenizer.batch_decode(
            input_ids, skip_special_tokens=True
        )
        return augmented_sequences

    def augment_sequence(self, seq):
        """
        Perform augmentation on a single sequence by predicting masked tokens.

        Args:
            seq (str): Input genomic sequence with masked tokens

        Returns:
            str: Augmented sequence with predicted tokens replacing masked tokens
        """
        # Keep single-sample path for API compatibility, implemented via batch for efficiency
        return self._augment_batch([seq])[0]

    def augment(self, seq, k=None):
        """
        Generate multiple augmented instances for a single sequence.

        Args:
            seq (str): Input genomic sequence
            k (int, optional): Number of augmented instances to generate (default: None, uses self.k)

        Returns:
            list: List of augmented sequences
        """
        k = self.k if k is None else int(k)
        # Prepare K noised variants, then process in mini-batches
        noised_variants = [self.apply_noise_to_sequence(seq) for _ in range(k)]
        augmented_sequences = []
        for i in range(0, len(noised_variants), self.batch_size):
            batch = noised_variants[i : i + self.batch_size]
            augmented_sequences.extend(self._augment_batch(batch))
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
        total = len(sequences) * max(1, self.k)
        pbar = tqdm.tqdm(total=total, desc="Augmenting Sequences", unit="inst")

        buffer = []
        # We accumulate masked sequences across inputs until batch_size, then run one forward pass
        for seq in sequences:
            # generate k masked copies for this sequence
            for _ in range(self.k):
                buffer.append(self.apply_noise_to_sequence(seq))
                # flush when buffer reaches batch size
                if len(buffer) >= self.batch_size:
                    all_augmented_sequences.extend(self._augment_batch(buffer))
                    pbar.update(len(buffer))
                    buffer = []
        # flush the tail
        if buffer:
            all_augmented_sequences.extend(self._augment_batch(buffer))
            pbar.update(len(buffer))
        pbar.close()
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
        batch_size=64,
    )
    aug = model.augment_sequence("ATCTTGCATTGAAG")
    input_file = "toy_datasets/test.json"
    output_file = "toy_datasets/augmented_sequences.json"

    model.augment_from_file(input_file, output_file)
