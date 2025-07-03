# -*- coding: utf-8 -*-
# file: model.py
# time: 18:37 22/09/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.

import torch
from transformers import AutoTokenizer, AutoModel

from omnigenome.src.misc.utils import fprint


class OmniModelForEmbedding(torch.nn.Module):
    def __init__(self, model_name_or_path, *args, **kwargs):
        """Initializes the embedding model."""
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path, *args, **kwargs)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self._device)
        self.model.eval()  # Set model to evaluation mode

    def batch_encode(self, sequences, batch_size=8, max_length=512, agg='head'):
        """
        Encode a list of sequences to their corresponding embeddings.

        Args:
            sequences (list of str): List of input sequences to encode.
            batch_size (int): Batch size for processing.
            max_length (int): Maximum sequence length for encoding.
            agg (str): Aggregation method for embeddings. Options are 'head', 'mean', 'tail'.

        Returns:
            torch.Tensor: Embeddings for the input sequences.
        """
        embeddings = []

        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i: i + batch_size]
            inputs = self.tokenizer(
                batch_sequences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            batch_embeddings = outputs.last_hidden_state.cpu()

            if agg == 'head':
                emb = batch_embeddings[:, 0, :]
            elif agg == 'mean':
                attention_mask = inputs["attention_mask"].cpu()
                masked_embeddings = batch_embeddings * attention_mask.unsqueeze(-1)
                lengths = attention_mask.sum(dim=1).unsqueeze(1)
                emb = masked_embeddings.sum(dim=1) / lengths
            elif agg == 'tail':
                attention_mask = inputs["attention_mask"]
                lengths = attention_mask.sum(dim=1) - 1
                emb = torch.stack([
                    batch_embeddings[i, l.item()] for i, l in enumerate(lengths)
                ])
            else:
                raise ValueError(f"Unsupported aggregation method: {agg}")

            embeddings.append(emb)

        embeddings = torch.cat(embeddings, dim=0)
        fprint(f"Generated embeddings for {len(sequences)} sequences.")
        return embeddings

    def encode(self, sequence, max_length=512, agg='head', keep_dim=False):
        """
        Encode a single sequence to its corresponding embedding.

        Args:
            sequence (str): Input sequence to encode.
            max_length (int): Maximum sequence length for encoding.
            agg (str): Aggregation method.
            keep_dim (bool): Whether to retain the batch dimension.

        Returns:
            torch.Tensor: Embedding for the input sequence.
        """
        inputs = self.tokenizer(
            sequence,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        last_hidden = outputs.last_hidden_state.cpu()

        if agg == 'head':
            emb = last_hidden[0, 0]
        elif agg == 'mean':
            attention_mask = inputs["attention_mask"].cpu()
            masked_embeddings = last_hidden * attention_mask.unsqueeze(-1)
            lengths = attention_mask.sum(dim=1).unsqueeze(1)
            emb = masked_embeddings.sum(dim=1) / lengths
            emb = emb.squeeze(0)
        elif agg == 'tail':
            attention_mask = inputs["attention_mask"]
            lengths = attention_mask.sum(dim=1) - 1
            emb = last_hidden[0, lengths[0].item()]
        else:
            raise ValueError(f"Unsupported aggregation method: {agg}")

        return emb.unsqueeze(0) if keep_dim else emb

    def save_embeddings(self, embeddings, output_path):
        """
        Save the generated embeddings to a file.

        Args:
            embeddings (torch.Tensor): The embeddings to save.
            output_path (str): Path to save the embeddings.
        """
        torch.save(embeddings, output_path)
        fprint(f"Embeddings saved to {output_path}")

    def load_embeddings(self, embedding_path):
        """
        Load embeddings from a file.

        Args:
            embedding_path (str): Path to the saved embeddings.

        Returns:
            torch.Tensor: The loaded embeddings.
        """
        embeddings = torch.load(embedding_path)
        fprint(f"Loaded embeddings from {embedding_path}")
        return embeddings

    def compute_similarity(self, embedding1, embedding2, dim=0):
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1 (torch.Tensor): The first embedding.
            embedding2 (torch.Tensor): The second embedding.
            dim (int): Dimension along which to compute cosine similarity.

        Returns:
            float: Cosine similarity score.
        """
        similarity = torch.nn.functional.cosine_similarity(
            embedding1, embedding2, dim=dim
        )
        return similarity

    @property
    def device(self):
        """Get the current device ('cuda' or 'cpu')."""
        return self._device


# Example usage
if __name__ == "__main__":
    model_name = "anonymous8/OmniGenome-186M"
    embedding_model = OmniModelForEmbedding(model_name)

    # Encode multiple sequences
    sequences = ["ATCGGCTA", "GGCTAGCTA"]
    embedding = embedding_model.encode(sequences[0])
    fprint(f"Single embedding shape: {embedding.shape}")

    embeddings = embedding_model.batch_encode(sequences)
    fprint(f"Embeddings for sequences: {embeddings}")

    # Save and load embeddings
    embedding_model.save_embeddings(embeddings, "embeddings.pt")
    loaded_embeddings = embedding_model.load_embeddings("embeddings.pt")

    # Compute similarity between two embeddings
    similarity = embedding_model.compute_similarity(
        loaded_embeddings[0], loaded_embeddings[1]
    )
    fprint(f"Cosine similarity: {similarity}")
