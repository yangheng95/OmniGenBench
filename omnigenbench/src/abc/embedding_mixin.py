# -*- coding: utf-8 -*-
# file: embedding_mixin.py
# time: 18:37 30/10/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2025. All Rights Reserved.

import torch
from ..misc.utils import fprint


class EmbeddingMixin:
    """
    Mixin class that adds embedding and attention extraction capabilities to any model.

    This mixin provides a unified interface for:
    - Generating sequence embeddings (pooled and token-level)
    - Extracting attention scores from transformer layers
    - Computing similarity between embeddings
    - Visualizing attention patterns

    Usage:
        Simply inherit from this mixin along with your base model class:

        >>> class OmniModel(EmbeddingMixin, torch.nn.Module):
        ...     pass

    All methods require the model to have:
    - self.model: The underlying transformer model
    - self.tokenizer: The tokenizer for processing sequences
    - self.device: The device where the model is located
    """

    def batch_encode(
        self,
        sequences,
        batch_size=8,
        max_length=512,
        agg="head",
        require_grad: bool = False,
        return_on_cpu: bool = True,
        use_autocast: bool = False,
        amp_dtype=None,
    ):
        """Batch encode sequences into aggregated (pooled) embeddings.

        Args:
            sequences (List[str]): Input DNA or RNA sequences for encoding.
            batch_size (int, default=8): Number of sequences to process per batch.
            max_length (int, default=512): Maximum sequence length for tokenization.
            agg (str, default="head"): Aggregation method for pooling. Options: "head", "mean", "tail".
            require_grad (bool, default=False): Whether to preserve gradients for fine-tuning.
            return_on_cpu (bool, default=True): Whether to move results to CPU memory.
            use_autocast (bool, default=False): Whether to enable mixed precision (CUDA only).
            amp_dtype (torch.dtype, optional): Data type for automatic mixed precision.

        Returns:
            torch.Tensor: Pooled embeddings with shape (num_sequences, hidden_size).

        Example:
            >>> sequences = ["ATCGGCTA", "GGCTAGCTA"]
            >>> embeddings = model.batch_encode(sequences, batch_size=4, agg="mean")
            >>> print(embeddings.shape)
            torch.Size([2, 768])
        """
        embeds = []
        device = self.device
        is_cuda = isinstance(device, torch.device) and device.type == "cuda"

        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i : i + batch_size]
            inputs = self.tokenizer(
                batch_sequences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            ctx = (
                (
                    torch.autocast(device_type="cuda", dtype=amp_dtype)
                    if (use_autocast and is_cuda)
                    else torch.enable_grad()
                )
                if require_grad
                else torch.no_grad()
            )

            with ctx:
                outputs = self.model(**inputs).last_hidden_state  # (B,L,H)

            hidden = outputs if not return_on_cpu else outputs.cpu()

            if agg == "head":
                pooled = hidden[:, 0, :]
            elif agg == "mean":
                mask = (
                    inputs["attention_mask"]
                    if not return_on_cpu
                    else inputs["attention_mask"].cpu()
                )
                pooled = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(
                    1, keepdim=True
                )
            elif agg == "tail":
                mask = inputs["attention_mask"]
                lengths = mask.sum(1) - 1
                pooled_list = []
                for bi, l in enumerate(lengths):
                    pooled_list.append(hidden[bi, l, :])
                pooled = torch.stack(pooled_list, 0)
            else:
                raise ValueError(f"Unsupported agg: {agg}")

            embeds.append(pooled)

        out = torch.cat(embeds, 0)
        return out

    def batch_encode_tokens(
        self,
        sequences,
        batch_size=8,
        max_length=512,
        use_autocast=False,
        amp_dtype=None,
        require_grad: bool = False,
        return_on_cpu: bool = True,
    ):
        """
        Encode sequences to token-level embeddings (last_hidden_state).

        Args:
            sequences (List[str]): Input DNA/RNA sequences for token-level encoding
            batch_size (int, default=8): Number of sequences to process per batch
            max_length (int, default=512): Maximum sequence length for tokenization
            use_autocast (bool, default=False): Enable mixed precision training (CUDA only)
            amp_dtype (torch.dtype, optional): Data type for automatic mixed precision
            require_grad (bool, default=False): Preserve gradient computation graph for fine-tuning
            return_on_cpu (bool, default=True): Transfer outputs to CPU memory

        Returns:
            torch.Tensor: Token embeddings with shape (num_sequences, max_length, hidden_size)

        Example:
            >>> sequences = ["ATCGGCTA", "GGCTAGCTA"]
            >>> token_embeddings = model.batch_encode_tokens(sequences)
            >>> print(token_embeddings.shape)
            torch.Size([2, 512, 768])
        """
        outputs = []
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i : i + batch_size]
            inputs = self.tokenizer(
                batch_sequences,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_length,
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}

            ctx = (
                (
                    torch.autocast(device_type="cuda", dtype=amp_dtype)
                    if (
                        use_autocast
                        and isinstance(self.device, torch.device)
                        and self.device.type == "cuda"
                    )
                    else torch.enable_grad()
                )
                if require_grad
                else torch.no_grad()
            )

            with ctx:
                last_hidden = self.model(**inputs).last_hidden_state  # (B, L, H)

            if return_on_cpu:
                last_hidden = last_hidden.cpu()
            outputs.append(last_hidden)

        out = torch.cat(outputs, dim=0)
        return out

    def encode_tokens(
        self,
        sequence,
        max_length=512,
        use_autocast=False,
        amp_dtype=None,
        require_grad: bool = False,
        return_on_cpu: bool = True,
    ):
        """
        Encode a single sequence to token-level embeddings.

        Args:
            sequence (str): Input DNA/RNA sequence for token-level encoding
            max_length (int, default=512): Maximum sequence length for tokenization
            use_autocast (bool, default=False): Enable mixed precision training (CUDA only)
            amp_dtype (torch.dtype, optional): Data type for automatic mixed precision
            require_grad (bool, default=False): Preserve gradient computation graph for fine-tuning
            return_on_cpu (bool, default=True): Transfer output to CPU memory

        Returns:
            torch.Tensor: Token embeddings with shape (max_length, hidden_size)

        Example:
            >>> sequence = "ATCGATCGATCG"
            >>> token_embeddings = model.encode_tokens(sequence, max_length=200)
            >>> print(f"Token embeddings shape: {token_embeddings.shape}")
            torch.Size([200, 768])
        """
        device = self.device
        inputs = self.tokenizer(
            sequence,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        is_cuda = isinstance(device, torch.device) and device.type == "cuda"
        ctx = (
            (
                torch.autocast(device_type="cuda", dtype=amp_dtype)
                if (use_autocast and is_cuda)
                else torch.enable_grad()
            )
            if require_grad
            else torch.no_grad()
        )

        with ctx:
            hidden = self.model(**inputs).last_hidden_state  # (1, L, H)

        if return_on_cpu:
            hidden = hidden.cpu()

        # Remove batch dimension for single sequence
        return hidden.squeeze(0)  # (L, H)

    def encode(
        self,
        sequence,
        max_length=512,
        agg="head",
        keep_dim=False,
        require_grad: bool = False,
        return_on_cpu: bool = True,
        use_autocast: bool = False,
        amp_dtype=None,
    ):
        """Encode a single sequence into pooled embeddings.

        Args:
            sequence (str): Input DNA or RNA sequence for encoding.
            max_length (int, default=512): Maximum sequence length for tokenization.
            agg (str, default="head"): Aggregation strategy for pooling. Options: "head", "mean", "tail".
            keep_dim (bool, default=False): Whether to preserve batch dimension in output.
            require_grad (bool, default=False): Whether to preserve gradients for fine-tuning.
            return_on_cpu (bool, default=True): Whether to move results to CPU memory.
            use_autocast (bool, default=False): Whether to enable mixed precision.
            amp_dtype (torch.dtype, optional): Data type for automatic mixed precision.

        Returns:
            torch.Tensor: Pooled embedding with shape (hidden_size,) or (1, hidden_size) if keep_dim=True.

        Example:
            >>> sequence = "ATCGATCGATCG"
            >>> embedding = model.encode(sequence, agg="mean", max_length=200)
            >>> print(embedding.shape)
            torch.Size([768])
        """
        device = self.device
        inputs = self.tokenizer(
            sequence,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        is_cuda = isinstance(device, torch.device) and device.type == "cuda"
        ctx = (
            (
                torch.autocast(device_type="cuda", dtype=amp_dtype)
                if (use_autocast and is_cuda)
                else torch.enable_grad()
            )
            if require_grad
            else torch.no_grad()
        )
        with ctx:
            hidden = self.model(**inputs).last_hidden_state  # (1,L,H)
        hidden = hidden if not return_on_cpu else hidden.cpu()
        if agg == "head":
            emb = hidden[:, 0, :]
        elif agg == "mean":
            mask = (
                inputs["attention_mask"]
                if not return_on_cpu
                else inputs["attention_mask"].cpu()
            )
            emb = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        elif agg == "tail":
            mask = inputs["attention_mask"]
            l = int(mask.sum(1).item()) - 1
            emb = hidden[:, l, :]
        else:
            raise ValueError(f"Unsupported agg: {agg}")
        if not keep_dim:
            emb = emb.squeeze(0)
        return emb

    def save_embeddings(self, embeddings, output_path):
        """
        Save the generated embeddings to a file.

        Args:
            embeddings (torch.Tensor): The embeddings to save
            output_path (str): Path to save the embeddings

        Example:
            >>> embeddings = model.batch_encode(sequences)
            >>> model.save_embeddings(embeddings, "embeddings.pt")
        """
        torch.save(embeddings, output_path)
        fprint(f"Embeddings saved to {output_path}")

    def load_embeddings(self, embedding_path):
        """
        Load embeddings from a file.

        Args:
            embedding_path (str): Path to the saved embeddings

        Returns:
            torch.Tensor: The loaded embeddings

        Example:
            >>> embeddings = model.load_embeddings("embeddings.pt")
            >>> print(f"Loaded embeddings shape: {embeddings.shape}")
        """
        embeddings = torch.load(embedding_path, map_location=self.device)
        fprint(f"Embeddings loaded from {embedding_path}")
        return embeddings

    def compute_similarity(self, embedding1, embedding2, dim=0):
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1 (torch.Tensor): The first embedding
            embedding2 (torch.Tensor): The second embedding
            dim (int, optional): Dimension along which to compute cosine similarity. Defaults to 0

        Returns:
            float: Cosine similarity score between -1 and 1

        Example:
            >>> emb1 = model.encode("ATCGGCTA")
            >>> emb2 = model.encode("GGCTAGCTA")
            >>> similarity = model.compute_similarity(emb1, emb2)
            >>> print(f"Cosine similarity: {similarity:.4f}")
        """
        similarity = torch.nn.functional.cosine_similarity(
            embedding1, embedding2, dim=dim
        )
        return similarity

    def extract_attention_scores(
        self,
        sequence,
        max_length=512,
        layer_indices=None,
        head_indices=None,
        return_on_cpu=True,
        use_autocast=False,
        amp_dtype=None,
    ):
        """Extract attention scores from a single genomic sequence.

        This method extracts attention weights from transformer layers, providing insights
        into which positions the model focuses on during sequence processing.

        Args:
            sequence (str): Input DNA or RNA sequence for attention extraction.
            max_length (int, default=512): Maximum sequence length for tokenization.
            layer_indices (List[int], optional): Specific transformer layer indices to extract.
                If None, extracts attention from all layers.
            head_indices (List[int], optional): Specific attention head indices to extract.
                If None, extracts attention from all heads.
            return_on_cpu (bool, default=True): Whether to transfer output to CPU memory.
            use_autocast (bool, default=False): Whether to enable mixed precision.
            amp_dtype (torch.dtype, optional): Data type for automatic mixed precision.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - 'attentions': Attention weights tensor with shape (num_layers, num_heads, seq_len, seq_len)
                - 'tokens': List of tokenized input tokens
                - 'attention_mask': Attention mask tensor indicating valid positions

        Example:
            >>> sequence = "ATCGATCGATCG"
            >>> result = model.extract_attention_scores(sequence, max_length=200)
            >>> print(f"Attention shape: {result['attentions'].shape}")
        """
        device = self.device
        inputs = self.tokenizer(
            sequence,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        is_cuda = isinstance(device, torch.device) and device.type == "cuda"
        ctx = (
            torch.autocast(device_type="cuda", dtype=amp_dtype)
            if (use_autocast and is_cuda)
            else torch.no_grad()
        )

        with ctx:
            outputs = self.model(**inputs, output_attentions=True)
            attentions = outputs.attentions

        # Convert tuple to tensor and stack all layers
        attentions_tensor = torch.stack(attentions, dim=1)
        attentions_tensor = attentions_tensor.squeeze(0)

        # Filter specific layers if requested
        if layer_indices is not None:
            attentions_tensor = attentions_tensor[layer_indices]

        # Filter specific heads if requested
        if head_indices is not None:
            attentions_tensor = attentions_tensor[:, head_indices]

        if return_on_cpu:
            attentions_tensor = attentions_tensor.cpu()

        # Get tokens for interpretation
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze(0))

        result = {
            "attentions": attentions_tensor,
            "tokens": tokens,
            "attention_mask": (
                inputs["attention_mask"].cpu()
                if return_on_cpu
                else inputs["attention_mask"]
            ),
        }

        return result

    def batch_extract_attention_scores(
        self,
        sequences,
        batch_size=4,
        max_length=512,
        layer_indices=None,
        head_indices=None,
        return_on_cpu=True,
        use_autocast=False,
        amp_dtype=None,
    ):
        """Extract attention scores from multiple genomic sequences in batches.

        Args:
            sequences (List[str]): List of input DNA or RNA sequences for attention extraction.
            batch_size (int, default=4): Number of sequences to process per batch.
            max_length (int, default=512): Maximum sequence length for tokenization.
            layer_indices (List[int], optional): Specific transformer layer indices to extract.
            head_indices (List[int], optional): Specific attention head indices to extract.
            return_on_cpu (bool, default=True): Whether to transfer outputs to CPU memory.
            use_autocast (bool, default=False): Whether to enable mixed precision.
            amp_dtype (torch.dtype, optional): Data type for automatic mixed precision.

        Returns:
            List[Dict[str, torch.Tensor]]: List of dictionaries, each containing attention data.

        Example:
            >>> sequences = ["ATCGATCGATCG", "GGCCTTAACCGG"]
            >>> results = model.batch_extract_attention_scores(sequences, batch_size=2)
            >>> print(f"Number of results: {len(results)}")
        """
        all_results = []
        device = self.device
        is_cuda = isinstance(device, torch.device) and device.type == "cuda"

        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i : i + batch_size]
            inputs = self.tokenizer(
                batch_sequences,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_length,
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}

            ctx = (
                torch.autocast(device_type="cuda", dtype=amp_dtype)
                if (use_autocast and is_cuda)
                else torch.no_grad()
            )

            with ctx:
                outputs = self.model(**inputs, output_attentions=True)
                attentions = outputs.attentions

            # Convert tuple to tensor and stack all layers
            attentions_tensor = torch.stack(attentions, dim=2)
            attentions_tensor = attentions_tensor.permute(0, 2, 1, 3, 4)

            # Process each sequence in the batch
            for batch_idx in range(attentions_tensor.size(0)):
                seq_attention = attentions_tensor[batch_idx]

                if layer_indices is not None:
                    seq_attention = seq_attention[layer_indices]

                if head_indices is not None:
                    seq_attention = seq_attention[:, head_indices]

                if return_on_cpu:
                    seq_attention = seq_attention.cpu()

                tokens = self.tokenizer.convert_ids_to_tokens(
                    inputs["input_ids"][batch_idx]
                )

                result = {
                    "attentions": seq_attention,
                    "tokens": tokens,
                    "attention_mask": (
                        inputs["attention_mask"][batch_idx].cpu()
                        if return_on_cpu
                        else inputs["attention_mask"][batch_idx]
                    ),
                }
                all_results.append(result)

        return all_results

    def get_attention_statistics(
        self,
        attention_scores,
        attention_mask=None,
        layer_aggregation="mean",
        head_aggregation="mean",
    ):
        """Compute comprehensive statistics from attention scores.

        Args:
            attention_scores (torch.Tensor): Attention tensor with shape (num_layers, num_heads, seq_len, seq_len).
            attention_mask (torch.Tensor, optional): Attention mask to exclude padding tokens.
            layer_aggregation (str, default="mean"): Method to aggregate across layers.
            head_aggregation (str, default="mean"): Method to aggregate across heads.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing attention statistics.

        Example:
            >>> result = model.extract_attention_scores(sequence)
            >>> stats = model.get_attention_statistics(result['attentions'])
        """
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(0).unsqueeze(0)
            mask = mask * attention_mask.unsqueeze(0).unsqueeze(-1)
            attention_scores = attention_scores * mask

        # Aggregate across heads
        if head_aggregation == "mean":
            head_aggregated = attention_scores.mean(dim=1)
        elif head_aggregation == "max":
            head_aggregated = attention_scores.max(dim=1)[0]
        elif head_aggregation == "sum":
            head_aggregated = attention_scores.sum(dim=1)
        else:
            raise ValueError(f"Unsupported head_aggregation: {head_aggregation}")

        # Aggregate across layers
        if layer_aggregation == "mean":
            layer_aggregated = head_aggregated.mean(dim=0)
        elif layer_aggregation == "max":
            layer_aggregated = head_aggregated.max(dim=0)[0]
        elif layer_aggregation == "sum":
            layer_aggregated = head_aggregated.sum(dim=0)
        elif layer_aggregation == "first":
            layer_aggregated = head_aggregated[0]
        elif layer_aggregation == "last":
            layer_aggregated = head_aggregated[-1]
        else:
            raise ValueError(f"Unsupported layer_aggregation: {layer_aggregation}")

        statistics = {
            "attention_matrix": layer_aggregated,
            "attention_entropy": -torch.sum(
                layer_aggregated * torch.log(layer_aggregated + 1e-9), dim=-1
            ),
            "max_attention_per_position": layer_aggregated.max(dim=-1)[0],
            "attention_concentration": (layer_aggregated**2).sum(dim=-1),
            "self_attention_scores": torch.diag(layer_aggregated),
        }

        return statistics

    def visualize_attention_pattern(
        self,
        attention_result,
        layer_idx=0,
        head_idx=0,
        save_path=None,
        figsize=(12, 10),
    ):
        """Visualize attention patterns as an interactive heatmap.

        Args:
            attention_result (Dict): Result dictionary from extract_attention_scores().
            layer_idx (int, default=0): Index of the transformer layer to visualize.
            head_idx (int, default=0): Index of the attention head to visualize.
            save_path (str, optional): File path to save the visualization.
            figsize (tuple, default=(12, 10)): Figure size as (width, height) in inches.

        Returns:
            matplotlib.figure.Figure: The generated figure, or None if matplotlib unavailable.

        Example:
            >>> result = model.extract_attention_scores(sequence)
            >>> fig = model.visualize_attention_pattern(result, save_path="attention.png")
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            fprint(
                "matplotlib is required for visualization. Install with: pip install matplotlib"
            )
            return None

        attention_matrix = attention_result["attentions"][layer_idx, head_idx].numpy()
        tokens = attention_result["tokens"]
        attention_mask = attention_result["attention_mask"].numpy()

        # Find the actual sequence length (excluding padding)
        seq_len = int(attention_mask.sum())

        # Truncate to actual sequence length
        attention_matrix = attention_matrix[:seq_len, :seq_len]
        tokens = tokens[:seq_len]

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(attention_matrix, cmap="Blues", aspect="auto")

        # Set ticks and labels
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha="right")
        ax.set_yticklabels(tokens)

        # Add colorbar
        plt.colorbar(im, ax=ax, label="Attention Weight")

        # Set title and labels
        ax.set_title(f"Attention Pattern - Layer {layer_idx}, Head {head_idx}")
        ax.set_xlabel("Key Positions")
        ax.set_ylabel("Query Positions")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            fprint(f"Attention visualization saved to {save_path}")

        return fig
