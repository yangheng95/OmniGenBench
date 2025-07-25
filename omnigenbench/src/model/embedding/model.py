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

from ....src.misc.utils import fprint


class OmniModelForEmbedding(torch.nn.Module):
    """
    This class provides a unified interface for loading pre-trained models and
    generating embeddings from genomic sequences. It supports various aggregation
    methods and batch processing for efficient embedding generation.

    Attributes:
        tokenizer: The tokenizer for processing input sequences
        model: The pre-trained model for generating embeddings
        _device: The device (CPU/GPU) where the model is loaded

    Example:
        >>> from omnigenbench import OmniModelForEmbedding
        >>> model = OmniModelForEmbedding("anonymous8/OmniGenome-186M")
        >>> sequences = ["ATCGGCTA", "GGCTAGCTA"]
        >>> embeddings = model.batch_encode(sequences)
        >>> print(f"Embeddings shape: {embeddings.shape}")
        torch.Size([2, 768])
    """

    def __init__(self, model_name_or_path, *args, **kwargs):
        """
        Initialize the embedding model.

        Args:
            model_name_or_path (str): Name or path of the pre-trained model to load
            *args: Additional positional arguments passed to AutoModel.from_pretrained
            **kwargs: Additional keyword arguments passed to AutoModel.from_pretrained
        """
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path, *args, **kwargs)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self._device)
        self.model.eval()  # Set model to evaluation mode

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
        """批量编码序列为 pooled 向量。

        Batch encode sequences into aggregated (pooled) embeddings.

        参数 / Args:
            sequences (List[str]): 输入序列 / input DNA (or RNA) sequences.
            batch_size (int): 批大小 / processing batch size.
            max_length (int): tokenizer 截断/填充长度 / truncate/pad length.
            agg (str): 聚合方式 head|mean|tail / aggregation method.
            require_grad (bool): 是否需要梯度; True 时允许反向传播 / keep graph for finetuning.
            return_on_cpu (bool): 若 True 输出放到 CPU, 否则保持在模型设备 / move result to CPU for memory relief.
            use_autocast (bool): 使用混合精度 / enable autocast (CUDA only).
            amp_dtype (torch.dtype|None): autocast 精度类型 / dtype for autocast.

        返回 / Returns:
            torch.Tensor 形状 (N, H) / shape (num_sequences, hidden_size)

        兼容性 / Compatibility:
            旧调用无需修改; 新参数有默认值。
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
                    pooled_list.append(hidden[bi, int(l.item()), :])
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

        Note:
            When require_grad=True, gradients flow through the transformer model for end-to-end training.
            Set return_on_cpu=False to keep tensors on GPU device for downstream processing.
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
            >>> model = OmniModelForEmbedding("yangheng/OmniGenome-52M")
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
        """编码单个序列 / Encode a single sequence.

        参数 / Args:
            sequence (str): 输入序列 / input sequence.
            max_length (int): 截断/填充长度 / tokenizer max length.
            agg (str): head|mean|tail 聚合策略 / aggregation strategy.
            keep_dim (bool): 是否保留 batch 维 / keep batch dimension.
            require_grad (bool): 是否保留梯度 / keep graph for finetune.
            return_on_cpu (bool): 输出是否转 CPU / move result to CPU.
            use_autocast (bool): 是否使用 autocast / enable autocast.
            amp_dtype (torch.dtype|None): autocast dtype.

        返回 / Returns:
            torch.Tensor shape (H,) 或 (1,H) / pooled embedding.
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
            >>> print("Embeddings saved successfully")
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
            torch.Size([100, 768])
        """
        embeddings = torch.load(embedding_path)
        fprint(f"Loaded embeddings from {embedding_path}")
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
            0.8234
        """
        similarity = torch.nn.functional.cosine_similarity(
            embedding1, embedding2, dim=dim
        )
        return similarity

    @property
    def device(self):
        """
        Get the current device for the underlying model.

        Returns:
            torch.device: The device where the model currently resides.

        Note:
            This queries the model parameters directly so it stays correct
            when external frameworks (e.g., Accelerate/DDP) move the module
            across devices after initialization.
        """
        try:
            return next(self.model.parameters()).device
        except StopIteration:
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
