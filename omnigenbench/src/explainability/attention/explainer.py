# -*- coding: utf-8 -*-
# file: __init__.py
# time: 2025-08-27 14:06
# author: Shasha Zhou <sz484@exeter.ac.uk>
# Copyright (C) 2020-2025. All Rights Reserved.

from ...abc.abstract_explainer import AbstractExplainer
import matplotlib.pyplot as plt
import torch
import numpy as np
from typing import List, Optional, Union
import math


class AttentionExplainer(AbstractExplainer):
    def __init__(self, model):
        super().__init__(model)

    def explain(self, sequence: str, **kwargs):
        # get attention maps
        inputs = self.model.tokenizer(
            sequence,
            padding=True,
            max_length=1024,
            truncation=True,
            return_tensors="pt",
        )

        inputs = inputs.to(self.model.model.device)

        with torch.no_grad():
            out = self.model.model(**inputs, output_attentions=True)

        return out["attentions"][-1][0].cpu().numpy()

    def visualize(
        self,
        attn_maps: np.ndarray,
        layers: Optional[Union[List[int], int]] = None,
        avg_attn: bool = True,
        **kwargs,
    ):
        """
        Visualize the attention maps.
        Args:
            attn_maps: (layers, num_heads, seq_len, seq_len)
            layers: the layers to visualize the attention maps. If None, the last layer will be visualized. If a single integer, the layer will be visualized. If a list of integers, the layers will be visualized.
            avg_attn: whether to average the attention maps over layers.
        """
        num_layers, num_heads, seq_len, _ = attn_maps.shape
        if layers is None:
            layers = [num_layers - 1]
        elif isinstance(layers, int):
            layers = [layers]
        elif isinstance(layers, list):
            layers = layers

        if not avg_attn and len(layers) > 1:
            print(
                "Warning: avg_attn is False and len(layers) > 1, the attention maps will be averaged over layers."
            )
            avg_attn = True

        if avg_attn:
            attn_head_mean = attn_maps.mean(dim=1)

            n_layers = len(layers)
            n_cols = min(5, n_layers)
            n_rows = math.ceil(n_layers / n_cols)

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
            axes = np.array(axes).reshape(-1)

            for i, layer in enumerate(layers):
                ax = axes[i]
                attn_final = attn_head_mean[layer]
                ax.imshow(attn_final, cmap="Reds")
                ax.set_title(f"Layer {layer}")
                ax.axis("off")

            for j in range(len(layers), len(axes)):
                axes[j].axis("off")

            plt.tight_layout()
            plt.show()

        else:
            layer = layers[0]
            layer_attn = attn_maps[layer]

            n_heads = layer_attn.shape[0]
            n_cols = min(6, n_heads)
            n_rows = math.ceil(n_heads / n_cols)

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
            axes = np.array(axes).reshape(-1)

            for h in range(n_heads):
                ax = axes[h]
                ax.imshow(layer_attn[h], cmap="Reds")
                ax.set_title(f"Layer {layer} - Head {h}")
                ax.axis("off")

            for j in range(n_heads, len(axes)):
                axes[j].axis("off")

            plt.tight_layout()
            plt.show()
