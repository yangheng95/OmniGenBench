# -*- coding: utf-8 -*-
# file: __init__.py
# time: 2025-07-10 13:57
# author: Shasha Zhou <sz484@exeter.ac.uk>
# Copyright (C) 2020-2025. All Rights Reserved.

from ...abc.abstract_explainer import AbstractExplainer
from ..shared_methods.squid_explainer import SQUIDExplainer
import plotly.graph_objects as go
import numpy as np
from ...misc.utils import fprint


EXPLAINER_REGISTRY = {
    "squid": SQUIDExplainer,
}


def get_explainer(name: str) -> AbstractExplainer:
    """Retrieves an explainer class from the registry by its name.

    Args:
        name (str): The name of the explainer method to retrieve.

    Returns:
        AbstractExplainer: The explainer class corresponding to the given name.
    """
    fprint(f"Getting explainer with method: {name}")
    return EXPLAINER_REGISTRY[name]


class EpistasisExplainer(AbstractExplainer):
    """Explains and visualizes pairwise epistatic interactions in a sequence.

    This explainer uses an underlying method (like SQUID) to fit a pairwise
    surrogate model to the target model's predictions. It then extracts the
    second-order interaction terms (epistasis) and visualizes them as an
    interactive heatmap, showing the effect of combining mutations at two
    different positions.

    Attributes:
        ExplainerClass (AbstractExplainer): The underlying explainer class (e.g., SQUIDExplainer).
        explainer (AbstractExplainer): An instance of the explainer, configured for pairwise analysis.
        matrix (np.ndarray): The most recently computed epistatic interaction matrix.
    """

    def __init__(self, model, method: str = "squid"):
        """Initializes the EpistasisExplainer.

        Args:
            model (Any): The model to explain, which must be compatible with the
                         chosen underlying explainer method.
            method (str, optional): The method to use for explaining epistasis.
                                    Defaults to "squid".
        """
        fprint(f"Initializing EpistasisExplainer with method: {method}")
        super().__init__(model)
        self.ExplainerClass = get_explainer(method)
        self.explainer = self.ExplainerClass(model, gpmap="pairwise")
        fprint("EpistasisExplainer initialized successfully")

    def explain(self, sequence, **kwargs):
        """Computes the pairwise interaction matrix for a given sequence.

        This method calls the underlying SQUID explainer to generate the epistasis
        matrix (`theta_lclc`), which quantifies the interaction effect between
        every pair of possible mutations.

        Args:
            sequence (str): The input sequence to explain.
            **kwargs: Additional keyword arguments passed to the underlying
                      explainer's `explain` method.

        Returns:
            np.ndarray: A 4D numpy array of shape `(L, A, L, A)`, where `L` is the
                        sequence length and `A` is the alphabet size. `matrix[l1, c1, l2, c2]`
                        represents the interaction effect between character `c1` at
                        position `l1` and character `c2` at position `l2`.
        """
        fprint(f"Generating explanations for sequence: {sequence}")
        matrix = self.explainer.explain(sequence, **kwargs)
        self.matrix = matrix
        return matrix

    def visualize_heatmap(self, matrix, sequence: str, save_path=None, **kwargs):
        """Visualizes the epistatic interaction matrix as an interactive heatmap.

        This method creates a detailed heatmap where each cell represents the
        interaction strength between two specific mutations. The heatmap is
        lower-triangular to avoid redundancy.

        Args:
            matrix (np.ndarray): The 4D epistasis matrix from the `explain` method.
            sequence (str): The original sequence, used for context.
            save_path (str, optional): Path to save the interactive HTML plot. Defaults to None.
            **kwargs: Not currently used, but included for future extensibility.
        """
        fprint("Visualizing the heatmap...")

        L, A, _, _ = matrix.shape
        matrix = matrix.reshape(L * A, L * A)
        mask = np.tri(L * A, L * A, k=0, dtype=bool)
        masked_matrix = np.where(mask, matrix, np.nan)

        labels = [f"{i}:{base}" for i in range(L) for base in self.explainer.alphabet]

        fig = go.Figure(
            data=go.Heatmap(
                z=masked_matrix,
                x=labels,
                y=labels,
                colorscale="RdBu_r",
                zmid=0,
                zmin=-np.max(np.abs(matrix)),
                zmax=np.max(np.abs(matrix)),
                colorbar=dict(thickness=10, len=0.5, yanchor="middle", y=0.5),
                hovertemplate="From %{y}<br>To %{x}<br>Value: %{z:.4f}<extra></extra>",
            )
        )

        shapes = []

        stair_path = ""

        for i in range(L * A):
            x = i - 0.5
            y = i - 0.5
            if i == 0:
                stair_path += f"M {x},{y} "
            stair_path += f"L {x + 1},{y} "
            if i < L * A - 1:
                stair_path += f"L {x + 1},{y + 1} "

        stair_path += f"L {(L*A) - 0.5},{(L*A) - 0.5}"

        shapes.append(
            dict(
                type="path",
                path=stair_path,
                line=dict(color="black", width=0.4),
                fillcolor="rgba(0,0,0,0)",
                layer="above",
            )
        )

        for i in range(0, L + 1):
            grid_pos = i * A - 0.5

            shapes.append(
                dict(
                    type="line",
                    x0=grid_pos,
                    y0=grid_pos,
                    x1=grid_pos,
                    y1=L * A - 0.5,
                    line=dict(color="black", width=0.4),
                )
            )

            shapes.append(
                dict(
                    type="line",
                    x0=-0.5,
                    y0=grid_pos,
                    x1=grid_pos,
                    y1=grid_pos,
                    line=dict(color="black", width=0.4),
                )
            )

        fig.update_layout(
            title=None,
            xaxis=dict(
                scaleanchor="y",
                scaleratio=1,
                tickangle=45,
                showgrid=False,
                showticklabels=False,
            ),
            yaxis=dict(
                scaleanchor="x",
                scaleratio=1,
                autorange="reversed",
                showgrid=False,
                showticklabels=False,
            ),
            shapes=shapes,
            width=500,
            height=500,
            margin=dict(t=0, b=0, l=20, r=50),
            plot_bgcolor="white",
        )

        fig.show()

    def __call__(self, sequence, save_path=None, **kwargs):
        """A convenience method to explain and visualize in one step.

        Args:
            sequence (str): The sequence to explain.
            save_path (str, optional): The path to save the figure. Defaults to None.
            **kwargs: Additional keyword arguments passed to the `explain` method.
        """
        fprint(f"Generating explanations for sequence: {sequence}")
        matrix = self.explainer.explain(sequence, gpmap="additive", **kwargs)
        self.visualize_heatmap(matrix, sequence, save_path=save_path, **kwargs)
