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
    """
    Get an explainer by name.
    """
    fprint(f"Getting explainer with method: {name}")
    return EXPLAINER_REGISTRY[name]


class EpistasisExplainer(AbstractExplainer):
    """
    Explain the sequence logo of the model.
    """

    def __init__(self, model, method: str = "squid"):
        """
        Initialize the explainer.

        Args:

        """
        fprint(f"Initializing EpistasisExplainer with method: {method}")
        super().__init__(model)
        self.ExplainerClass = get_explainer(method)
        self.explainer = self.ExplainerClass(model, gpmap="pairwise")
        fprint("EpistasisExplainer initialized successfully")

    def explain(self, sequence, **kwargs):
        """
        Explain the sequence logo of the model.
        """
        fprint(f"Generating explanations for sequence: {sequence}")
        matrix = self.explainer.explain(sequence, **kwargs)
        self.matrix = matrix
        return matrix

    def visualize_heatmap(self, matrix, sequence: str, save_path=None, **kwargs):
        """
        Visualize the heatmap of the model.
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
        """
        Explain the sequence logo of the model.

        Args:
            sequence: The sequence to explain.
            save_path: The path to save the figure.
            visualize_type: The type of visualization. Can be "logo" or "heatmap".
            **kwargs: Additional keyword arguments.
        """
        fprint(f"Generating explanations for sequence: {sequence}")
        matrix = self.explainer.explain(sequence, gpmap="additive", **kwargs)
        self.visualize_heatmap(matrix, sequence, save_path=save_path, **kwargs)
