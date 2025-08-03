#
# Author: Shasha Zhou <sz484@exeter.ac.uk>
# Description:
#
# Copyright (C) 2020-2025. All Rights Reserved.
#
# -*- coding: utf-8 -*-
# file: __init__.py
# time: 2025-07-10 10:25
# author: Shasha Zhou <sz484@exeter.ac.uk>
# Copyright (C) 2020-2025. All Rights Reserved.

from ...abc.abstract_explainer import AbstractExplainer
from ..shared_methods.squid_explainer import SQUIDExplainer
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import logomaker
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


class SequenceLogoExplainer(AbstractExplainer):
    """
    Explain the sequence logo of the model.
    """

    def __init__(self, model, method: str = "squid"):
        """
        Initialize the explainer.

        Args:

        """
        fprint(f"Initializing SequenceLogoExplainer with method: {method}")
        super().__init__(model)
        self.ExplainerClass = get_explainer(method)
        self.explainer = self.ExplainerClass(model, gpmap="additive")
        fprint("SequenceLogoExplainer initialized successfully")

    def explain(self, sequence, **kwargs):
        """
        Explain the sequence logo of the model.
        """
        fprint(f"Generating explanations for sequence: {sequence}")
        matrix = self.explainer.explain(sequence, **kwargs)
        self.matrix = matrix
        return matrix

    def visualize_logo(self, matrix, save_path=None, **kwargs):
        """
        Visualize the sequence logo of the model. We use the logomaker and matplotlib packages to visualize the logo.

        Args:
            logo: The logo to visualize.
            **kwargs: Additional keyword arguments. Mainly used for the following parameters:
                - figsize: The size of the figure.
                - ylabel: The label of the y-axis.
                - xlabel: The label of the x-axis.
                - title: The title of the figure.
                - color_scheme: The color scheme of the logo.
                - show_spines: bool, whether to show the spines of the logo.
                - spines: list, the spines to show.
                - show_ticks: bool, whether to show x/y ticks.
                - colors: dict, the colors of the logo.
        """
        fprint("Visualizing the sequence logo...")
        # Transform the logo to a DataFrame
        df = pd.DataFrame(matrix, columns=self.explainer.alphabet)

        figsize = kwargs.get("figsize", (12, 3))
        ylabel = kwargs.get("ylabel", "Additive effect")
        xlabel = kwargs.get("xlabel", None)
        title = kwargs.get("title", None)
        color_scheme = kwargs.get("color_scheme", "classic")
        show_spines = kwargs.get("show_spines", True)
        spines = kwargs.get("spines", ["left", "bottom"])
        show_ticks = kwargs.get("show_ticks", True)
        colors = kwargs.get("colors", None)

        plt.figure(figsize=figsize)
        logo = logomaker.Logo(df, color_scheme=color_scheme)

        if show_spines:
            logo.style_spines(spines=spines, visible=True)
        else:
            logo.style_spines(visible=False)

        if not show_ticks:
            logo.ax.set_xticks([])
            logo.ax.set_yticks([])

        logo.ax.set_ylabel(ylabel)
        logo.ax.set_xlabel(xlabel)
        logo.ax.set_title(title)
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path)

        plt.show()

    def visualize_heatmap(self, matrix, sequence: str, save_path=None, **kwargs):
        """
        Visualize the heatmap of the model.
        """
        fprint("Visualizing the heatmap...")
        df = pd.DataFrame(matrix, columns=self.explainer.alphabet)

        seq_labels = [f"{i+1}-{base}" for i, base in enumerate(sequence)]

        # Transpose to get base × position
        heatmap_data = df.T.values  # shape: (4, L)

        fig = go.Figure(
            data=go.Heatmap(
                z=heatmap_data,
                x=seq_labels,
                y=self.explainer.alphabet,
                colorscale="RdBu_r",
                zmid=0,
                colorbar=dict(
                    thickness=kwargs.get("colorbar_thickness", 15),
                ),
                hovertemplate="Position %{x}<br>Base %{y}<br>Additive effect %{z:.4f}<extra></extra>",
            )
        )

        fig.update_layout(
            title=dict(
                text=kwargs.get("title", None),
                x=0.5,
            ),
            xaxis=dict(
                title=kwargs.get("xaxis_title", None),
                tickangle=45,
                showgrid=False,
            ),
            yaxis=dict(
                title=kwargs.get("yaxis_title", "Additive effect"),
                showgrid=False,
            ),
            margin=dict(l=40, r=40, t=60, b=40),
            plot_bgcolor="white",
            paper_bgcolor="white",
            width=min(1000, 40 * len(sequence)),
            height=kwargs.get("height", 200),
        )

        fig.show()

    def __call__(self, sequence, save_path=None, visualize_type="logo", **kwargs):
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
        if visualize_type == "logo":
            self.visualize_logo(matrix, save_path=save_path, **kwargs)
        elif visualize_type == "heatmap":
            self.visualize_heatmap(matrix, sequence, **kwargs)
        else:
            raise ValueError(f"Invalid visualize_type: {visualize_type}")
