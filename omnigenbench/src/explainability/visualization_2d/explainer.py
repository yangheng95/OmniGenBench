# -*- coding: utf-8 -*-
# file: __init__.py
# time: 2025-06-16 21:06
# author: Shasha Zhou <sz484@exeter.ac.uk>
# Copyright (C) 2020-2025. All Rights Reserved.

from ...abc.abstract_explainer import AbstractExplainer
from ...model.embedding.model import OmniModelForEmbedding
from ..shared_methods.tsne_explainer import TSNEExplainer
import pandas as pd
import plotly.express as px

from ...misc.utils import fprint

EXPLAINER_REGISTRY = {
    "tsne": TSNEExplainer,
}


def get_explainer(name: str) -> AbstractExplainer:
    """
    Get an explainer by name.
    """
    fprint(f"Getting explainer with method: {name}")
    return EXPLAINER_REGISTRY[name]


class Visualization2DExplainer(AbstractExplainer):
    """
    Visualize the dataset in two dimensions.
    """

    def __init__(self, model, method: str = "tsne"):
        """
        Initialize the explainer.

        Args:
            model (AbstractModel): The model to explain.
            dataset (AbstractDataset): The dataset to explain.
            method (str): The method to use for the explainer, "tsne" by default.
        """
        fprint(f"Initializing Visualization2DExplainer with method: {method}")
        super().__init__(model)
        assert isinstance(
            model, OmniModelForEmbedding
        ), "Model must be an instance of OmniModelForEmbedding"
        self.ExplainerClass = get_explainer(method)
        self.explainer = self.ExplainerClass(model)
        fprint("Visualization2DExplainer initialized successfully")

    def explain(self, sequences, labels=None, **kwargs):
        """
        Explain the input.

        Args:
            input: The input to explain.
            **kwargs: Additional keyword arguments.

        Returns:
            Any: The explanation.
        """
        fprint(f"Generating explanations for {len(sequences)} sequences")
        self.sequences = sequences
        self.labels = labels
        embeddings = self.explainer.explain(sequences, labels, **kwargs)
        fprint(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings

    def visualize(
        self,
        embeddings,
        sequences,
        labels=None,
        width=800,
        height=600,
        title="2D Visualization of Sequence Embeddings",
        point_size=8,
        point_opacity=0.8,
        wrap_width=50,
        color_palette=None,
        save_path=None,
        **kwargs,
    ):
        """
        Visualize the explanation.

        Args:
            embeddings: The embeddings to visualize.
            sequences: The sequences to visualize.
            labels: The labels to visualize. If None, all points are assigned 'Unlabeled'.
            width: The width of the figure.
            height: The height of the figure.
            title: The title of the figure.
            point_size: The size of the points.
            point_opacity: The opacity of the points.
            wrap_width: The width of the sequence.
            color_palette: The color palette.
            save_path: The path to save the figure.
            **kwargs: Additional keyword arguments.

        Returns:
            plotly.graph_objs._figure.Figure: The scatter plot figure object
        """
        fprint("Starting visualization process")
        fprint(f"Processing {len(sequences)} sequences for visualization")

        # Truncate long sequences
        wrapped_sequences = [
            seq[:wrap_width] + "..." if len(seq) > wrap_width else seq
            for seq in sequences
        ]

        # Handle labels
        if labels is None:
            labels_str = ["Unlabeled"] * len(sequences)
            fprint("No labels provided, using 'Unlabeled' for all points")
        else:
            labels_str = [str(label) for label in labels]
            fprint(f"Processing {len(set(labels_str))} unique labels")

        # Unique labels and colors
        unique_labels = sorted(set(labels_str))
        if color_palette is None:
            color_palette = (
                px.colors.qualitative.Set3
                + px.colors.qualitative.Pastel
                + px.colors.qualitative.Bold
            )
        color_discrete_map = {
            label: color_palette[i % len(color_palette)]
            for i, label in enumerate(unique_labels)
        }

        # DataFrame for plotting
        df = pd.DataFrame(
            {
                "x": embeddings[:, 0],
                "y": embeddings[:, 1],
                "label": labels_str,
                "sequence": wrapped_sequences,
            }
        )

        # Create scatter plot
        fig = px.scatter(
            df,
            x="x",
            y="y",
            color="label",
            hover_data={"sequence": True, "label": True},
            color_discrete_map=color_discrete_map,
            labels={"x": "Component 1", "y": "Component 2"},
        )

        # Style
        fig.update_traces(
            marker=dict(size=point_size, opacity=point_opacity, line=dict(width=0.5))
        )
        fig.update_layout(
            width=width,
            height=height,
            title={
                "text": title,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
            },
            legend_title_text="Label",
            legend=dict(bordercolor="Black", borderwidth=0.5, itemsizing="constant"),
            plot_bgcolor="rgba(245, 245, 245, 1)",
            paper_bgcolor="rgba(255,255,255,1)",
        )

        if save_path:
            fprint(f"Saving visualization to: {save_path}")
            fig.write_html(save_path)

        fprint("Visualization completed successfully")
        return fig

    def __call__(self, sequences, labels=None, **kwargs):
        """
        Call the explainer.
        """
        embeddings = self.explainer.explain(sequences, labels, **kwargs)
        fig = self.visualize(embeddings, sequences, labels, **kwargs)
        return fig
