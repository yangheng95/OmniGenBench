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
    """Retrieves an explainer class from the registry by its name.

    This function acts as a factory, allowing for dynamic selection of the
    dimensionality reduction algorithm to be used.

    Args:
        name (str): The name of the explainer method to retrieve (e.g., "tsne").

    Returns:
        AbstractExplainer: The explainer class corresponding to the given name.
    """
    fprint(f"Getting explainer with method: {name}")
    return EXPLAINER_REGISTRY[name]


class Visualization2DExplainer(AbstractExplainer):
    """A high-level explainer for creating 2D visualizations of sequence embeddings.

    This class provides a convenient wrapper around various dimensionality reduction
    algorithms (like t-SNE) to generate and visualize 2D representations of
    high-dimensional sequence embeddings. It simplifies the process of creating
    interactive scatter plots to explore the structure of the embedding space.

    Attributes:
        model (OmniModelForEmbedding): The model used for generating embeddings.
        ExplainerClass (AbstractExplainer): The specific dimensionality reduction class being used (e.g., TSNEExplainer).
        explainer (AbstractExplainer): An instance of the `ExplainerClass`.
    """

    def __init__(self, model, method: str = "tsne"):
        """Initializes the Visualization2DExplainer.

        Args:
            model (OmniModelForEmbedding): The model to explain. It must be an instance of
                                           `OmniModelForEmbedding` as it needs the
                                           `batch_encode` method.
            method (str, optional): The dimensionality reduction method to use.
                                    Currently, only "tsne" is supported. Defaults to "tsne".
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
        """Generates the 2D embeddings for the input sequences.

        This method acts as a wrapper, calling the `explain` method of the underlying
        dimensionality reduction explainer (e.g., TSNEExplainer).

        Args:
            sequences (List[str]): The list of input sequences to explain.
            labels (Optional[List[Any]], optional): A list of corresponding labels.
                                                    Not used in computation but passed down.
                                                    Defaults to None.
            **kwargs: Additional keyword arguments to be passed to the underlying
                      explainer's `explain` method (e.g., `perplexity` for t-SNE).

        Returns:
            np.ndarray: An array of shape `(n_sequences, 2)` containing the
                        generated 2D coordinates.
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
        """Creates an interactive 2D scatter plot of the embeddings.

        This method uses Plotly Express to generate a rich, interactive visualization
        where each point represents a sequence. Hovering over a point reveals its
        sequence and label.

        Args:
            embeddings (np.ndarray): The 2D coordinates to visualize, shape `(n, 2)`.
            sequences (List[str]): The original sequences, used for hover-over tooltips.
            labels (Optional[List[Any]], optional): Labels for coloring points. If None, all points
                                                    are assigned a single 'Unlabeled' category. Defaults to None.
            width (int, optional): The width of the figure in pixels. Defaults to 800.
            height (int, optional): The height of the figure in pixels. Defaults to 600.
            title (str, optional): The title of the plot. Defaults to "2D Visualization of Sequence Embeddings".
            point_size (int, optional): The size of the scatter plot points. Defaults to 8.
            point_opacity (float, optional): The opacity of the points. Defaults to 0.8.
            wrap_width (int, optional): The maximum width for sequence text in the hover tooltip
                                        before it's truncated. Defaults to 50.
            color_palette (Optional[List[str]], optional): A list of CSS colors to use. If None, a default
                                                           Plotly palette is used. Defaults to None.
            save_path (Optional[str], optional): The file path to save the interactive plot as an HTML file.
                                                 If None, the plot is not saved. Defaults to None.
            **kwargs: Not currently used, but included for future extensibility.

        Returns:
            plotly.graph_objs._figure.Figure: The Plotly scatter plot figure object, which can be
                                              further customized or displayed.
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
        """A convenience method to generate and visualize the explanation in one step.

        This method chains the `explain` and `visualize` calls, providing a simple
        one-line interface for the most common use case.

        Args:
            sequences (List[str]): The list of input sequences.
            labels (Optional[List[Any]], optional): The corresponding labels for the sequences.
                                                    Defaults to None.
            **kwargs: Additional keyword arguments passed to both the `explain` and
                      `visualize` methods.

        Returns:
            plotly.graph_objs._figure.Figure: The final, interactive Plotly figure object.
        """
        embeddings = self.explainer.explain(sequences, labels, **kwargs)
        fig = self.visualize(embeddings, sequences, labels, **kwargs)
        return fig
