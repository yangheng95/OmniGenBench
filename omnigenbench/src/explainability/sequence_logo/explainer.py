#
# Author: Shasha Zhou <sz484@exeter.ac.uk>
# Description:
#
# Copyright (C) 2020-2025. All Rights Reserved.


from ...abc.abstract_explainer import AbstractExplainer
from ..shared_methods.squid_explainer import SQUIDExplainer
from ..shared_methods.ism_explainer import ISMExplainer
from ..shared_methods.lime_explainer import LIMEExplainer
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import logomaker
from ...misc.utils import fprint

EXPLAINER_REGISTRY = {
    "ism": ISMExplainer,
    "squid": SQUIDExplainer,
    "lime": LIMEExplainer,
}


def get_explainer(name: str) -> AbstractExplainer:
    """Retrieves an explainer class from the registry by name.

    This function acts as a factory to access different explanation
    methods that have been registered in the `EXPLAINER_REGISTRY`.

    Args:
        name (str): The name of the explainer method to retrieve.

    Returns:
        AbstractExplainer: The explainer class corresponding to the given name.
    """
    fprint(f"Getting explainer with method: {name}")
    return EXPLAINER_REGISTRY[name]


class SequenceLogoExplainer(AbstractExplainer):
    """
    A high-level wrapper for generating and visualizing model explanations.

    This class provides a simple interface to use various underlying attribution
    methods (like 'squid') to explain a model's predictions on a given sequence.
    It can generate attribution scores and visualize them as either a sequence logo
    or an interactive heatmap.

    Attributes:
        ExplainerClass: The underlying explainer class retrieved from the registry.
        explainer: An instance of the `ExplainerClass` used to compute attributions.
        matrix: Stores the most recently computed attribution matrix.

    Example:
        >>> from omnigenbench import OmniModelForPrediction
        >>> from omnigenbench.explainers import SequenceLogoExplainer
        >>> # Load a model trained for a specific task
        >>> model = OmniModelForPrediction.from_pretrained("anonymous8/OmniGenome-186M-Promoter")
        >>> # Initialize the explainer
        >>> explainer = SequenceLogoExplainer(model)
        >>> sequence = "AGCGTTAGAC"
        >>> # Generate and visualize the explanation as a sequence logo
        >>> explainer(sequence, visualize_type="logo")
    """

    def __init__(self, model, method: str = "squid"):
        """Initializes the SequenceLogoExplainer.

        Args:
            model: The pre-trained model to be explained. This should be a model
                   compatible with the chosen explanation method.
            method (str, optional): The specific explanation method to use. The method
                                    must be registered in `EXPLAINER_REGISTRY`.
                                    Defaults to "squid".
        """
        fprint(f"Initializing SequenceLogoExplainer with method: {method}")
        super().__init__(model)
        self.ExplainerClass = get_explainer(method)
        self.explainer = self.ExplainerClass(model, gpmap="additive")
        fprint("SequenceLogoExplainer initialized successfully")

    def explain(self, sequence, **kwargs):
        """Generates an attribution matrix for a given sequence.

        This method uses the underlying explainer (e.g., 'squid') to compute
        the attribution scores for each character at each position in the input
        sequence.

        Args:
            sequence (str): The input DNA or protein sequence to explain.
            **kwargs: Additional keyword arguments to be passed to the underlying
                      explainer's `explain` method.

        Returns:
            np.ndarray: A matrix of attribution scores, typically with a shape of
                        (sequence_length, alphabet_size).
        """
        fprint(f"Generating explanations for sequence: {sequence}")
        matrix = self.explainer.explain(sequence, **kwargs)
        self.matrix = matrix
        return matrix

    def visualize_logo(self, matrix, save_path=None, **kwargs):
        """
        Visualizes an attribution matrix as a sequence logo.

        This method uses the `logomaker` library to create a sequence logo. The
        height of each character visually represents its attribution score at that
        position. The plot can be customized using various keyword arguments.


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
        Visualizes an attribution matrix as an interactive heatmap.

        This method uses the `plotly` library to create a heatmap where the color of
        each cell represents the attribution score for a specific character at a
        specific position. The plot is interactive, allowing for hovering to see
        exact values.

        Args:
            matrix (np.ndarray): The attribution matrix to visualize, with shape
                                 (sequence_length, alphabet_size).
            sequence (str): The input sequence, used for labeling the x-axis.
            save_path (str, optional): The file path to save the generated plot.
                                       Note: saving interactive plots may require
                                       additional libraries like 'kaleido'.
            **kwargs: Additional keyword arguments for customizing the plot, including:
                - title (str): The title of the plot.
                - width (int): The width of the plot in pixels.
                - height (int): The height of the plot in pixels.
                - xaxis_title (str): The title for the x-axis.
                - yaxis_title (str): The title for the y-axis.
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
        Generates and visualizes an explanation for a sequence.

        This is a convenience method that combines the `explain` and visualization
        steps into a single call. It computes the attribution matrix and then
        displays it as either a sequence logo or a heatmap.

        Args:
            sequence (str): The input sequence to explain and visualize.
            save_path (str, optional): The file path to save the visualization.
                                       Defaults to None.
            visualize_type (str, optional): The type of visualization to generate,
                                            either "logo" or "heatmap".
                                            Defaults to "logo".
            **kwargs: Additional keyword arguments passed to both the `explain`
                      method and the chosen visualization method.
        """
        fprint(f"Generating explanations for sequence: {sequence}")
        matrix = self.explainer.explain(sequence, gpmap="additive", **kwargs)
        if visualize_type == "logo":
            self.visualize_logo(matrix, save_path=save_path, **kwargs)
        elif visualize_type == "heatmap":
            self.visualize_heatmap(matrix, sequence, **kwargs)
        else:
            raise ValueError(f"Invalid visualize_type: {visualize_type}")
