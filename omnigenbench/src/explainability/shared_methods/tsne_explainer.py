#
# Author: Shasha Zhou <sz484@exeter.ac.uk>
# Description:
#
# Copyright (C) 2020-2025. All Rights Reserved.
#
# -*- coding: utf-8 -*-
# file: tsne_explainer.py
# time: 2025-06-16 21:33
# author: Shasha Zhou <sz484@exeter.ac.uk>
# Copyright (C) 2020-2025. All Rights Reserved.

from sklearn.manifold import TSNE
from typing import List, Union, Optional
from ...abc.abstract_explainer import AbstractExplainer
from ...misc.utils import fprint


class TSNEExplainer(AbstractExplainer):
    """Visualizes high-dimensional sequence embeddings in 2D using t-SNE.

    This explainer generates high-dimensional embeddings from a set of input sequences
    using a given model. It then applies the t-SNE (t-Distributed Stochastic
    Neighbor Embedding) algorithm to project these embeddings into a two-dimensional
    space. This is useful for visualizing the structure of the learned embedding
    space and observing how sequences with different labels cluster.

    Attributes:
        model: The model used to generate sequence embeddings.
        tsne (sklearn.manifold.TSNE): The t-SNE transformer instance.
    """

    def __init__(self, model, **kwargs):
        """Initializes the TSNEExplainer.

        Args:
            model: A model object capable of generating embeddings, which should
                   have a `batch_encode` method (e.g., `OmniModelForEmbedding`).
            **kwargs: Additional keyword arguments to be passed directly to the
                      `sklearn.manifold.TSNE` constructor. This allows for customization
                      of parameters like `perplexity`, `learning_rate`, `n_iter`, etc.
        """
        super().__init__(model)
        self.tsne = TSNE(n_components=2, **kwargs)

    def explain(
        self,
        sequences: List[str],
        labels: List[Union[int, str]],
        embedding_file: Optional[str] = None,
        **kwargs,
    ):
        """Generates 2D embeddings for a set of sequences using t-SNE.

        This method first obtains high-dimensional embeddings for the input sequences,
        either by generating them with the model or by loading them from a file.
        It then applies the fitted t-SNE algorithm to project these embeddings
        into a two-dimensional representation suitable for plotting.
        """
        fprint("Starting t-SNE explanation")

        if embedding_file is not None:
            fprint(f"Loading embeddings from {embedding_file}")
            model_embeddings = self.model.load_embeddings(embedding_file)
        else:
            fprint("Encoding sequences")
            model_embeddings = self.model.batch_encode(sequences, **kwargs)

        fprint("Fitting t-SNE")
        tsne_embeddings = self.tsne.fit_transform(model_embeddings)
        fprint("t-SNE explanation completed")
        return tsne_embeddings
