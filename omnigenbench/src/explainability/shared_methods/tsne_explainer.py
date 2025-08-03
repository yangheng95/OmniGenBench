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
    """
    TSNEExplainer is a class that uses t-SNE to visualize the dataset embedding in two dimensions.
    """
    def __init__(self, model, **kwargs):
        super().__init__(model)
        self.tsne = TSNE(n_components=2, **kwargs)

    def explain(self, sequences: List[str], labels: List[Union[int, str]], embedding_file: Optional[str] = None,  **kwargs):
        """
        Execute the t-SNE algorithm on the dataset.

        Args:
            embeddings: The embeddings to explain.
            embedding_file: load the embeddings from the file.
            **kwargs: Additional keyword arguments.

        Returns:
            Any: The explanation.
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