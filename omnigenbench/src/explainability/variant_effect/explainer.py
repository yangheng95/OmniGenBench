# -*- coding: utf-8 -*-
# file: explainer.py
# time: 2025-06-23 15:00
# author: Shasha Zhou <sz484@exeter.ac.uk>
# Copyright (C) 2020-2025. All Rights Reserved.

from ...abc.abstract_explainer import AbstractExplainer
from ...misc.utils import fprint

EXPLAINER_REGISTRY = {
    # "model_output": ModelOutputExplainer,
}

def get_explainer(name: str) -> AbstractExplainer:
    """
    Get an explainer by name.
    """
    fprint(f"Getting explainer with method: {name}")
    return EXPLAINER_REGISTRY[name]

class VariantEffectExplainer(AbstractExplainer):
    """
    Explain the variant effect of a sequence.
    """
    def __init__(self, model, method: str="model_output"):
        super().__init__(model)
        self.model = model
        self.ExplainerClass = get_explainer(method)
        self.explainer = self.ExplainerClass(model)
        fprint("VariantEffectExplainer initialized successfully")

    def explain(self, sequence, variant, **kwargs):
        

        
