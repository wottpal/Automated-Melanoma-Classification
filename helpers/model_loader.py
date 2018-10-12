# -*- coding: utf-8 -*-
"""
Loads model and keeps it in memory while the script is running.
"""

from helpers import metrics, weighted_loss
from keras import models
import os



LOADED_MODELS = {}

def get_model(model_path):
    """Saves loaded model globally to speed up subsequent loads."""
    if model_path in LOADED_MODELS:
        return LOADED_MODELS[model_path]

    print(f"\nLoad Model under '{model_path}'..")
    if not os.path.isfile(model_path): raise ValueError(f"The model '{model_path}' doesn't exist")
    LOADED_MODELS[model_path] = models.load_model(model_path, custom_objects={
        'auc': metrics.auc, 
        'precision': metrics.precision, 
        'recall': metrics.recall,
        'w_categorical_crossentropy': weighted_loss.WeightedCategoricalCrossEntropy({0: 0.64, 1: 2.28})  # TODO IMPORTANT

    })

    return LOADED_MODELS[model_path]
