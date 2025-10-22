"""Iris Insights package."""

from .data import load_dataset
from .evaluation import (
    FIGURE_DIR,
    plot_confusion_matrix,
    plot_correlation_matrix,
    plot_pairwise,
)
from .features import FeaturePipeline, split_features_target
from .model import IrisClassifier, ModelResult

__all__ = [
    "load_dataset",
    "FIGURE_DIR",
    "plot_confusion_matrix",
    "plot_correlation_matrix",
    "plot_pairwise",
    "FeaturePipeline",
    "split_features_target",
    "IrisClassifier",
    "ModelResult",
]
