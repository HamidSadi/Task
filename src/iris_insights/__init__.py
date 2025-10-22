"""Iris Insights package."""

from .data import FEATURE_NAMES, SPECIES_NAMES, load_dataset
from .evaluation import (
    FIGURE_DIR,
    plot_confusion_matrix,
    plot_correlation_matrix,
    plot_pairwise,
)
from .features import FeaturePipeline, split_features_target
from .ensembles import (
    GradientBoostingModel,
    RandomForestModel,
    train_gradient_boosting,
    train_random_forest,
)
from .experiment_logging import MLFlowLikeLogger
from .model import IrisClassifier, ModelResult, NearestCentroidModel

__all__ = [
    "FEATURE_NAMES",
    "SPECIES_NAMES",
    "load_dataset",
    "FIGURE_DIR",
    "plot_confusion_matrix",
    "plot_correlation_matrix",
    "plot_pairwise",
    "FeaturePipeline",
    "split_features_target",
    "IrisClassifier",
    "ModelResult",
    "NearestCentroidModel",
    "RandomForestModel",
    "GradientBoostingModel",
    "train_random_forest",
    "train_gradient_boosting",
    "MLFlowLikeLogger",
]
