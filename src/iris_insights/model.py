"""Model training utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from .features import FeaturePipeline


@dataclass
class ModelResult:
    """Container with model artifacts and evaluation outputs."""

    best_estimator: Pipeline
    best_params: Dict[str, object]
    classification_report: str
    confusion_matrix: np.ndarray


class IrisClassifier:
    """Train and evaluate classifiers for the Iris dataset."""

    def __init__(self, feature_pipeline: FeaturePipeline | None = None) -> None:
        self.feature_pipeline = feature_pipeline

    def _build_pipeline(self) -> Pipeline:
        classifier: ClassifierMixin = LogisticRegression(max_iter=500, multi_class="auto")
        pipeline = Pipeline(
            steps=[
                ("preprocess", self.feature_pipeline.transformer if self.feature_pipeline else "passthrough"),
                ("classifier", classifier),
            ]
        )
        return pipeline

    def train(self, features: np.ndarray, target: np.ndarray) -> ModelResult:
        """Train a logistic regression model with a simple hyper-parameter search."""

        pipeline = self._build_pipeline()
        param_grid = {
            "classifier__C": [0.1, 1.0, 10.0],
            "classifier__solver": ["lbfgs", "liblinear"],
        }
        search = GridSearchCV(pipeline, param_grid=param_grid, cv=5)
        search.fit(features, target)

        predictions = search.predict(features)
        report = classification_report(target, predictions)
        matrix = confusion_matrix(target, predictions)

        return ModelResult(
            best_estimator=search.best_estimator_,
            best_params=search.best_params_,
            classification_report=report,
            confusion_matrix=matrix,
        )
