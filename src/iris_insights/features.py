"""Feature engineering helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class FeaturePipeline:
    """Wraps a Scikit-Learn column transformer for consistent preprocessing."""

    transformer: ColumnTransformer

    @classmethod
    def default(cls, columns: Iterable[str]) -> "FeaturePipeline":
        """Create the default feature pipeline for numeric columns."""

        transformer = ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    StandardScaler(),
                    list(columns),
                )
            ],
            remainder="passthrough",
        )
        return cls(transformer)

    def fit_transform(self, features: pd.DataFrame) -> np.ndarray:
        """Fit the transformer and return the transformed array."""

        return self.transformer.fit_transform(features)

    def transform(self, features: pd.DataFrame) -> np.ndarray:
        """Transform new feature data."""

        return self.transformer.transform(features)


def split_features_target(dataset: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split the dataset into features and target."""

    features = dataset.drop(columns=["species", "species_name"], errors="ignore")
    target = dataset.get("species")
    if target is None:
        raise KeyError("Dataset must include a 'species' column")
    return features, target
