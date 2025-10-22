"""Feature engineering helpers without external dependencies."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple
import math
import statistics

from .data import FEATURE_NAMES


@dataclass
class FeaturePipeline:
    """Simple z-score normalisation pipeline."""

    feature_names: Tuple[str, ...]
    means: Dict[str, float] = field(default_factory=dict)
    stds: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def default(cls, columns: Iterable[str] = FEATURE_NAMES) -> "FeaturePipeline":
        return cls(tuple(columns))

    def fit_transform(self, features: Sequence[Dict[str, float]]) -> List[List[float]]:
        matrix = self._to_matrix(features)
        self._fit_statistics(matrix)
        return [self._standardise(row) for row in matrix]

    def transform(self, features: Sequence[Dict[str, float]]) -> List[List[float]]:
        if not self.means or not self.stds:
            raise ValueError("Pipeline must be fitted before calling transform().")
        matrix = self._to_matrix(features)
        return [self._standardise(row) for row in matrix]

    def _to_matrix(self, features: Sequence[Dict[str, float]]) -> List[List[float]]:
        return [[row[name] for name in self.feature_names] for row in features]

    def _fit_statistics(self, matrix: Sequence[Sequence[float]]) -> None:
        for col_index, name in enumerate(self.feature_names):
            column = [row[col_index] for row in matrix]
            mean = statistics.fmean(column)
            std = statistics.pstdev(column)
            self.means[name] = mean
            self.stds[name] = std if not math.isclose(std, 0.0) else 1.0

    def _standardise(self, row: Sequence[float]) -> List[float]:
        transformed: List[float] = []
        for idx, name in enumerate(self.feature_names):
            value = row[idx]
            transformed.append((value - self.means[name]) / self.stds[name])
        return transformed


def split_features_target(dataset: Sequence[Dict[str, float | int | str]]) -> tuple[List[Dict[str, float]], List[int]]:
    """Split dataset dictionaries into features and numeric targets."""

    feature_dicts: List[Dict[str, float]] = []
    targets: List[int] = []
    for row in dataset:
        feature_dicts.append({name: float(row[name]) for name in FEATURE_NAMES})
        targets.append(int(row["species"]))
    return feature_dicts, targets


__all__ = ["FeaturePipeline", "split_features_target"]
