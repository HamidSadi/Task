"""Model training utilities implemented with only the standard library."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence
import math

from .data import SPECIES_NAMES
from .ensembles import GradientBoostingModel, RandomForestModel, train_gradient_boosting, train_random_forest


@dataclass
class NearestCentroidModel:
    """Minimal classifier storing centroids for each class."""

    centroids: Dict[int, List[float]]

    def predict(self, samples: Sequence[Sequence[float]]) -> List[int]:
        return [self._predict_single(sample) for sample in samples]

    def _predict_single(self, sample: Sequence[float]) -> int:
        best_label = -1
        best_distance = float("inf")
        for label, centroid in self.centroids.items():
            distance = math.dist(sample, centroid)
            if distance < best_distance:
                best_distance = distance
                best_label = label
        return best_label


@dataclass
class ModelResult:
    """Container with model artefacts and evaluation outputs."""

    best_estimator: object
    best_params: Dict[str, object]
    classification_report: str
    confusion_matrix: List[List[int]]
    classes: List[int]
    metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        if "accuracy" in self.metrics:
            return self.metrics["accuracy"]
        total_correct = sum(row[idx] for idx, row in enumerate(self.confusion_matrix))
        total = sum(sum(row) for row in self.confusion_matrix)
        if total == 0:
            return 0.0
        return total_correct / total


class IrisClassifier:
    """Train and evaluate the configured classifier."""

    def __init__(
        self,
        feature_pipeline: object | None = None,
        algorithm: str = "nearest_centroid",
        **algorithm_params: object,
    ) -> None:
        self.feature_pipeline = feature_pipeline
        self.algorithm = algorithm
        self.algorithm_params = algorithm_params

    def train(self, features: Sequence[Sequence[float]], target: Sequence[int]) -> ModelResult:
        if not features:
            raise ValueError("Cannot train classifier with no feature data.")
        if len(features) != len(target):
            raise ValueError("Features and target must be the same length.")

        if self.algorithm == "nearest_centroid":
            return self._train_nearest_centroid(features, target)
        if self.algorithm == "random_forest":
            return self._train_random_forest(features, target)
        if self.algorithm == "gradient_boosting":
            return self._train_gradient_boosting(features, target)
        raise ValueError(f"Unknown algorithm '{self.algorithm}'.")

    def _train_nearest_centroid(
        self, features: Sequence[Sequence[float]], target: Sequence[int]
    ) -> ModelResult:
        centroids = self._compute_centroids(features, target)
        model = NearestCentroidModel(centroids)
        predictions = model.predict(features)
        return self._build_result(
            estimator=model,
            params={"strategy": "nearest_centroid"},
            target=target,
            predictions=predictions,
        )

    def _train_random_forest(
        self, features: Sequence[Sequence[float]], target: Sequence[int]
    ) -> ModelResult:
        params = {
            "n_estimators": int(self.algorithm_params.get("n_estimators", 25)),
            "max_depth": int(self.algorithm_params.get("max_depth", 4)),
            "max_features": self.algorithm_params.get("max_features"),
            "seed": int(self.algorithm_params.get("seed", 42)),
        }
        model = train_random_forest(
            features,
            target,
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            max_features=params["max_features"],
            seed=params["seed"],
        )
        predictions = model.predict(features)
        params_filtered = {k: v for k, v in params.items() if v is not None}
        params_filtered["strategy"] = "random_forest"
        return self._build_result(
            estimator=model,
            params=params_filtered,
            target=target,
            predictions=predictions,
        )

    def _train_gradient_boosting(
        self, features: Sequence[Sequence[float]], target: Sequence[int]
    ) -> ModelResult:
        params = {
            "n_estimators": int(self.algorithm_params.get("n_estimators", 60)),
            "learning_rate": float(self.algorithm_params.get("learning_rate", 0.2)),
        }
        model = train_gradient_boosting(
            features,
            target,
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
        )
        predictions = model.predict(features)
        params["strategy"] = "gradient_boosting"
        return self._build_result(
            estimator=model,
            params=params,
            target=target,
            predictions=predictions,
        )

    def _build_result(
        self,
        estimator: NearestCentroidModel | RandomForestModel | GradientBoostingModel | object,
        params: Dict[str, object],
        target: Sequence[int],
        predictions: Sequence[int],
    ) -> ModelResult:
        confusion_matrix, classes = self._confusion_matrix(target, predictions)
        report = self._classification_report(confusion_matrix, classes)
        accuracy = self._accuracy(confusion_matrix)
        return ModelResult(
            best_estimator=estimator,
            best_params=params,
            classification_report=report,
            confusion_matrix=confusion_matrix,
            classes=list(classes),
            metrics={"accuracy": accuracy},
        )

    @staticmethod
    def _compute_centroids(features: Sequence[Sequence[float]], target: Sequence[int]) -> Dict[int, List[float]]:
        sums: Dict[int, List[float]] = {}
        counts: Dict[int, int] = {}
        for vector, label in zip(features, target):
            if label not in sums:
                sums[label] = [0.0 for _ in vector]
                counts[label] = 0
            sums[label] = [prev + value for prev, value in zip(sums[label], vector)]
            counts[label] += 1
        return {label: [value / counts[label] for value in values] for label, values in sums.items()}

    @staticmethod
    def _confusion_matrix(true_labels: Sequence[int], predicted: Sequence[int]) -> tuple[List[List[int]], List[int]]:
        classes = sorted({*true_labels, *predicted})
        index = {label: idx for idx, label in enumerate(classes)}
        matrix = [[0 for _ in classes] for _ in classes]
        for actual, guess in zip(true_labels, predicted):
            matrix[index[actual]][index[guess]] += 1
        return matrix, classes

    @staticmethod
    def _classification_report(matrix: Sequence[Sequence[int]], classes: Sequence[int]) -> str:
        lines: List[str] = []
        total_correct = 0
        total = 0
        for idx, label in enumerate(classes):
            tp = matrix[idx][idx]
            fp = sum(row[idx] for row in matrix) - tp
            fn = sum(matrix[idx]) - tp
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            support = sum(matrix[idx])
            species_name = SPECIES_NAMES[label]
            lines.append(
                f"{species_name:>10} | precision={precision:.3f} recall={recall:.3f} f1={f1:.3f} support={support}"
            )
            total_correct += tp
            total += support
        accuracy = total_correct / total if total else 0.0
        lines.append(f"Overall accuracy: {accuracy:.3f}")
        return "\n".join(lines)

    @staticmethod
    def _accuracy(matrix: Sequence[Sequence[int]]) -> float:
        total_correct = sum(row[idx] for idx, row in enumerate(matrix))
        total = sum(sum(row) for row in matrix)
        return total_correct / total if total else 0.0


__all__ = ["NearestCentroidModel", "ModelResult", "IrisClassifier"]
