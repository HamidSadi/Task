"""Ensemble learning utilities implemented with the standard library."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence
import math
import random
import statistics


@dataclass
class TreeNode:
    """Decision tree node used by the random forest implementation."""

    prediction: int
    feature_index: int | None = None
    threshold: float | None = None
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None

    def is_leaf(self) -> bool:
        return self.feature_index is None or self.left is None or self.right is None


def _gini_impurity(labels: Sequence[int]) -> float:
    total = len(labels)
    if total == 0:
        return 0.0
    counts: dict[int, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    impurity = 1.0
    for count in counts.values():
        prob = count / total
        impurity -= prob * prob
    return impurity


def _majority_vote(labels: Sequence[int]) -> int:
    counts: dict[int, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return max(counts.items(), key=lambda item: (item[1], item[0]))[0]


def _best_split(
    samples: Sequence[Sequence[float]],
    labels: Sequence[int],
    feature_indices: Sequence[int],
) -> tuple[int | None, float | None, float]:
    best_feature: int | None = None
    best_threshold: float | None = None
    best_score = math.inf

    for feature_index in feature_indices:
        sorted_pairs = sorted(zip(samples, labels), key=lambda item: item[0][feature_index])
        if not sorted_pairs:
            continue
        feature_values = [pair[0][feature_index] for pair in sorted_pairs]
        label_values = [pair[1] for pair in sorted_pairs]
        unique_values = sorted(set(feature_values))
        if len(unique_values) == 1:
            continue
        # evaluate midpoints between sorted unique values
        for left_value, right_value in zip(unique_values, unique_values[1:]):
            threshold = (left_value + right_value) / 2.0
            left_labels = [
                label
                for value, label in zip(feature_values, label_values)
                if value <= threshold
            ]
            right_labels = [
                label
                for value, label in zip(feature_values, label_values)
                if value > threshold
            ]
            if not left_labels or not right_labels:
                continue
            left_impurity = _gini_impurity(left_labels)
            right_impurity = _gini_impurity(right_labels)
            score = (
                len(left_labels) / len(labels) * left_impurity
                + len(right_labels) / len(labels) * right_impurity
            )
            if score < best_score:
                best_score = score
                best_feature = feature_index
                best_threshold = threshold
    return best_feature, best_threshold, best_score


def _build_tree(
    samples: Sequence[Sequence[float]],
    labels: Sequence[int],
    max_depth: int,
    depth: int,
    rng: random.Random,
    max_features: int,
) -> TreeNode:
    prediction = _majority_vote(labels)
    if depth >= max_depth or len(set(labels)) == 1:
        return TreeNode(prediction=prediction)

    feature_indices = list(range(len(samples[0])))
    rng.shuffle(feature_indices)
    feature_subset = feature_indices[: max_features or len(feature_indices)]
    best_feature, threshold, _ = _best_split(samples, labels, feature_subset)

    if best_feature is None or threshold is None:
        return TreeNode(prediction=prediction)

    left_samples: List[List[float]] = []
    left_labels: List[int] = []
    right_samples: List[List[float]] = []
    right_labels: List[int] = []
    for sample, label in zip(samples, labels):
        if sample[best_feature] <= threshold:
            left_samples.append(list(sample))
            left_labels.append(label)
        else:
            right_samples.append(list(sample))
            right_labels.append(label)

    if not left_samples or not right_samples:
        return TreeNode(prediction=prediction)

    left_child = _build_tree(left_samples, left_labels, max_depth, depth + 1, rng, max_features)
    right_child = _build_tree(right_samples, right_labels, max_depth, depth + 1, rng, max_features)
    return TreeNode(
        prediction=prediction,
        feature_index=best_feature,
        threshold=threshold,
        left=left_child,
        right=right_child,
    )


@dataclass
class RandomForestModel:
    """Random forest classifier composed of decision trees."""

    trees: List[TreeNode]
    classes: List[int]

    def predict(self, samples: Sequence[Sequence[float]]) -> List[int]:
        predictions: List[int] = []
        for sample in samples:
            votes: dict[int, int] = {label: 0 for label in self.classes}
            for tree in self.trees:
                prediction = self._predict_tree(tree, sample)
                votes[prediction] = votes.get(prediction, 0) + 1
            predictions.append(max(votes.items(), key=lambda item: (item[1], item[0]))[0])
        return predictions

    def _predict_tree(self, node: TreeNode, sample: Sequence[float]) -> int:
        if node.is_leaf():
            return node.prediction
        assert node.feature_index is not None and node.threshold is not None
        if sample[node.feature_index] <= node.threshold:
            assert node.left is not None
            return self._predict_tree(node.left, sample)
        assert node.right is not None
        return self._predict_tree(node.right, sample)


def train_random_forest(
    samples: Sequence[Sequence[float]],
    labels: Sequence[int],
    n_estimators: int = 15,
    max_depth: int = 4,
    max_features: int | None = None,
    seed: int = 42,
) -> RandomForestModel:
    if not samples:
        raise ValueError("Cannot train random forest with no samples.")
    rng = random.Random(seed)
    n_features = len(samples[0])
    feature_subset = max_features or max(1, int(math.sqrt(n_features)))
    trees: List[TreeNode] = []
    for _ in range(n_estimators):
        bootstrap_samples: List[List[float]] = []
        bootstrap_labels: List[int] = []
        for _ in range(len(samples)):
            index = rng.randrange(len(samples))
            bootstrap_samples.append(list(samples[index]))
            bootstrap_labels.append(labels[index])
        tree = _build_tree(bootstrap_samples, bootstrap_labels, max_depth, 0, rng, feature_subset)
        trees.append(tree)
    classes = sorted(set(labels))
    return RandomForestModel(trees=trees, classes=classes)


@dataclass
class DecisionStump:
    feature_index: int
    threshold: float
    left_value: float
    right_value: float

    def predict(self, sample: Sequence[float]) -> float:
        if sample[self.feature_index] <= self.threshold:
            return self.left_value
        return self.right_value


def _fit_stump(
    samples: Sequence[Sequence[float]],
    residuals: Sequence[float],
) -> DecisionStump:
    best_feature = 0
    best_threshold = 0.0
    best_error = math.inf
    best_left = 0.0
    best_right = 0.0

    n_features = len(samples[0])
    for feature_index in range(n_features):
        sorted_pairs = sorted(
            zip(samples, residuals), key=lambda item: item[0][feature_index]
        )
        feature_values = [pair[0][feature_index] for pair in sorted_pairs]
        residual_values = [pair[1] for pair in sorted_pairs]
        unique_values = sorted(set(feature_values))
        if len(unique_values) == 1:
            continue
        for left_value, right_value in zip(unique_values, unique_values[1:]):
            threshold = (left_value + right_value) / 2.0
            left_residuals = [
                residual
                for value, residual in zip(feature_values, residual_values)
                if value <= threshold
            ]
            right_residuals = [
                residual
                for value, residual in zip(feature_values, residual_values)
                if value > threshold
            ]
            if not left_residuals or not right_residuals:
                continue
            left_mean = statistics.fmean(left_residuals)
            right_mean = statistics.fmean(right_residuals)
            error = sum((residual - left_mean) ** 2 for residual in left_residuals)
            error += sum((residual - right_mean) ** 2 for residual in right_residuals)
            if error < best_error:
                best_error = error
                best_feature = feature_index
                best_threshold = threshold
                best_left = left_mean
                best_right = right_mean
    return DecisionStump(best_feature, best_threshold, best_left, best_right)


@dataclass
class GradientBoostingModel:
    """Simple gradient boosting model that performs stage-wise regression."""

    initial_prediction: float
    stumps: List[DecisionStump]
    learning_rate: float
    classes: List[int]

    def predict(self, samples: Sequence[Sequence[float]]) -> List[int]:
        predictions: List[int] = []
        min_class = min(self.classes)
        max_class = max(self.classes)
        for sample in samples:
            value = self.initial_prediction
            for stump in self.stumps:
                value += self.learning_rate * stump.predict(sample)
            rounded = int(round(value))
            bounded = max(min_class, min(max_class, rounded))
            predictions.append(bounded)
        return predictions


def train_gradient_boosting(
    samples: Sequence[Sequence[float]],
    labels: Sequence[int],
    n_estimators: int = 40,
    learning_rate: float = 0.2,
) -> GradientBoostingModel:
    if not samples:
        raise ValueError("Cannot train gradient boosting model with no samples.")
    initial_prediction = statistics.fmean(labels)
    predictions = [initial_prediction for _ in labels]
    stumps: List[DecisionStump] = []
    for _ in range(n_estimators):
        residuals = [target - pred for target, pred in zip(labels, predictions)]
        stump = _fit_stump(samples, residuals)
        updates = [stump.predict(sample) for sample in samples]
        for index, update in enumerate(updates):
            predictions[index] += learning_rate * update
        stumps.append(stump)
    classes = sorted(set(labels))
    return GradientBoostingModel(
        initial_prediction=initial_prediction,
        stumps=stumps,
        learning_rate=learning_rate,
        classes=classes,
    )


__all__ = [
    "RandomForestModel",
    "GradientBoostingModel",
    "train_random_forest",
    "train_gradient_boosting",
]

