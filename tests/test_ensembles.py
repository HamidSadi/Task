from __future__ import annotations

from iris_insights import (
    FEATURE_NAMES,
    FeaturePipeline,
    IrisClassifier,
    load_dataset,
    split_features_target,
)


def _prepare_features():
    dataset = load_dataset()
    feature_rows, target = split_features_target(dataset)
    pipeline = FeaturePipeline.default(FEATURE_NAMES)
    transformed = pipeline.fit_transform(feature_rows)
    return transformed, target, pipeline


def test_random_forest_classifier_matches_training_labels():
    transformed, target, pipeline = _prepare_features()
    classifier = IrisClassifier(
        pipeline,
        algorithm="random_forest",
        n_estimators=20,
        max_depth=5,
        seed=123,
    )
    result = classifier.train(transformed, target)
    assert result.best_params["strategy"] == "random_forest"
    assert result.accuracy == 1.0


def test_gradient_boosting_classifier_matches_training_labels():
    transformed, target, pipeline = _prepare_features()
    classifier = IrisClassifier(
        pipeline,
        algorithm="gradient_boosting",
        n_estimators=90,
        learning_rate=0.3,
    )
    result = classifier.train(transformed, target)
    assert result.best_params["strategy"] == "gradient_boosting"
    assert result.accuracy == 1.0
