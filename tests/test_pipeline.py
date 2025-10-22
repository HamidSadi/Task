from __future__ import annotations

from iris_insights import (
    FEATURE_NAMES,
    SPECIES_NAMES,
    FeaturePipeline,
    IrisClassifier,
    load_dataset,
    split_features_target,
)


def test_full_training_pipeline():
    dataset = load_dataset()
    feature_rows, target = split_features_target(dataset)

    pipeline = FeaturePipeline.default(FEATURE_NAMES)
    transformed = pipeline.fit_transform(feature_rows)

    classifier = IrisClassifier(pipeline)
    result = classifier.train(transformed, target)

    assert result.best_params == {"strategy": "nearest_centroid"}
    assert len(result.confusion_matrix) == len(result.classes)
    correct_predictions = sum(
        row[idx] for idx, row in enumerate(result.confusion_matrix)
    )
    assert correct_predictions == len(dataset)
    assert [SPECIES_NAMES[label] for label in result.classes] == list(SPECIES_NAMES)
    assert result.metrics["accuracy"] == 1.0
