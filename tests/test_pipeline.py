from __future__ import annotations

import numpy as np

from iris_insights import data, features, model


def test_full_training_pipeline():
    dataset = data.load_dataset()
    feature_frame, target = features.split_features_target(dataset)
    feature_pipeline = features.FeaturePipeline.default(feature_frame.columns)
    transformed_features = feature_pipeline.fit_transform(feature_frame)

    classifier = model.IrisClassifier(feature_pipeline)
    result = classifier.train(transformed_features, target.to_numpy())

    assert set(result.best_params.keys()) == {"classifier__C", "classifier__solver"}
    assert result.confusion_matrix.shape == (3, 3)
    assert np.trace(result.confusion_matrix) == len(dataset)
