"""Run the complete Iris Insights workflow."""
from __future__ import annotations

from pathlib import Path

from iris_insights import data, evaluation, features, model


REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    dataset = data.load_dataset()
    evaluation.plot_correlation_matrix(dataset)
    evaluation.plot_pairwise(dataset)

    feature_frame, target = features.split_features_target(dataset)
    feature_pipeline = features.FeaturePipeline.default(feature_frame.columns)
    transformed_features = feature_pipeline.fit_transform(feature_frame)

    classifier = model.IrisClassifier(feature_pipeline)
    result = classifier.train(transformed_features, target.to_numpy())

    evaluation.plot_confusion_matrix(result.confusion_matrix, class_names=sorted(dataset["species_name"].unique()))

    report_path = REPORTS_DIR / "model_report.txt"
    report_content = [
        "Iris Insights Model Report",
        "===========================",
        "",
        "Best Parameters:",
        str(result.best_params),
        "",
        "Classification Report:",
        result.classification_report,
    ]
    report_path.write_text("\n".join(report_content), encoding="utf-8")

    print(f"Report saved to {report_path.resolve()}")


if __name__ == "__main__":
    main()
