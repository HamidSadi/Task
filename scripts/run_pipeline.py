"""Run the complete Iris Insights workflow using only the standard library."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from iris_insights import (
    FEATURE_NAMES,
    SPECIES_NAMES,
    FeaturePipeline,
    IrisClassifier,
    MLFlowLikeLogger,
    ModelResult,
    load_dataset,
    plot_confusion_matrix,
    plot_correlation_matrix,
    plot_pairwise,
    split_features_target,
)


REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    dataset = load_dataset()
    correlation_path = plot_correlation_matrix(dataset)
    pairwise_path = plot_pairwise(dataset)

    feature_rows, target = split_features_target(dataset)
    feature_pipeline = FeaturePipeline.default(FEATURE_NAMES)
    transformed_features = feature_pipeline.fit_transform(feature_rows)

    logger = MLFlowLikeLogger(experiment_name="iris_workflow")
    experiments = [
        ("nearest_centroid", {"algorithm": "nearest_centroid"}),
        (
            "random_forest",
            {
                "algorithm": "random_forest",
                "n_estimators": 25,
                "max_depth": 4,
                "seed": 99,
            },
        ),
        (
            "gradient_boosting",
            {
                "algorithm": "gradient_boosting",
                "n_estimators": 80,
                "learning_rate": 0.25,
            },
        ),
    ]

    experiment_results: list[tuple[str, ModelResult, Path]] = []
    for run_name, classifier_kwargs in experiments:
        classifier = IrisClassifier(feature_pipeline, **classifier_kwargs)
        result = classifier.train(transformed_features, target)
        class_names = [SPECIES_NAMES[label] for label in result.classes]
        confusion_filename = f"confusion_matrix_{run_name}.txt"
        confusion_path = plot_confusion_matrix(
            result.confusion_matrix, class_names, filename=confusion_filename
        )
        with logger.run(run_name):
            logger.log_params(result.best_params)
            logger.log_metrics({"accuracy": result.accuracy})
            logger.set_tags({"algorithm": result.best_params["strategy"]})
            logger.log_text(result.classification_report, "classification_report.txt")
            logger.log_artifact(confusion_path)
        experiment_results.append((run_name, result, confusion_path))

    report_path = REPORTS_DIR / "model_report.txt"
    report_lines = [
        "Iris Insights Model Report",
        "===========================",
        "",
        f"Correlation matrix saved to: {correlation_path}",
        f"Pairwise summary saved to: {pairwise_path}",
        "",
        "Experiment Summary:",
    ]
    for run_name, result, confusion_path in experiment_results:
        report_lines.extend(
            [
                f"- {run_name}: accuracy={result.accuracy:.3f}",
                f"  params: {result.best_params}",
                f"  confusion matrix: {confusion_path}",
            ]
        )
    report_lines.extend(
        [
            "",
            "See reports/experiments for per-run classification reports and metrics.",
        ]
    )
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"Report saved to {report_path.resolve()}")


if __name__ == "__main__":
    main()
