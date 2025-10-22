"""Evaluation helpers that emit human-readable text files."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence
import math
import statistics

from .data import FEATURE_NAMES, SPECIES_NAMES


FIGURE_DIR = Path("reports/figures")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def _extract_feature_columns(dataset: Sequence[Dict[str, float | int | str]]) -> Dict[str, List[float]]:
    columns: Dict[str, List[float]] = {name: [] for name in FEATURE_NAMES}
    for row in dataset:
        for name in FEATURE_NAMES:
            columns[name].append(float(row[name]))
    return columns


def _pearson(x_values: Sequence[float], y_values: Sequence[float]) -> float:
    mean_x = statistics.fmean(x_values)
    mean_y = statistics.fmean(y_values)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_values, y_values))
    denominator = math.sqrt(
        sum((x - mean_x) ** 2 for x in x_values) * sum((y - mean_y) ** 2 for y in y_values)
    )
    if math.isclose(denominator, 0.0):
        return 0.0
    return numerator / denominator


def plot_correlation_matrix(
    dataset: Sequence[Dict[str, float | int | str]], filename: str = "correlation_matrix.txt"
) -> Path:
    """Compute a correlation matrix and persist it as CSV formatted text."""

    columns = _extract_feature_columns(dataset)
    header = ["feature"] + list(FEATURE_NAMES)
    lines = [",".join(header)]
    for feature in FEATURE_NAMES:
        row: List[str] = [feature]
        for other in FEATURE_NAMES:
            correlation = _pearson(columns[feature], columns[other])
            row.append(f"{correlation:.3f}")
        lines.append(",".join(row))
    output_path = FIGURE_DIR / filename
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def plot_pairwise(dataset: Sequence[Dict[str, float | int | str]], filename: str = "pairwise_summary.txt") -> Path:
    """Write summary statistics for each species and feature."""

    grouped: Dict[str, Dict[str, List[float]]] = {
        species: {feature: [] for feature in FEATURE_NAMES} for species in SPECIES_NAMES
    }
    for row in dataset:
        species = str(row["species_name"])
        for feature in FEATURE_NAMES:
            grouped[species][feature].append(float(row[feature]))

    lines: List[str] = []
    for species in SPECIES_NAMES:
        lines.append(f"## {species}")
        for feature in FEATURE_NAMES:
            values = grouped[species][feature]
            mean = statistics.fmean(values)
            minimum = min(values)
            maximum = max(values)
            stdev = statistics.pstdev(values)
            lines.append(
                f"{feature}: mean={mean:.3f} min={minimum:.3f} max={maximum:.3f} std={stdev:.3f}"
            )
        lines.append("")
    output_path = FIGURE_DIR / filename
    output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return output_path


def plot_confusion_matrix(
    matrix: Sequence[Sequence[int]],
    class_names: Sequence[str],
    filename: str = "confusion_matrix.txt",
) -> Path:
    """Render the confusion matrix as a table stored on disk."""

    header = ["actual\\predicted"] + list(class_names)
    lines = [",".join(header)]
    for name, row in zip(class_names, matrix):
        row_values = [str(value) for value in row]
        lines.append(",".join([name] + row_values))
    output_path = FIGURE_DIR / filename
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


__all__ = [
    "FIGURE_DIR",
    "plot_correlation_matrix",
    "plot_pairwise",
    "plot_confusion_matrix",
]
