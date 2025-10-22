"""Model evaluation helpers."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


FIGURE_DIR = Path("reports/figures")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def plot_correlation_matrix(dataset: pd.DataFrame, filename: str = "correlation_matrix.png") -> Path:
    """Generate and save a correlation heatmap."""

    correlation = dataset.drop(columns=["species_name"], errors="ignore").corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation, annot=True, cmap="viridis", fmt=".2f")
    plt.title("Iris Feature Correlation Matrix")
    output_path = FIGURE_DIR / filename
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def plot_pairwise(dataset: pd.DataFrame, filename: str = "pairplot.png") -> Path:
    """Generate a pairplot showing the feature distributions."""

    pairplot = sns.pairplot(dataset, hue="species_name")
    pairplot.fig.suptitle("Iris Pairplot", y=1.02)
    output_path = FIGURE_DIR / filename
    pairplot.savefig(output_path)
    plt.close("all")
    return output_path


def plot_confusion_matrix(matrix: np.ndarray, class_names: list[str], filename: str = "confusion_matrix.png") -> Path:
    """Create a labelled confusion matrix plot."""

    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Model Confusion Matrix")
    output_path = FIGURE_DIR / filename
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path
