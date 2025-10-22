"""Synthetic Iris-like dataset generation for offline execution."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import random

FEATURE_NAMES: tuple[str, ...] = (
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
)

SPECIES_NAMES: tuple[str, ...] = ("setosa", "versicolor", "virginica")


@dataclass(frozen=True)
class IrisRecord:
    """Container representing a single Iris sample."""

    sample_id: int
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    species: int
    species_name: str

    def as_dict(self) -> Dict[str, float | int | str]:
        return {
            "sample_id": self.sample_id,
            "sepal_length": self.sepal_length,
            "sepal_width": self.sepal_width,
            "petal_length": self.petal_length,
            "petal_width": self.petal_width,
            "species": self.species,
            "species_name": self.species_name,
        }


_BASE_FEATURES: dict[str, tuple[float, float, float, float]] = {
    "setosa": (5.0, 3.5, 1.4, 0.2),
    "versicolor": (6.0, 2.8, 4.5, 1.3),
    "virginica": (6.5, 3.0, 5.5, 2.0),
}

_SPREAD: tuple[float, float, float, float] = (0.6, 0.4, 1.2, 0.6)


def _generate_sample(
    rng: random.Random,
    base: tuple[float, float, float, float],
    sample_id: int,
    species_index: int,
    species_name: str,
) -> IrisRecord:
    features = [
        round(base[idx] + (rng.random() - 0.5) * _SPREAD[idx], 3)
        for idx in range(len(base))
    ]
    return IrisRecord(
        sample_id=sample_id,
        sepal_length=features[0],
        sepal_width=features[1],
        petal_length=features[2],
        petal_width=features[3],
        species=species_index,
        species_name=species_name,
    )


def load_dataset(samples_per_class: int = 50) -> List[Dict[str, float | int | str]]:
    """Return a reproducible Iris-style dataset without external dependencies."""

    records: List[Dict[str, float | int | str]] = []
    global_sample_id = 0
    for species_index, species_name in enumerate(SPECIES_NAMES):
        rng = random.Random(100 + species_index)
        base = _BASE_FEATURES[species_name]
        for _ in range(samples_per_class):
            record = _generate_sample(rng, base, global_sample_id, species_index, species_name)
            records.append(record.as_dict())
            global_sample_id += 1
    return records


__all__ = ["FEATURE_NAMES", "SPECIES_NAMES", "IrisRecord", "load_dataset"]
