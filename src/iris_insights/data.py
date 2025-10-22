"""Data loading utilities for the Iris Insights project."""
from __future__ import annotations

import pandas as pd
from sklearn.datasets import load_iris


def load_dataset(as_frame: bool = True) -> pd.DataFrame:
    """Load the Iris dataset.

    Parameters
    ----------
    as_frame:
        When ``True`` the function returns a :class:`pandas.DataFrame`. If ``False`` it
        returns the raw :class:`sklearn.utils.Bunch` object provided by
        :func:`sklearn.datasets.load_iris`.

    Returns
    -------
    pandas.DataFrame
        Feature matrix combined with the target column when ``as_frame`` is ``True``.
    sklearn.utils.Bunch
        Raw dataset bundle when ``as_frame`` is ``False``.
    """

    dataset = load_iris(as_frame=True)
    if not as_frame:
        return dataset

    frame = dataset.frame.copy()
    frame.rename(columns={"target": "species"}, inplace=True)
    frame["species_name"] = frame["species"].map(dict(enumerate(dataset.target_names)))
    return frame
