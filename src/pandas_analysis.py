"""Utilities for preparing the pandas DataFrames used across the notebooks."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable

import pandas as pd


def export_reproduit_longform_entries(
    df: pd.DataFrame,
    *,
    output_dir: Path | str = Path("./temp"),
    filename: str = "st_reproduit_summary.csv",
) -> pd.DataFrame:
    """
    Filter the pandas dataframe to keep reproduced stereotypes and export summary.

    Parameters
    ----------
    df:
        Source Mandas dataframe.
    output_dir:
        Directory used to store the CSV output (created if missing).
    filename:
        Name of the CSV file written in ``output_dir``.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the filtered entries with the selected columns.
    """

    working_df = df.copy()

    required_columns: Iterable[str] = (
        "st_present",
        "st_mode",
        "title",
        "st_resume",
        "st_justification",
    )

    missing_columns = [col for col in required_columns if col not in working_df.columns]
    if missing_columns:
        missing_str = ", ".join(missing_columns)
        raise ValueError(f"Missing required columns in dataframe: {missing_str}.")

    mask = (working_df["st_present"] == "yes") & (working_df["st_mode"] == "reproduit")
    summary_df = working_df.loc[
        mask, ["title", "st_resume", "st_justification"]
    ].reset_index(drop=True)

    dest_dir = Path(output_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    output_path = dest_dir / filename
    summary_df.to_csv(output_path, index=False)

    return summary_df


def normalize_st_prismes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace malformed st_prismes entries with their ``nom`` field when needed.

    Rows containing serialized dictionaries (coming from a mistaken export)
    are converted so the ``nom`` value becomes the stored string in the list.
    """

    if "st_prismes" not in df.columns:
        raise ValueError("Missing required column 'st_prismes'.")

    working_df = df.copy()

    def _normalize_entry(entry: object) -> object:
        if isinstance(entry, dict):
            name = entry.get("nom")
            return name if isinstance(name, str) else entry

        if isinstance(entry, str):
            trimmed = entry.strip()
            if trimmed.startswith("{") and trimmed.endswith("}"):
                try:
                    parsed = ast.literal_eval(trimmed)
                except (ValueError, SyntaxError):
                    return entry
                if isinstance(parsed, dict):
                    return _normalize_entry(parsed)
            return entry

        return entry

    working_df["st_prismes"] = working_df["st_prismes"].apply(
        lambda values: [_normalize_entry(value) for value in values]
        if isinstance(values, list)
        else _normalize_entry(values)
    )

    return working_df
