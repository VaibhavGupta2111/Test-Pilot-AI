#!/usr/bin/env python
import pandas as pd
from io import BytesIO
from typing import Any, Dict, List

REQUIRED_COLS = [
    "Summary",
    "Issue key",
    "Description",
    "Priority",
    "Status",
    "Linked Issues",
]


def load_issues(source: Any) -> pd.DataFrame:
    """
    Reads an Excel file (path or file-like) and returns a cleaned DataFrame.
    Ensures REQUIRED_COLS exist and parses 'Linked Issues' into a list.
    """
    df = pd.read_excel(source, engine="openpyxl")
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError("Missing columns in input: " + ", ".join(missing))

    df = df.copy()
    df["Issue key"]   = df["Issue key"].astype(str).str.strip()
    df["Summary"]     = df["Summary"].fillna("").astype(str).str.strip()
    df["Description"] = df["Description"].fillna("").astype(str).str.strip()
    df["Linked Issues"] = (
        df["Linked Issues"]
        .fillna("")
        .astype(str)
        .apply(lambda s: [x.strip() for x in s.split(",") if x.strip()])
    )
    return df


def export_to_excel(
    scored_df: pd.DataFrame,
    testcases_df: pd.DataFrame
) -> bytes:
    """
    Bundles two DataFrames into a single .xlsx in memory:
      - Sheet 'Scored Issues'
      - Sheet 'Test Cases'
    Returns raw bytes of the workbook.
    """
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        scored_df.to_excel(writer, sheet_name="Scored Issues", index=False)
        testcases_df.to_excel(writer, sheet_name="Test Cases",   index=False)
    return buffer.getvalue()
