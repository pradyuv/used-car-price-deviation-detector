import pandas as pd
import numpy as np
from __future__ import annotations
from src.config import TARGET_COLS, CORE_FEATURES, AUX_FEATURES, DROP_COLUMNS


def validate_columns (df: pd.DataFrame) -> None:
    """
    Ensure failure if columns are missing. 
    Enforce used columns
    """
    required = set(TARGET_COLS + CORE_FEATURES + AUX_FEATURES)
    missing = sorted(required - set(df.columns))

    if missing:
        raise ValueError(
            "Missing required columns in input dataset: "
            + ", ".join(missing)
            + f"\nFound columns: {list(df.columns)}"
        )
    
def drop_declared_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop specified columns from a df.
    do NOT error, datasets vary and this is not critical
    """
    return df.drop(columns=DROP_COLUMNS, errors="ignore")


def preprocess (csv_path: str) -> pd.DataFrame:

    df = pd.read_csv(csv_path)
    validate_columns(df) 
    df = drop_declared_columns(df)

    keep = TARGET_COLS + CORE_FEATURES + AUX_FEATURES # Ensure we keep the columns we want
    df = df[keep].copy()

    return df

