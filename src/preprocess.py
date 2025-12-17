from __future__ import annotations
import pandas as pd
from config import (
    TARGET_COLS, CORE_FEATURES, AUX_FEATURES, DROP_COLUMNS,
    NUMERIC_COLS, CATEGORICAL_COLS
)
from datetime import datetime

required_cols = TARGET_COLS + CORE_FEATURES + AUX_FEATURES
required_nonnull = ["price", "brand", "model", "model_year", "milage", "accident"] # What must be present for this row to represent a coherent vehicle


current_year = datetime.now().year

"""
accident
"none reported" → 0
"at least 1 accident or damage reported" → 1
anything else → NA
Missing accident → drop row (core condition signal)
clean_title
"yes" → 1
blank / missing → "Unknown"
Encode as categorical, not boolean
"""

def coerce_numeric(df, cols):
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def normalize_strings(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        # string dtype avoids mixed types -> strip removes trailing spaces
        df[c] = df[c].astype("string").str.strip()
    return df

def normalize_accident(df: pd.DataFrame) -> pd.DataFrame:
    s = df["accident"].astype("string").str.strip().str.lower()

    none_mask = s.str.contains("none", na=False)
    accident_mask = s.str.contains("accident", na=False) | s.str.contains("damage", na=False)

    # Start as missing, then fill known cases
    out = pd.Series(pd.NA, index=df.index, dtype="Int64")
    out[none_mask] = 0
    out[accident_mask] = 1

    df["accident"] = out
    return df

def normalize_clean_title(df: pd.DataFrame) -> pd.DataFrame:
    s = df["clean_title"].astype("string").str.strip().str.lower()

    yes_mask = s.eq("yes")
    blank_mask = s.isna() | s.eq("")  # catches NaN and empty strings

    out = pd.Series(pd.NA, index=df.index, dtype="string")
    out[yes_mask] = "Yes"
    out[blank_mask] = "Unknown"
    out = out.fillna("Unknown") # fill remaining knows with unknown
    # any unexpected nonblank value -> Unknown, conservative estimate

    df["clean_title"] = out
    return df
    
def assert_required_columns(df: pd.DataFrame, required_cols: list[str]) -> None:
    missing = sorted(set(required_cols) - set(df.columns))
    if missing:
        raise ValueError (f"Missing required columns: {missing}")

def drop_rows_missing_required(df: pd.DataFrame, required_nonnull_cols: list[str]) -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=required_nonnull_cols)
    after = len(df)
    print(f"Dropped {before - after} rows due to missing required values.")
    return df

def apply_sanity_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with impossible or invalid values while preserving
    the natural market distribution.
    """
    before = len(df)

    current_year = datetime.now().year

    df = df[
        (df["price"] > 0) &
        (df["milage"] >= 0) &
        (df["model_year"] >= 1886) & # year of the first car, Benz Patent-Motorwagen :)
        (df["model_year"] <= current_year + 1)
    ]

    after = len(df)
    print(f"Dropped {before - after} rows due to sanity filters.")

    return df

def normalize_categorical_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in CATEGORICAL_COLS:
        df = df[col].astype("string").str.strip().str.lower()
        

def preprocess (csv_path: str) -> pd.DataFrame:

    df = pd.read_csv(csv_path)
    assert_required_columns(df, required_cols)

    df = df.drop(columns=DROP_COLUMNS, errors="ignore") #non essential columns, we don't consider it failure (hence errors ignore)

    # Normalize base types
    df = coerce_numeric(df, NUMERIC_COLS)
    df = normalize_strings(df, CATEGORICAL_COLS)

    # Normalize the special condition columns
    df = normalize_accident(df)
    df = normalize_clean_title(df)

    # Keep intended columns
    df = df[required_cols].copy()

    df = drop_rows_missing_required(df, required_nonnull)
    df = apply_sanity_filters(df)


    return df




