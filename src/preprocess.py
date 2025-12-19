from __future__ import annotations
from datetime import datetime
import pandas as pd
import numpy as np
from config import (
    TARGET_COLS, CORE_FEATURES, AUX_FEATURES, DROP_COLUMNS,
    NUMERIC_COLS, CATEGORICAL_COLS
)

required_cols = TARGET_COLS + CORE_FEATURES + AUX_FEATURES
raw_aux_cols = ["fuel_type", "engine", "transmission"]
required_raw_cols = TARGET_COLS + CORE_FEATURES + raw_aux_cols
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
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def sanitize_price_milage(df: pd.DataFrame) -> pd.DataFrame:
    # Strip currency symbols, commas, and units before numeric coercion
    # coercing numeric columsn before was just coercing these to NaN, causing the rows with these columns missing to be dropped
    df["price"] = df["price"].astype("string").str.replace(r"[^\d.]", "", regex=True)
    df["milage"] = df["milage"].astype("string").str.replace(r"[^\d.]", "", regex=True)
    return df

def normalize_strings(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        # string dtype avoids mixed types -> strip removes trailing spaces and lower too
        df[c] = df[c].astype("string").str.strip().str.lower()
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
    out[yes_mask] = "yes"
    out[blank_mask] = "unknown"
    out = out.fillna("unknown") # fill remaining knows with unknown
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

def derive_engine_displacement_liters(df: pd.DataFrame) -> pd.DataFrame:
    s = df["engine"].astype("string").str.strip().str.lower()

    #extract patterns like "2.0L" or "2.0 liter", we are trying to simplify the engine feature
    liters = s.str.extract(r"(\d+(?:\.\d+)?)\s*l\b", expand=False)
    liters_alt = s.str.extract(r"(\d+(?:\.\d+)?)\s*liter", expand=False)
    out = liters.combine_first(liters_alt)
    out = pd.to_numeric(out, errors="coerce")

    electric_mask = s.str.contains("electric", na=False)
    out = out.where(~electric_mask, other=np.nan)

    df["engine_displacement_liters"] = out
    return df

def derive_transmission_type(df: pd.DataFrame) -> pd.DataFrame:
    s = df["transmission"].astype("string").str.strip().str.lower()

    auto_mask = s.str.contains(r"\bauto\b|\bautomatic\b|\ba/t\b|\bcvt\b", na=False)
    manual_mask = s.str.contains(r"\bmanual\b|\bm/t\b", na=False)

    out = pd.Series("unknown", index=df.index, dtype="string")
    out[auto_mask] = "automatic"
    out[manual_mask] = "manual"

    df["transmission_type"] = out
    return df

def preprocess (csv_path: str) -> pd.DataFrame:

    df = pd.read_csv(csv_path)
    assert_required_columns(df, required_raw_cols)

    df = df.drop(columns=DROP_COLUMNS, errors="ignore") #non essential columns, we don't consider it failure (hence errors ignore)

    # Normalize base types
    df = sanitize_price_milage(df)
    df = coerce_numeric(df, ["price", "milage", "model_year"])
    df = normalize_strings(df, ["brand", "model", "fuel_type", "clean_title", "engine", "transmission"])

    # Normalize the special condition columns
    df = normalize_accident(df)
    df = normalize_clean_title(df)

    # Derive clean features from raw text fields
    df = derive_engine_displacement_liters(df)
    df = derive_transmission_type(df)

    df = df.drop(columns=["engine", "transmission"], errors="ignore")
    df = coerce_numeric(df, ["engine_displacement_liters"])

    # Keep intended columns
    df = df[required_cols].copy()

    df = drop_rows_missing_required(df, required_nonnull)
    df = apply_sanity_filters(df)


    return df



df = pd.read_csv("data/raw/used_cars.csv")
# ===== Diagnostic: engine & transmission pattern coverage =====

"""
I used this as a view into the frequency of extractable features present 
ex. The occurent of L engine displacement, cylinder config etc.
My findings were:
ENGINE SIGNAL COVERAGE
  Displacement (X.XL):        90.6%
  Cylinder config (I4/V6):    34.3%
  Any engine signal present:  90.7%

TRANSMISSION SIGNAL COVERAGE
  Automatic detected:         61.7%
  Manual detected:            9.3%
  Unknown / ambiguous:        29.0%

SAMPLE ENGINE VALUES:
['300.0hp 3.7l v6 cylinder engine flex fuel capability', '3.8l v6 24v gdi dohc', '3.5 liter dohc', '354.0hp 3.5l v6 cylinder engine gas/electric hybrid', '2.0l i4 16v gdi dohc turbo', '2.4 liter', '292.0hp 2.0l 4 cylinder engine gasoline fuel', '282.0hp 4.4l 8 cylinder engine gasoline fuel', '311.0hp 3.5l v6 cylinder engine gasoline fuel', '534.0hp electric motor electric fuel system']

SAMPLE TRANSMISSION VALUES:
['6-speed a/t', '8-speed automatic', 'automatic', '7-speed a/t', '8-speed automatic', 'f', '6-speed a/t', 'a/t', '6-speed a/t', 'a/t']
"""

engine_series = df["engine"].astype("string").str.lower()
trans_series = df["transmission"].astype("string").str.lower()

# --- Engine patterns ---
has_displacement = engine_series.str.contains(r"\b\d+(\.\d+)?l\b", na=False)
has_cylinders = engine_series.str.contains(r"\b[vi]\d\b", na=False)
has_any_engine_signal = has_displacement | has_cylinders

print("ENGINE SIGNAL COVERAGE")
print(f"  Displacement (X.XL):        {has_displacement.mean() * 100:.1f}%")
print(f"  Cylinder config (I4/V6):    {has_cylinders.mean() * 100:.1f}%")
print(f"  Any engine signal present:  {has_any_engine_signal.mean() * 100:.1f}%")

# --- Transmission patterns ---
auto_mask = trans_series.str.contains(r"\bauto\b|\ba/t\b", na=False)
manual_mask = trans_series.str.contains(r"\bmanual\b|\bm/t\b", na=False)
known_trans = auto_mask | manual_mask

print("\nTRANSMISSION SIGNAL COVERAGE")
print(f"  Automatic detected:         {auto_mask.mean() * 100:.1f}%")
print(f"  Manual detected:            {manual_mask.mean() * 100:.1f}%")
print(f"  Unknown / ambiguous:        {(~known_trans).mean() * 100:.1f}%")

# --- Optional sanity samples ---
print("\nSAMPLE ENGINE VALUES:")
print(engine_series.dropna().head(10).tolist())

print("\nSAMPLE TRANSMISSION VALUES:")
print(trans_series.dropna().head(10).tolist())
