from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def plot_numeric_histograms(df: pd.DataFrame, out_dir: Path) -> None:
    numeric_cols = ["price", "milage", "model_year", "engine_displacement_liters"]
    for col in numeric_cols:
        if col not in df.columns:
            continue
        data = df[col].dropna()
        bins = 40
        title = f"Distribution of {col}"
        if col == "price":
            data = data[data <= 1_000_000]
            bins = 80
            title = "Distribution of price (0-1M)"
        plt.figure(figsize=(8, 5))
        data.hist(bins=bins)
        plt.title(title)
        plt.xlabel(col)
        plt.ylabel("Count")
        if col == "price":
            plt.xlim(0, 1_000_000)
        plt.tight_layout()
        plt.savefig(out_dir / f"hist_{col}.png", dpi=150)
        plt.close()


def plot_categorical_bars(df: pd.DataFrame, out_dir: Path) -> None:
    categorical_cols = ["brand", "model", "fuel_type", "transmission_type", "clean_title", "accident"]
    for col in categorical_cols:
        if col not in df.columns:
            continue
        counts = df[col].value_counts(dropna=False)
        
        # Keep high-cardinality plots readable
        if col in {"brand", "model"}:
            counts = counts.head(20)
        plt.figure(figsize=(10, 6))
        counts.plot(kind="bar")
        plt.title(f"Top categories for {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(out_dir / f"bar_{col}.png", dpi=150)
        plt.close()


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data_path = repo_root / "data" / "processed" / "used_cars_clean.csv"
    out_dir = repo_root / "reports" / "figures"

    if not data_path.exists():
        raise FileNotFoundError(f"Missing cleaned data at {data_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_path)

    plot_numeric_histograms(df, out_dir)
    plot_categorical_bars(df, out_dir)
    print(f"Wrote EDA plots to {out_dir}")


if __name__ == "__main__":
    main()
