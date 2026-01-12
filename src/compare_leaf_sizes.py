from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from train_model import build_model, build_preprocessor, load_clean_data, split_features_target

"""
This script compares RF stability across two conservative leaf sizes (5 vs 10)
and multiple random seeds. It trains both models on the same split per seed, computes
residuals (listed - expected), and maps them to labels using absolute and percentage
thresholds (underpriced, fair, overpriced). I report MAE/RMSE (mean±std), residual spread, and label flip rates to 
quantify how often deviation labels change under small model variations. 
The goal is not tuning for accuracy, but verifying label stability for the deviation detector.

"""

def residual_stats(residuals: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(residuals)),
        "std": float(np.std(residuals)),
        "p50": float(np.percentile(residuals, 50)),
        "p90": float(np.percentile(residuals, 90)),
        "p95": float(np.percentile(residuals, 95)),
        "p99": float(np.percentile(residuals, 99)),
    } #  builds a dict of mean, std, and percentiles (50/90/95/99) for quick comparison



def label_residuals(
    residuals: np.ndarray,
    listed_prices: np.ndarray,
    abs_threshold: float | None = None,
    pct_threshold: float | None = None,
) -> np.ndarray:
    if abs_threshold is None and pct_threshold is None:
        raise ValueError("Provide abs_threshold or pct_threshold.")
    if abs_threshold is not None and pct_threshold is not None:
        raise ValueError("Provide only one threshold type at a time.")

    if abs_threshold is not None:
        threshold = abs_threshold
    else:
        threshold = listed_prices * pct_threshold

    labels = np.where(
        residuals <= -threshold,
        "underpriced",
        np.where(residuals >= threshold, "overpriced", "fair"),
    )
    return labels


def compare_leaf_and_seed_sizes(
    clean_csv_path: Path | None = None,
    leaf_sizes: tuple[int, int] = (5, 10), 
    seeds: list[int] = [42, 123, 2024, 7, 99]
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if clean_csv_path is None:
        clean_csv_path = repo_root / "data" / "processed" / "used_cars_clean.csv"

    df = load_clean_data(Path(clean_csv_path))
    X, y = split_features_target(df)

    results = {leaf: {"mae": [], "rmse": [], "residuals": []} for leaf in leaf_sizes}  # dictionary of a dictionary of lists
    abs_thresholds = [2000, 5000, 10000]
    pct_thresholds = [0.05, 0.10] # absolute vs percentage thresholds, so static vs relative, we calculate deviation labels using these
    flip_abs = {thr: [] for thr in abs_thresholds}
    flip_pct = {pct: [] for pct in pct_thresholds}

    for seed in seeds:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=seed
        )
        seed_residuals = {}
        for leaf in leaf_sizes:
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", build_preprocessor()),
                    ("model", build_model(min_samples_leaf=leaf, seed=seed)),  # build a model for both leaf sizes
                ]
            )
            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_val)
            residuals = y_val.values - preds

            seed_residuals[leaf] = residuals
            results[leaf]["mae"].append(mean_absolute_error(y_val, preds))
            results[leaf]["rmse"].append(root_mean_squared_error(y_val, preds))
            results[leaf]["residuals"].append(residuals)

        for threshold in abs_thresholds:
            labels_a = label_residuals(
                seed_residuals[leaf_sizes[0]],
                y_val.values,
                abs_threshold=threshold,
            )
            labels_b = label_residuals(
                seed_residuals[leaf_sizes[1]],
                y_val.values,
                abs_threshold=threshold,
            )
            flip_abs[threshold].append(float(np.mean(labels_a != labels_b)))

        for pct in pct_thresholds:
            labels_a = label_residuals(
                seed_residuals[leaf_sizes[0]],
                y_val.values,
                pct_threshold=pct,
            )
            labels_b = label_residuals(
                seed_residuals[leaf_sizes[1]],
                y_val.values,
                pct_threshold=pct,
            )
            flip_pct[pct].append(float(np.mean(labels_a != labels_b)))

    print("=== Metric comparison (mean ± std) ===")
    for leaf, data in results.items():
        mae_mean, mae_std = np.mean(data["mae"]), np.std(data["mae"])
        rmse_mean, rmse_std = np.mean(data["rmse"]), np.std(data["rmse"])
        print(
            f"leaf={leaf} | MAE={mae_mean:,.0f}±{mae_std:,.0f} | RMSE={rmse_mean:,.0f}±{rmse_std:,.0f}"
        )

    print("\n=== Residual spread (pooled across seeds) ===")
    for leaf, data in results.items():
        pooled = np.concatenate(data["residuals"])
        stats = residual_stats(pooled)
        print(
            f"leaf={leaf} | mean={stats['mean']:,.0f} | std={stats['std']:,.0f} "
            f"| p50={stats['p50']:,.0f} | p90={stats['p90']:,.0f} "
            f"| p95={stats['p95']:,.0f} | p99={stats['p99']:,.0f}"
        )

    print("\n=== Label flip rates (absolute thresholds, mean ± std) ===") # depending on the different seed/leaf configs, how many listings' deviation labels change?
    for threshold, values in flip_abs.items():
        print(
            f"threshold=±{threshold:,.0f} | flip_rate={np.mean(values):.3f}±{np.std(values):.3f}"
        )

    print("\n=== Label flip rates (percentage thresholds, mean ± std) ===")
    for pct, values in flip_pct.items():
        print(
            f"threshold=±{pct*100:.0f}% | flip_rate={np.mean(values):.3f}±{np.std(values):.3f}"
        )


if __name__ == "__main__":
    compare_leaf_and_seed_sizes()
