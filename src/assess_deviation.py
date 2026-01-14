from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from train_model import load_clean_data, split_features_target, build_preprocessor, build_model
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

repo_root = Path(__file__).resolve().parents[1]
bands = [0, 25000, 50000, 100000, 200000, float("inf")]
band_labels = ["<25k", "25-50k", "50-100k", "100-200k", "200k+"]


def residual_distribution(y_val: pd.Series, residuals: np.ndarray, out_dir: Path) -> None:
    bands = [0, 25000, 50000, 100000, 200000, float("inf")]
    labels = ["<25k", "25-50k", "50-100k", "100-200k", "200k+"]

    band = pd.cut(y_val.values, bins=bands, labels=labels, include_lowest=True)
    abs_resid = np.abs(residuals)

    summary = (
        pd.DataFrame({"band": band, "abs_resid": abs_resid, "resid": residuals})
        .groupby("band", observed=False)
        .agg(mean_abs_resid=("abs_resid", "mean"), count=("abs_resid", "size"))
    )

    # Bar chart: mean absolute residual by price band
    summary["mean_abs_resid"].plot(
        kind="bar",
        figsize=(8, 4),
        title="Mean |residual| by price band",
    )
    plt.ylabel("Mean |residual|")
    plt.tight_layout()
    plt.savefig(out_dir / "residuals_mean_by_price_band.png", dpi=150)
    plt.close()

    # Boxplot: residual distribution by price band
    pd.DataFrame({"band": band, "resid": residuals}).boxplot(
        by="band",
        column="resid",
        figsize=(8, 4),
    )
    plt.title("Residuals by price band")
    plt.suptitle("")
    plt.ylabel("Residual")
    plt.tight_layout()
    plt.savefig(out_dir / "residuals_box_by_price_band.png", dpi=150)
    plt.close()


def compute_band_thresholds(
    y_values: pd.Series,
    residuals: np.ndarray,
    bands: list[float],
    labels: list[str],
    quantile: float = 0.8,
) -> tuple[pd.Series, pd.Series, dict[str, float], float]:
    band = pd.cut(y_values, bins=bands, labels=labels, include_lowest=True)
    abs_resid = np.abs(residuals)
    band_thresholds = (
        pd.DataFrame({"band": band, "abs_resid": abs_resid})
        .groupby("band", observed=False)["abs_resid"]
        .quantile(quantile)
    )
    band_thresholds_dict = band_thresholds.to_dict()
    fallback = float(np.percentile(abs_resid, quantile * 100))
    thresholds = (
        pd.Series(band, index=y_values.index)
        .astype("string")
        .map(band_thresholds_dict)
        .fillna(fallback)
        .astype(float)
    )
    return thresholds, band, band_thresholds_dict, fallback


def label_from_thresholds(residuals: np.ndarray, thresholds: pd.Series) -> np.ndarray:
    return np.where(
        residuals <= -thresholds,
        "underpriced",
        np.where(residuals >= thresholds, "overpriced", "fair"),
    )


def assess_deviation(
    pipeline: Pipeline,
    clean_csv_path: Path | None = None,
    seed: int = 53,
    quantile: float = 0.8,
) -> None:
    if clean_csv_path is None:
        clean_csv_path = repo_root / "data" / "processed" / "used_cars_clean.csv"

    out_dir = repo_root / "reports" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_clean_data(clean_csv_path)
    X, y = split_features_target(df)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_val)
    residuals = y_val.values - preds

    residual_distribution(y_val, residuals, out_dir)

    thresholds, band, band_thresholds_dict, fallback = compute_band_thresholds(
        y_val, residuals, bands, band_labels, quantile=quantile
    )
    labels = label_from_thresholds(residuals, thresholds)

    print(f"Band thresholds ({int(quantile * 100)}th percentile |residual|):")
    for name in band_labels:
        value = band_thresholds_dict.get(name, fallback)
        print(f"  {name}: {value:,.0f}")

    counts = pd.Series(labels).value_counts().reindex(
        ["underpriced", "fair", "overpriced"], fill_value=0
    )
    percents = counts / counts.sum() * 100
    print("Band-threshold label distribution (validation):")
    for label, pct_val in percents.items():
        print(f"  {label}: {pct_val:.2f}%")

    val_labeled = X_val.copy()
    val_labeled["listed_price"] = y_val.values
    val_labeled["expected_price"] = preds
    val_labeled["residual"] = residuals
    val_labeled["label"] = labels
    val_labeled["price_band"] = band.astype("string").values
    val_labeled["threshold"] = thresholds.values

    print("\nSample listings per label (validation):")
    sample_cols = [
        "brand",
        "model",
        "model_year",
        "milage",
        "listed_price",
        "expected_price",
        "residual",
        "label",
        "price_band",
        "threshold",
    ]
    for label in ["underpriced", "fair", "overpriced"]:
        subset = val_labeled[val_labeled["label"] == label]
        sample = subset.sample(
            n=min(3, len(subset)),
            random_state=seed,
        )
        print(f"\n{label}:")
        print(sample[sample_cols].to_string(index=False))

    tables_dir = repo_root / "reports" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    val_labeled.to_csv(tables_dir / "validation_labeled.csv", index=False)
    sample_size = min(20, len(val_labeled))
    if sample_size:
        val_labeled.sample(n=sample_size, random_state=seed).to_csv(
            tables_dir / "validation_labeled_sample.csv", index=False
        )

    pipeline.fit(X, y)
    preds_full = pipeline.predict(X)
    residuals_full = y.values - preds_full
    thresholds_full, band_full, _, _ = compute_band_thresholds(
        y, residuals_full, bands, band_labels, quantile=quantile
    )
    labels_full = label_from_thresholds(residuals_full, thresholds_full)

    full_labeled = X.copy()
    full_labeled["listed_price"] = y.values
    full_labeled["expected_price"] = preds_full
    full_labeled["residual"] = residuals_full
    full_labeled["label"] = labels_full
    full_labeled["price_band"] = band_full.astype("string").values
    full_labeled["threshold"] = thresholds_full.values

    full_out = repo_root / "data" / "processed" / "used_cars_labeled.csv"
    full_labeled.to_csv(full_out, index=False)


if __name__ == "__main__":
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", build_model(min_samples_leaf=5, seed=53)),
        ]
    )
    assess_deviation(pipeline)
    
    


  



    
