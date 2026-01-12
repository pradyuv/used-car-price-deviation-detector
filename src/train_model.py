from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from config import CORE_FEATURES, AUX_FEATURES, TARGET_COLS


def load_clean_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    feature_cols = CORE_FEATURES + AUX_FEATURES
    target_col = TARGET_COLS[0]
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """
    Model‑ready transformation
    Clean dataframe → numeric matrix for the RF (impute missing values + one‑hot
    encode categoricals)
    Fit only on the training split to avoid leakage, and it’s bundled with the
    model in the pipeline so inference uses the exact same transformations
    Different from preprocess.py where we clean the data and derive features
    """
    numeric_features = ["model_year", "milage", "accident", "engine_displacement_liters"]
    categorical_features = [
        col for col in (CORE_FEATURES + AUX_FEATURES) if col not in numeric_features
    ]

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))] # impute missing values with median of feature (numerical)
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")), #impute missing categorical features to unknown
            ("onehot", OneHotEncoder(handle_unknown="ignore")), #prevent errors when new categories show up at inference, safe and predictable
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def build_model(
    min_samples_leaf: int = 5,
    min_samples_split: int = 20,
    seed: int = 42
) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=200,  # enough trees to stabilize averages without being too heavy
        random_state=seed,  # random seed for reproducibility
        min_samples_leaf=min_samples_leaf,  # larger values reduce variance
        min_samples_split=min_samples_split,  # stabilizes trees, prevents too many splits
        max_features="sqrt",
        n_jobs=-1,
    )


def summarize_price_distribution(df: pd.DataFrame) -> None:
    target_col = TARGET_COLS[0]
    percentiles = [0.5, 0.9, 0.95, 0.99]
    summary = df[target_col].describe(percentiles=percentiles)
    print("Price distribution (cleaned data):")
    print(summary.to_string())


def plot_residuals(y_true: pd.Series, y_pred: np.ndarray, out_dir: Path) -> None:
    residuals = y_true - y_pred
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.hist(residuals, bins=60)
    plt.title("Residual distribution")
    plt.xlabel("Residual (price - expected)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_dir / "residuals_hist.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.scatter(y_true, residuals, alpha=0.3, s=10)
    plt.title("Residuals vs listed price")
    plt.xlabel("Listed price")
    plt.ylabel("Residual")
    plt.tight_layout()
    plt.savefig(out_dir / "residuals_vs_price.png", dpi=150)
    plt.close()


def train_model(clean_csv_path: Path | None = None, test_size : int = 0.2, random_state : int = 42 ) -> Pipeline:
    repo_root = Path(__file__).resolve().parents[1]
    if clean_csv_path is None:
        clean_csv_path = repo_root / "data" / "processed" / "used_cars_clean.csv"

    clean_csv_path = Path(clean_csv_path)
    if not clean_csv_path.exists():
        raise FileNotFoundError(f"Missing cleaned data at {clean_csv_path}")

    df = load_clean_data(clean_csv_path)
    summarize_price_distribution(df)
    X, y = split_features_target(df)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    ) # standard 80/20 split

    """
    - Ensures fit time and predict time preprocessing are identical....
    - Prevents data leakage (imputers/encoders learn only from training split)
    - Makes the model reproducible and deployable as a SINGLE object
    - Plays well with train/val splits or future cross‑validation
    """
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", build_model()),
        ]
    ) 
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_val)
    mae = mean_absolute_error(y_val, preds)
    rmse = mean_squared_error(y_val, preds, squared=False)
    print(f"Validation MAE: {mae:,.0f}")
    print(f"Validation RMSE: {rmse:,.0f}")

    plot_residuals(y_val, preds, repo_root / "reports" / "figures")

    return pipeline


if __name__ == "__main__":
    train_model()
