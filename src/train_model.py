from __future__ import annotations

from pathlib import Path

import pandas as pd
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


def build_model() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=200, # enough trees to stabilize averages without being too heavy
        random_state=42, # RANDOM SEED FOR REPRODUCIBILITY
        min_samples_leaf=10, #each leaf must sample at least 10 samples, reduce variance
        min_samples_split=20, #stabilizes trees, prevents too many splits
        max_features="sqrt",
        n_jobs=-1,
    )


def train_model(clean_csv_path: Path | None = None) -> Pipeline:
    repo_root = Path(__file__).resolve().parents[1]
    if clean_csv_path is None:
        clean_csv_path = repo_root / "data" / "processed" / "used_cars_clean.csv"

    clean_csv_path = Path(clean_csv_path)
    if not clean_csv_path.exists():
        raise FileNotFoundError(f"Missing cleaned data at {clean_csv_path}")

    df = load_clean_data(clean_csv_path)
    X, y = split_features_target(df)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
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

    return pipeline


if __name__ == "__main__":
    train_model()
