from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from train_model import load_clean_data, split_features_target, build_preprocessor, build_model
from compare_leaf_sizes import label_residuals
from sklearn.pipeline import Pipeline

repo_root = Path(__file__).resolve().parents[1]





if __name__ == "__main__":
    clean_csv_path = repo_root / "data" / "processed" / "used_cars_clean.csv"
    df = load_clean_data(clean_csv_path)
    X, y = split_features_target(df)
    X_train, X_val, y_train, y_val = train_test_split(
      X, y, test_size=0.2, random_state=53
    )

    pipeline = Pipeline(
      steps=[
          ("preprocessor", build_preprocessor()),
          ("model", build_model(min_samples_leaf=5, seed=53)),
      ]
    )
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_val)
    residuals = y_val.values - preds

    for pct in [0.10, 0.15, 0.20]:
      labels = label_residuals(residuals, y_val.values, pct_threshold=pct)
      counts = pd.Series(labels).value_counts().reindex(
          ["underpriced", "fair", "overpriced"], fill_value=0
      )
      percents = counts / counts.sum() * 100
      print(f"{int(pct*100)}% threshold:")
      for label, pct_val in percents.items():
          print(f"  {label}: {pct_val:.2f}%")


    """
    FIRST RUN: 
    FOR ABS 2000:
    underpriced    67.820513
    overpriced     27.435897
    fair            4.743590
    Name: count, dtype: float64
    """



    







