from __future__ import annotations
from pathlib import Path
from preprocess import preprocess
from train_model import train_model
import joblib


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    raw_path = repo_root / "data" / "raw" / "used_cars.csv"
    out_dir = repo_root / "data" / "processed"
    out_path = out_dir / "used_cars_clean.csv"

    out_dir.mkdir(parents=True, exist_ok=True)
    df = preprocess(str(raw_path))
    df.to_csv(out_path, index=False)
    print(f"Wrote cleaned data to {out_path}")

    pipeline = train_model(out_path)
    joblib.dump(pipeline, "models/rf_expected_price.joblib")
    rf = pipeline.named_steps["model"]
    depths = [t.tree_.max_depth for t in rf.estimators_]
    print("min depth:", min(depths))
    print("avg depth:", sum(depths) / len(depths))
    print("max depth:", max(depths))
        



if __name__ == "__main__":
    main()
