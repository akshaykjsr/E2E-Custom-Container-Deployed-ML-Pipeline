import os
import argparse
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import gcsfs
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")


def load_csv(uri: str) -> pd.DataFrame:
    """Load CSV from GCS or local path."""
    if uri.startswith("gs://"):
        # pandas can read directly via fsspec/gcsfs
        return pd.read_csv(uri)
    return pd.read_csv(uri)


def train_and_save(df: pd.DataFrame, out_dir: str, train_split: float,
                   learning_rate: float, n_estimators: int, random_state: int) -> None:
    train_size = int(len(df) * train_split)
    train_df, test_df = df.iloc[:train_size], df.iloc[train_size:]

    feature_cols = [
        c for c in df.columns
        if c not in ["Date", "ED_patient_volume", "log_ED_volume", "is_non_zero"]
    ]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols])
    X_test  = scaler.transform(test_df[feature_cols])
    y_train = train_df["log_ED_volume"]
    y_test  = test_df["log_ED_volume"]

    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=random_state,
    )
    print("Training XGBoost model …")
    model.fit(X_train, y_train)

    rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
    mae  = mean_absolute_error(y_test, model.predict(X_test))
    print(f"RMSE={rmse:.4f}   MAE={mae:.4f}")

    bundle = {"model": model, "scaler": scaler, "features": feature_cols}
    save_path = f"{out_dir.rstrip('/')}/model.joblib"

    if out_dir.startswith("gs://"):
        fs = gcsfs.GCSFileSystem()
        with fs.open(save_path, "wb") as f:
            joblib.dump(bundle, f)
    else:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, save_path)

    print(f"Saved artefacts to {save_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", required=True)
    p.add_argument("--model-dir", required=False)
    p.add_argument("--train-split", type=float, default=0.8)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--n-estimators", type=int, default=500)
    p.add_argument("--random-state", type=int, default=42)
    args = p.parse_args()

    df = load_csv(args.data_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")

    out_dir = os.environ.get("AIP_MODEL_DIR") or args.model_dir
    if not out_dir:
        raise RuntimeError("Provide --model-dir outside Vertex AI")

    train_and_save(
        df=df,
        out_dir=out_dir,
        train_split=args.train_split,
        learning_rate=args.learning_rate,
        n_estimators=args.n_estimators,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()

