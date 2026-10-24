"""Shared helpers: locate files, load saved models, or (re)train them from the CSVs."""
from pathlib import Path

import joblib
import pandas as pd

BASE = Path(__file__).resolve().parent
SEARCH_DIRS = ["", "models", "models/models", "data"]

CROP_CSV = "Crop_recommendation.csv"
YIELD_CSV = "Crop Yiled with Soil and Weather.csv"


def find_file(name):
    """Look for a file in the project root and the usual sub-folders."""
    for d in SEARCH_DIRS:
        p = BASE / d / name
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find '{name}' anywhere in the project.")


def train_all(save=False, verbose=False):
    """Train both models from the CSV files. Returns (rf, le, xgb, fert_mean, metrics)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from xgboost import XGBRegressor

    # --- crop recommendation ---
    df_crop = pd.read_csv(find_file(CROP_CSV))
    X_c = df_crop.drop(columns=["label"])
    le = LabelEncoder()
    y_c = le.fit_transform(df_crop["label"])
    Xtr, Xte, ytr, yte = train_test_split(
        X_c, y_c, test_size=0.2, random_state=42, stratify=y_c
    )
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(Xtr, ytr)
    acc = accuracy_score(yte, rf.predict(Xte))

    # --- yield prediction ---
    df_y = pd.read_csv(find_file(YIELD_CSV)).rename(columns={"yeild": "yield"})
    X_y = df_y.drop(columns=["yield"])
    y_y = df_y["yield"]
    Xtr, Xte, ytr, yte = train_test_split(X_y, y_y, test_size=0.2, random_state=42)
    xgb = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6,
                       random_state=42, n_jobs=-1)
    xgb.fit(Xtr, ytr)
    r2 = r2_score(yte, xgb.predict(Xte))
    fert_mean = float(X_y["Fertilizer"].mean())

    if verbose:
        print(f"RF accuracy: {acc:.4f}")
        print(f"XGB R2: {r2:.4f}")

    if save:
        out = BASE / "models"
        out.mkdir(exist_ok=True)
        joblib.dump(rf, out / "rf_model.joblib")
        joblib.dump(le, out / "label_encoder.joblib")
        joblib.dump(xgb, out / "xgb_model.joblib")
        joblib.dump(fert_mean, out / "fert_mean.joblib")
        if verbose:
            print("Saved models to ./models")

    return rf, le, xgb, fert_mean, {"accuracy": acc, "r2": r2}


def load_or_train():
    """Load the saved models; if that fails (or versions differ), train fresh ones.

    Returns (rf, le, xgb, fert_mean, retrained_flag).
    """
    import warnings
    from sklearn.exceptions import InconsistentVersionWarning

    try:
        with warnings.catch_warnings():
            # a scikit-learn version mismatch can silently give wrong results,
            # so treat it as a failure and retrain on this server instead
            warnings.simplefilter("error", InconsistentVersionWarning)
            rf = joblib.load(find_file("rf_model.joblib"))
            le = joblib.load(find_file("label_encoder.joblib"))
            xgb = joblib.load(find_file("xgb_model.joblib"))
            fert_mean = float(joblib.load(find_file("fert_mean.joblib")))
        return rf, le, xgb, fert_mean, False
    except Exception:
        rf, le, xgb, fert_mean, _ = train_all(save=False)
        return rf, le, xgb, fert_mean, True
