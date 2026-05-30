import json
import joblib
import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from src.preprocessing import load_and_clean
from src.feature_engineering import build_training_frame

def train_and_select(
    data_path="data/raw.csv",
    model_out="models/best_model.pkl",
    meta_out="models/metadata.json"
):
    df_raw, colmap = load_and_clean(data_path)
    df = build_training_frame(df_raw, colmap)

    cat_features = [
        "city",
        "road type",
        "face"
    ]

    num_features = [
        "area",
        "bedrooms",
        "bathrooms",
        "floors",
        "parking",
        "year",
        "road_width",
        "road_distance_score"
    ]

    target = "price"

    df = df.copy()

    for col in num_features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    for col in cat_features:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("unknown")

    df = df.dropna(subset=["price"])

    df["price"] = np.log1p(df["price"])

    X = df[cat_features + num_features]
    y = df[target].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.30,
        random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
            ("num", "passthrough", num_features),
        ]
    )

    lr = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", LinearRegression())
    ])

    rf = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=400,
            max_depth=None,
            random_state=42,
            n_jobs=-1
        ))
    ])

    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    lr_pred = lr.predict(X_test)
    rf_pred = rf.predict(X_test)

    lr_r2 = r2_score(y_test, lr_pred)
    rf_r2 = r2_score(y_test, rf_pred)

    lr_mse = mean_squared_error(y_test, lr_pred)
    rf_mse = mean_squared_error(y_test, rf_pred)

    # =========================
    # Select best model
    # =========================
    if rf_r2 >= lr_r2:
        best_model = rf
        best_name = "RandomForestRegressor"
        best_pred = rf_pred
        best_r2 = rf_r2
        best_mse = rf_mse
    else:
        best_model = lr
        best_name = "LinearRegression"
        best_pred = lr_pred
        best_r2 = lr_r2
        best_mse = lr_mse

    # =========================
    # Residual uncertainty
    # =========================
    residuals = y_test - best_pred
    sigma = float(np.std(residuals))

    # =========================
    # Save model
    # =========================
    os.makedirs("models", exist_ok=True)

    joblib.dump(best_model, model_out)

    metadata = {
        "best_model": best_name,
        "lr_r2": float(lr_r2),
        "rf_r2": float(rf_r2),
        "lr_mse": float(lr_mse),
        "rf_mse": float(rf_mse),
        "selected_r2": float(best_r2),
        "selected_mse": float(best_mse),
        "sigma_residual": sigma,
        "features": {
            "categorical": cat_features,
            "numerical": num_features
        },
        "note": "Price was log-transformed using log1p for better regression stability"
    }

    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print("\n===== TRAINING COMPLETE =====")
    print(f"Linear Regression R²: {lr_r2:.4f}")
    print(f"Random Forest R²: {rf_r2:.4f}")
    print(f"Best Model: {best_name}")
    print(f"Model saved to: {model_out}")

    return metadata