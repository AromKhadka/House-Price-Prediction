import json
import joblib
import numpy as np
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

def train_and_select(data_path="data/raw.csv", model_out="models/best_model.pkl", meta_out="models/metadata.json"):
    df_raw, colmap = load_and_clean(data_path)
    df = build_training_frame(df_raw, colmap)

    cat_features = ["city"]
    num_features = ["area", "bedrooms", "bathrooms"]
    target = "price"

    X = df[cat_features + num_features]
    y = df[target].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42
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
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        ))
    ])

    # Train
    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    # Evaluate
    lr_pred = lr.predict(X_test)
    rf_pred = rf.predict(X_test)

    lr_r2 = r2_score(y_test, lr_pred)
    rf_r2 = r2_score(y_test, rf_pred)

    lr_mse = mean_squared_error(y_test, lr_pred)
    rf_mse = mean_squared_error(y_test, rf_pred)

    # Select best
    if rf_r2 >= lr_r2:
        best = rf
        best_name = "RandomForestRegressor"
        best_pred = rf_pred
        best_r2 = rf_r2
        best_mse = rf_mse
    else:
        best = lr
        best_name = "LinearRegression"
        best_pred = lr_pred
        best_r2 = lr_r2
        best_mse = lr_mse

    # Residual sigma for range prediction (matches your report)
    residuals = y_test - best_pred
    sigma = float(np.std(residuals))

    # Save
    joblib.dump(best, model_out)

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
        "defaults": {
            "bedrooms": 2,
            "bathrooms": 1
        }
    }

    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)

    # Save model
    joblib.dump(best, model_out)

    # Save metadata safely
    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print("metadata.json saved successfully")

    print("Training complete")
    print(f"Linear Regression  R²={lr_r2:.4f}  MSE={lr_mse:.2f}")
    print(f"Random Forest     R²={rf_r2:.4f}  MSE={rf_mse:.2f}")
    print(f"Best Model: {best_name}")
    print(f"Saved: {model_out}")
    print(f"Saved: {meta_out}")

    return metadata