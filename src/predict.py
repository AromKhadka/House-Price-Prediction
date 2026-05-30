import json
import joblib
import numpy as np
import pandas as pd

from src.location_resolver import parse_location

MODEL_PATH = "models/best_model.pkl"
META_PATH = "models/metadata.json"

def _load_assets():
    model = joblib.load(MODEL_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta

def _confidence_from_sigma(sigma: float, pred: float) -> str:
    # relative uncertainty
    rel = sigma / (abs(pred) + 1e-9)
    if rel < 0.10:
        return "High"
    if rel < 0.20:
        return "Medium"
    return "Low"

def predict_price_range(location_text: str, area: float) -> dict:
    model, meta = _load_assets()

    town, city = parse_location(location_text)

    bedrooms = meta["defaults"]["bedrooms"]
    bathrooms = meta["defaults"]["bathrooms"]

    # road_distance_score (1 best, 5 worst)
    # simple heuristic: valley cities slightly better road access
    valley = {"kathmandu", "lalitpur", "bhaktapur"}
    road_distance_score = 2 if city.lower() in valley else 4

    X = pd.DataFrame([{
        "city": city.lower(),
        "town": town.lower(),
        "area": float(area),
        "bedrooms": float(bedrooms),
        "bathrooms": float(bathrooms),
        "road_distance_score": float(road_distance_score),
    }])

    base_pred = float(model.predict(X)[0])
    sigma = float(meta["sigma_residual"])

    min_price = max(0.0, base_pred - sigma)
    max_price = max(min_price, base_pred + sigma)

    confidence = _confidence_from_sigma(sigma, base_pred)

    # short explanation
    reasons = []
    if city.lower() in valley:
        reasons.append("Kathmandu Valley locations generally have higher demand.")
    if area >= 2000:
        reasons.append("Larger area increases the estimated price.")
    if road_distance_score <= 2:
        reasons.append("Better road accessibility can increase value.")
    else:
        reasons.append("Lower road accessibility can reduce value.")

    explanation = " ".join(reasons[:3])

    return {
        "town": town,
        "city": city,
        "best_model": meta["best_model"],
        "price_min": min_price,
        "price_max": max_price,
        "confidence": confidence,
        "explanation": explanation,
    }