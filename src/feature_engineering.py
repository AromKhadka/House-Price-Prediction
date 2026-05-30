import numpy as np
import pandas as pd

VALLEY_CITIES = {"kathmandu", "lalitpur", "bhaktapur"}

def create_road_distance_score(df: pd.DataFrame, col_road_width: str | None, col_city: str) -> pd.Series:
    """
    If road width exists: larger width => closer/better access => smaller distance score.
    If not: heuristic based on city type + area (bigger area often more outside core).
    Score range: 1 (best/closest) to 5 (worst/farthest)
    """
    if col_road_width and col_road_width in df.columns:
        w = pd.to_numeric(df[col_road_width], errors="coerce").fillna(df[col_road_width].median())
        # Convert width into 1..5 (bigger width => better => lower distance score)
        # Use quantiles to avoid hardcoding
        q = w.rank(pct=True)
        score = 5 - np.ceil(q * 4)  # 1..5
        return score.clip(1, 5).astype(int)

    # Heuristic proxy
    city = df[col_city].astype(str).str.lower().fillna("unknown")
    area = pd.to_numeric(df["area"], errors="coerce").fillna(df["area"].median())

    base = np.where(city.isin(VALLEY_CITIES), 2.0, 3.5)
    # Larger area slightly increases distance score
    area_factor = (area - area.min()) / (area.max() - area.min() + 1e-9)
    score = base + area_factor * 2.0  # ~2..5.5
    return np.clip(np.round(score), 1, 5).astype(int)

def build_training_frame(df_raw: pd.DataFrame, colmap: dict) -> pd.DataFrame:
    """
    Outputs a normalized dataframe with canonical columns:
    price, area, bedrooms, bathrooms, town, city, road_distance_score
    """
    df = df_raw.copy()

    # Canonical numeric features
    df["price"] = df[colmap["price"]]
    df["area"] = df[colmap["area"]]
    df["bedrooms"] = df[colmap["bedrooms"]]
    df["bathrooms"] = df[colmap["bathrooms"]]

    # Location columns: if missing, create unknown
    if colmap.get("town") and colmap["town"] in df.columns:
        df["town"] = df[colmap["town"]].astype(str).str.lower()
    else:
        df["town"] = "unknown"

    if colmap.get("city") and colmap["city"] in df.columns:
        df["city"] = df[colmap["city"]].astype(str).str.lower()
    else:
        # If dataset has only one location-like column, we keep city unknown
        df["city"] = "unknown"

    # Road distance score
    road_col = colmap.get("road_width")
    df["road_distance_score"] = create_road_distance_score(df, road_col, "city")

    # Final cleanup
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["price", "area", "bedrooms", "bathrooms"])
    return df