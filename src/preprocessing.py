import pandas as pd
import re

# Common land conversion used in Nepal (approx). Good enough for ML feature scale.
AANA_TO_SQFT = 342.25  # 1 aana ≈ 342.25 sq.ft

def parse_aana_to_sqft(value) -> float | None:
    """
    Converts formats like:
      '1-0-0-0 Aana'  -> 1 aana
      '0-21-0-0 Aana' -> 21 aana
      '0-10-1-0 Aana' -> 10 aana 1 paisa
    We treat: aana-paisa-daam (common) approximately:
      1 aana = 4 paisa, 1 paisa = 4 daam
    If parsing fails, return None.
    """
    if pd.isna(value):
        return None
    s = str(value).strip().lower()

    # If already numeric-like, try directly
    if re.fullmatch(r"[\d.]+", s):
        return float(s)

    # Extract the "x-x-x-x" part
    m = re.search(r"(\d+)\s*-\s*(\d+)\s*-\s*(\d+)\s*-\s*(\d+)", s)
    if not m:
        # Sometimes just "10 aana" style
        m2 = re.search(r"(\d+(?:\.\d+)?)\s*aana", s)
        if m2:
            aana = float(m2.group(1))
            return aana * AANA_TO_SQFT
        return None

    ropani = int(m.group(1))   # often 0 in your data
    aana   = int(m.group(2))
    paisa  = int(m.group(3))
    daam   = int(m.group(4))

    # Convert everything to aana (approx)
    total_aana = ropani * 16 + aana + (paisa / 4.0) + (daam / 16.0)
    return total_aana * AANA_TO_SQFT

def parse_road_width_feet(value) -> float | None:
    """
    Parses '20 Feet' or '20 Feet / Blacktopped' etc.
    """
    if pd.isna(value):
        return None
    s = str(value).lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*feet", s)
    if m:
        return float(m.group(1))
    # fallback: any number
    m2 = re.search(r"(\d+(?:\.\d+)?)", s)
    if m2:
        return float(m2.group(1))
    return None

def extract_town_from_address(address) -> str:
    """
    Address examples:
      'Budhanikantha, Budhanilkantha, Kathmandu'
      'Dhapasi, Dhapasi, Kathmandu'
    We'll take the first chunk as town/area name.
    """
    if pd.isna(address):
        return "unknown"
    parts = [p.strip().lower() for p in str(address).split(",") if p.strip()]
    return parts[0] if parts else "unknown"

def load_and_clean(path: str):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()

    # --- Create canonical columns expected by your pipeline ---
    # price is already numeric in your sample, but keep safe conversion
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    # area is in aana format -> convert to sqft
    df["area"] = df["area"].apply(parse_aana_to_sqft)

    # bedrooms/bathrooms numeric
    df["bedrooms"] = pd.to_numeric(df.get("bedroom"), errors="coerce")
    df["bathrooms"] = pd.to_numeric(df.get("bathroom"), errors="coerce")

    # city exists
    df["city"] = df.get("city").astype(str).str.strip().str.lower()

    # town derived from address
    if "address" in df.columns:
        df["town"] = df["address"].apply(extract_town_from_address)
    else:
        df["town"] = "unknown"

    # road width: prefer 'road width' column, else parse from 'road'
    road_width_col = "road width" if "road width" in df.columns else None
    if road_width_col:
        df["road_width"] = df[road_width_col].apply(parse_road_width_feet)
    elif "road" in df.columns:
        df["road_width"] = df["road"].apply(parse_road_width_feet)
    else:
        df["road_width"] = None

    # drop only essential missing
    df = df.dropna(subset=["price", "area"])

    # fill missing beds/baths with medians
    df["bedrooms"] = df["bedrooms"].fillna(df["bedrooms"].median() if df["bedrooms"].notna().any() else 2)
    df["bathrooms"] = df["bathrooms"].fillna(df["bathrooms"].median() if df["bathrooms"].notna().any() else 1)

    # if road_width missing, fill median
    if df["road_width"].notna().any():
        df["road_width"] = df["road_width"].fillna(df["road_width"].median())
    else:
        df["road_width"] = 20  # reasonable default in feet

    colmap = {
        "price": "price",
        "area": "area",
        "bedrooms": "bedrooms",
        "bathrooms": "bathrooms",
        "city": "city",
        "town": "town",
        "road_width": "road_width",
    }

    print("Cleaned dataset shape:", df.shape)
    return df, colmap