import re

DEFAULT_CITY = "kathmandu"
UNKNOWN = "unknown"

# Add more as you like
TOWN_TO_CITY = {
    "budhanilkantha": "kathmandu",
    "kirtipur": "kathmandu",
    "tokha": "kathmandu",
    "balkot": "bhaktapur",
    "suryabinayak": "bhaktapur",
    "thimi": "bhaktapur",
    "imadol": "lalitpur",
    "gwarko": "lalitpur",
}

def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def resolve_city_from_town(town: str) -> str:
    town_n = normalize_text(town)
    return TOWN_TO_CITY.get(town_n, DEFAULT_CITY)

def parse_location(location_text: str) -> tuple[str, str]:
    """
    Input: "Budhanilkantha" or "Balkot, Bhaktapur"
    Output: (town, city)
    Logic:
      - If "town, city" provided → use both
      - Else treat as town and infer city with mapping; default Kathmandu
    """
    loc = normalize_text(location_text)

    if not loc:
        return UNKNOWN, DEFAULT_CITY

    if "," in loc:
        parts = [p.strip() for p in loc.split(",") if p.strip()]
        if len(parts) >= 2:
            town = parts[0]
            city = parts[-1]  # last part as city
            return town, city

    # Only one word/name: treat as town, infer city
    town = loc
    city = resolve_city_from_town(town)
    return town, city