import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import streamlit as st
from src.predict import predict_price_range

st.set_page_config(page_title="Nepal House Price Prediction", layout="centered")

st.title("🏠 Nepal House Price Prediction System")
st.caption("Linear Regression vs Random Forest (Auto-selected best model)")

location = st.text_input("Location", value="Budhanilkantha")
area = st.number_input("Area (sq ft)", min_value=1.0, value=1000.0, step=50.0)

if st.button("Predict Price"):
    result = predict_price_range(location, area)

    st.subheader("✅ Prediction Result")
    st.write(f"📍 **Resolved Location:** {result['town'].title()}, {result['city'].title()}")
    st.write(f"🤖 **Best Model Used:** {result['best_model']}")
    st.write(f"📊 **Confidence:** {result['confidence']}")

    st.success(
        f"Estimated Price Range: NPR {int(result['price_min']):,} - NPR {int(result['price_max']):,}"
    )

    st.info(f"**Explanation:** {result['explanation']}")