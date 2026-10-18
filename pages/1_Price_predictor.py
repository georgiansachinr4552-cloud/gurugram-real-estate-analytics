
import streamlit as st
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Gurugram Real Estate Price Predictor",
    page_icon="🏡",
    layout="wide"
)

st.title("🏡 Gurugram Real Estate Price Predictor")
st.caption(
    "Estimate property prices across sectors in Gurugram, Haryana."
)

st.divider()

# --------------------------------------------------
# 2. LOAD SAVED FILES
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

try:
    with open(BASE_DIR / "df.pkl", "rb") as file:
        df = pickle.load(file)

    with open(BASE_DIR / "pipeline.pkl", "rb") as file:
        pipeline = pickle.load(file)

except FileNotFoundError as e:
    st.error(f"Required file not found: {e.filename}")
    st.stop()

except Exception as e:
    st.error(f"Could not load the saved files: {e}")
    st.stop()

# --------------------------------------------------
# 3. INPUT FORM
# --------------------------------------------------

st.header("Enter Property Details")

col1, col2 = st.columns(2)

with col1:

    property_type = st.selectbox(
        "Property Type",
        sorted(df["property_type"].dropna().unique().tolist())
    )

    sector = st.selectbox(
        "Gurugram Sector",
        sorted(df["sector"].dropna().unique().tolist())
    )

    bedrooms = st.selectbox(
        "Number of Bedrooms",
        sorted(df["bedRoom"].dropna().unique().tolist())
    )

    bathroom = st.selectbox(
        "Number of Bathrooms",
        sorted(df["bathroom"].dropna().unique().tolist())
    )

    balcony = st.selectbox(
        "Number of Balconies",
        sorted(df["balcony"].dropna().unique().tolist())
    )

    property_age = st.selectbox(
        "Property Age",
        sorted(df["agePossession"].dropna().unique().tolist())
    )

with col2:

    built_up_area = st.number_input(
        "Built-up Area (sq. ft.)",
        min_value=1.0,
        value=1000.0,
        step=100.0
    )

    servant_room = st.selectbox(
        "Servant Room",
        [0.0, 1.0],
        format_func=lambda x: "Yes" if x == 1.0 else "No"
    )

    store_room = st.selectbox(
        "Store Room",
        [0.0, 1.0],
        format_func=lambda x: "Yes" if x == 1.0 else "No"
    )

    furnishing_type = st.selectbox(
        "Furnishing Type",
        sorted(df["furnishing_type"].dropna().unique().tolist())
    )

    luxury_category = st.selectbox(
        "Luxury Category",
        sorted(df["luxury_category"].dropna().unique().tolist())
    )

    floor_category = st.selectbox(
        "Floor Category",
        sorted(df["floor_category"].dropna().unique().tolist())
    )

st.divider()

# --------------------------------------------------
# 4. PREDICTION
# --------------------------------------------------

if st.button("Predict Property Price", type="primary"):

    # Create input DataFrame
    data = [[
        property_type,
        sector,
        bedrooms,
        bathroom,
        balcony,
        property_age,
        built_up_area,
        servant_room,
        store_room,
        furnishing_type,
        luxury_category,
        floor_category
    ]]

    columns = [
        "property_type",
        "sector",
        "bedRoom",
        "bathroom",
        "balcony",
        "agePossession",
        "built_up_area",
        "servant room",
        "store room",
        "furnishing_type",
        "luxury_category",
        "floor_category"
    ]

    one_df = pd.DataFrame(data, columns=columns)

    st.subheader("Your Selected Property")
    st.dataframe(one_df, use_container_width=True)

    try:
        # Get prediction from the trained pipeline
        prediction = float(pipeline.predict(one_df)[0])

        # Reverse log1p transformation if used during training
        # Assumes the model target was log1p(price in crores)
        base_price = np.expm1(prediction)

        if not np.isfinite(base_price) or base_price <= 0:
            st.error(
                "The model returned an invalid price. "
                "Check the target transformation and training units."
            )
        else:
            st.success("Gurugram property price predicted successfully!")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    label="Estimated Property Price",
                    value=f"₹ {base_price:.2f} Crore"
                )

            with col2:
                st.metric(
                    label="Estimated Price in Lakhs",
                    value=f"₹ {base_price * 100:.2f} Lakh"
                )

            st.caption(
                "This is a model-generated estimate, not a verified "
                "market valuation or guaranteed selling price."
            )

    except Exception as e:
        st.error(f"Prediction failed: {e}")

        st.info(
            "Check that the input column names, data types, "
            "and feature order match those used to train pipeline.pkl."
        )
