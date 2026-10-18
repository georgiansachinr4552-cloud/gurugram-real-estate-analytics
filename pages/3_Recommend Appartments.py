
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Apartment Recommendation System", layout="wide")

st.title("🏠 Apartment Recommendation System")
st.write(
    "Find apartments near your preferred landmark and discover "
    "five similar properties."
)

# --------------------------------------------------
# 1. Load the existing dataset
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "datasets" / "data_viz1.xls"


@st.cache_data
def load_data(file_path):
    # The file may have an .xls extension but contain CSV text.
    try:
        df = pd.read_csv(file_path)
        if len(df.columns) == 1:
            raise ValueError("Not a CSV table")
    except Exception:
        df = pd.read_excel(file_path)

    df.columns = df.columns.astype(str).str.strip()

    # Use the existing property name column if available.
    if "property_name" in df.columns:
        df["property_name"] = df["property_name"].fillna("Unnamed property")
    elif "society" in df.columns:
        df["property_name"] = df["society"].fillna("Unnamed property")
    else:
        df["property_name"] = [
            f"Property {i + 1}" for i in range(len(df))
        ]

    # Convert coordinates to numeric values.
    for col in ["latitude", "longitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep only properties with usable coordinates.
    if {"latitude", "longitude"}.issubset(df.columns):
        df = df.dropna(subset=["latitude", "longitude"]).copy()
        df = df[
            df["latitude"].between(-90, 90)
            & df["longitude"].between(-180, 180)
        ].copy()
    else:
        raise ValueError(
            "The dataset must contain latitude and longitude columns."
        )

    df = df.reset_index(drop=True)
    return df


try:
    df = load_data(DATA_PATH)
except Exception as e:
    st.error(f"Could not load dataset: {e}")
    st.stop()

if df.empty:
    st.error("No properties with valid latitude and longitude were found.")
    st.stop()


# --------------------------------------------------
# 2. Calculate distance using latitude and longitude
# --------------------------------------------------

def distance_km(lat1, lon1, lat2, lon2):
    earth_radius = 6371.0

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )

    return 2 * earth_radius * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# --------------------------------------------------
# 3. Select a landmark and search radius
# --------------------------------------------------

st.header("1. Find Apartments Near a Landmark")

if "sector" in df.columns:
    location_column = "sector"
elif "society" in df.columns:
    location_column = "society"
else:
    location_column = "property_name"

locations = sorted(
    df[location_column].dropna().astype(str).unique().tolist()
)

if not locations:
    st.error("No locations are available in the dataset.")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    selected_location = st.selectbox(
        "Select a landmark / location",
        locations
    )

with col2:
    radius = st.slider(
        "Search radius (km)",
        min_value=1,
        max_value=30,
        value=5
    )

# Use the mean coordinates of properties in the selected location
# as an approximate landmark position.
landmark_properties = df[
    df[location_column].astype(str) == selected_location
]

landmark_lat = landmark_properties["latitude"].mean()
landmark_lon = landmark_properties["longitude"].mean()

df["distance_km"] = distance_km(
    landmark_lat,
    landmark_lon,
    df["latitude"].to_numpy(),
    df["longitude"].to_numpy()
)

nearby = df[df["distance_km"] <= radius].copy()
nearby = nearby.sort_values("distance_km")

st.metric("Apartments found within radius", len(nearby))

if not nearby.empty:
    display_columns = [
        c for c in [
            "property_name", "property_type", "society", "sector",
            "price", "price_per_sqft", "bedRoom", "bathroom",
            "built_up_area", "furnishing_type", "distance_km"
        ]
        if c in nearby.columns
    ]

    st.dataframe(
        nearby[display_columns].round(2),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Nearby Apartments Map")
    st.map(nearby[["latitude", "longitude"]])

else:
    st.warning(
        "No apartments were found within this radius. "
        "Try increasing the search radius."
    )


# --------------------------------------------------
# 4. Prepare property features for similarity
# --------------------------------------------------

st.header("2. Discover Similar Apartments")

# These features are used only when they exist in the dataset.
numeric_candidates = [
    "price", "price_per_sqft", "bedRoom", "bathroom",
    "balcony", "built_up_area", "floorNum", "luxury_score"
]

categorical_candidates = [
    "property_type", "furnishing_type", "agePossession",
    "sector", "society"
]

numeric_features = [
    c for c in numeric_candidates if c in df.columns
]
categorical_features = [
    c for c in categorical_candidates if c in df.columns
]

if not numeric_features and not categorical_features:
    st.error("No usable property features were found for recommendations.")
    st.stop()

feature_columns = numeric_features + categorical_features
features = df[feature_columns].copy()

for col in numeric_features:
    features[col] = pd.to_numeric(features[col], errors="coerce")

for col in categorical_features:
    features[col] = features[col].fillna("Unknown").astype(str)

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

transformers = []

if numeric_features:
    transformers.append(
        ("numeric", numeric_pipeline, numeric_features)
    )

if categorical_features:
    transformers.append(
        ("categorical", categorical_pipeline, categorical_features)
    )

preprocessor = ColumnTransformer(transformers=transformers)
feature_matrix = preprocessor.fit_transform(features)

# Similarity scores range from 0 to 1 for these non-negative
# encoded/scaled feature vectors only in some cases; display as a
# percentage for convenient comparison, not as a probability.
similarity_matrix = cosine_similarity(feature_matrix)


# --------------------------------------------------
# 5. Select a property and recommend five similar ones
# --------------------------------------------------

if nearby.empty:
    st.info("First find nearby apartments to select a property.")
else:
    # Use row indices so properties with duplicate society names
    # can still be selected individually.
    property_options = {
        f"{row['property_name']} | Row {idx}": idx
        for idx, row in nearby.iterrows()
    }

    selected_property_label = st.selectbox(
        "Select an apartment to get similar recommendations",
        list(property_options.keys())
    )

    selected_index = property_options[selected_property_label]

    st.subheader("Selected Apartment")

    chosen = df.loc[selected_index]

    details = [
        c for c in [
            "property_name", "property_type", "society", "sector",
            "price", "price_per_sqft", "bedRoom", "bathroom",
            "built_up_area", "furnishing_type", "luxury_score",
            "distance_km"
        ]
        if c in df.columns
    ]

    st.dataframe(
        chosen[details].to_frame("Details"),
        use_container_width=True
    )

    if st.button("Recommend 5 Similar Apartments", type="primary"):
        scores = similarity_matrix[selected_index].copy()

        # Do not recommend the selected property itself.
        scores[selected_index] = -1

        # Restrict recommendations to other nearby properties.
        nearby_indices = nearby.index.to_numpy()
        candidate_indices = [
            idx for idx in nearby_indices if idx != selected_index
        ]

        if not candidate_indices:
            st.warning(
                "There are no other apartments within this radius. "
                "Increase the radius to find similar properties."
            )
        else:
            ranked_indices = sorted(
                candidate_indices,
                key=lambda idx: scores[idx],
                reverse=True
            )[:5]

            recommendations = df.loc[ranked_indices].copy()
            recommendations["similarity_score"] = [
                scores[idx] * 100 for idx in ranked_indices
            ]

            st.subheader("Top Similar Apartments")

            result_columns = [
                c for c in [
                    "property_name", "property_type", "society", "sector",
                    "price", "price_per_sqft", "bedRoom", "bathroom",
                    "built_up_area", "furnishing_type", "distance_km",
                    "similarity_score"
                ]
                if c in recommendations.columns
            ]

            st.dataframe(
                recommendations[result_columns].round(2),
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "Recommendations are ranked by similarity of the available "
                "property features and restricted to the selected radius. "
                "The similarity score is a ranking score, not a probability."
            )


