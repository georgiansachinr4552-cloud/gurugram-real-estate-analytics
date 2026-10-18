import seaborn as sns
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import ast
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# ==================================================
# PAGE CONFIGURATION
# ==================================================
st.set_page_config(
    page_title="Gurugram Real Estate Analytics",
    layout="wide"
)

st.title("Gurugram Real Estate Analytics")
st.write("Explore property prices and locations across Gurugram sectors.")

# ==================================================
# 1. LOAD DATASETS
# ==================================================
PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_DIR / "datasets" / "data_viz1.xls"
AMENITIES_PATH = PROJECT_DIR / "datasets" / "wordcloud_data.csv"


@st.cache_data
def load_data(file_path):
    # The .xls file contains CSV-formatted data.
    try:
        df = pd.read_csv(file_path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding="latin1")

    df.columns = df.columns.str.strip()
    return df


@st.cache_data
def load_amenities_data(file_path):
    df = pd.read_csv(file_path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    return df


if not DATA_PATH.exists():
    st.error(f"Main dataset not found: {DATA_PATH}")
    st.stop()

if not AMENITIES_PATH.exists():
    st.error(f"Amenities dataset not found: {AMENITIES_PATH}")
    st.stop()

new_df = load_data(str(DATA_PATH))


# ==================================================
# 2. VALIDATE AND CLEAN MAIN DATASET
# ==================================================
required_columns = [
    "sector",
    "price",
    "price_per_sqft",
    "built_up_area",
    "latitude",
    "longitude"
]

missing_columns = [
    col for col in required_columns
    if col not in new_df.columns
]

if missing_columns:
    st.error(f"Missing required columns: {missing_columns}")
    st.write("Available columns:", new_df.columns.tolist())
    st.stop()

# Find the property-type column without changing the source dataset.
possible_type_columns = [
    "property_type",
    "property type",
    "type",
    "property",
    "prop_type",
    "property_category",
    "property category"
]

type_col = next(
    (
        col for col in new_df.columns
        if col.strip().lower() in possible_type_columns
    ),
    None
)

# Detect the bedroom column.
bedroom_candidates = [
    "bedRoom",
    "bedroom",
    "bedrooms",
    "bedroom_count"
]

bedroom_col = next(
    (
        col for col in new_df.columns
        if col.strip().lower() in [
            name.lower() for name in bedroom_candidates
        ]
    ),
    None
)

new_df["sector"] = new_df["sector"].astype("string").str.strip()

numeric_columns = [
    "price",
    "price_per_sqft",
    "built_up_area",
    "latitude",
    "longitude"
]

if bedroom_col is not None:
    numeric_columns.append(bedroom_col)

for col in numeric_columns:
    new_df[col] = pd.to_numeric(new_df[col], errors="coerce")

new_df = new_df.dropna(subset=required_columns).copy()

new_df = new_df[
    (new_df["built_up_area"] > 0) &
    (new_df["price"] > 0) &
    (new_df["price_per_sqft"] > 0) &
    (new_df["latitude"].between(27, 29)) &
    (new_df["longitude"].between(76, 78)) &
    (new_df["sector"] != "")
].copy()

if new_df.empty:
    st.error("No valid property records remain after cleaning.")
    st.stop()


# ==================================================
# 3. SIDEBAR FILTERS
# ==================================================
st.sidebar.header("Filters")

# Keep an unfiltered copy for the amenities section.
# The amenities word clouds continue to use their own dataset.
base_df = new_df.copy()

# Property-type filter for the analytics and map.
if type_col is not None:
    raw_types = (
        base_df[type_col]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    # Map real dataset labels to friendly dropdown labels.
    type_mapping = {}

    for value in raw_types:
        normalized = value.lower()

        if normalized in ["flat", "flats", "apartment", "apartments"]:
            type_mapping[value] = "Flat"
        elif normalized in [
            "house", "houses", "independent house",
            "independent houses", "independent-house"
        ]:
            type_mapping[value] = "House"

    available_friendly_types = sorted(
        set(type_mapping.values())
    )

    # Offer Flat and House when these categories are present.
    dropdown_options = ["All"] + available_friendly_types

    selected_property_type = st.sidebar.selectbox(
        "Select Property Type",
        dropdown_options,
        key="analytics_property_type"
    )

    if selected_property_type != "All":
        matching_values = [
            original
            for original, friendly in type_mapping.items()
            if friendly == selected_property_type
        ]

        new_df = base_df[
            base_df[type_col]
            .astype(str)
            .str.strip()
            .isin(matching_values)
        ].copy()
    else:
        new_df = base_df.copy()

    if not available_friendly_types:
        st.sidebar.warning(
            f"Column '{type_col}' exists, but its values do not match "
            "the expected Flat/House labels."
        )

else:
    new_df = base_df.copy()
    st.sidebar.warning(
        "No property-type column was detected. "
        "The Flat/House dropdown needs the correct dataset column."
    )

if new_df.empty:
    st.warning("No properties match the selected property type.")
    st.stop()

# Sector-level analytics are recalculated after property filtering.
group_df = new_df.groupby("sector", as_index=False).agg(
    price=("price", "mean"),
    price_per_sqft=("price_per_sqft", "mean"),
    built_up_area=("built_up_area", "mean"),
    property_count=("sector", "size")
)

sector_options = sorted(group_df["sector"].unique().tolist())

selected_sectors = st.sidebar.multiselect(
    "Select Sectors",
    options=sector_options,
    default=sector_options
)

if not selected_sectors:
    st.warning("Please select at least one sector.")
    st.stop()

filtered_df = group_df[
    group_df["sector"].isin(selected_sectors)
].copy()

# Keep property-level records for map and scatter plot.
property_map_df = new_df[
    new_df["sector"].isin(selected_sectors)
].copy()

if property_map_df.empty:
    st.warning("No properties found for the selected sectors.")
    st.stop()


# ==================================================
# 4. SUMMARY METRICS
# ==================================================

# --------------------------------------------------
# 5. Summary metrics
# --------------------------------------------------

col1, col2, col3 = st.columns(3)

# Your dataset stores price in crore units.
# Convert the average to rupees for clear display.
average_price_crore = property_map_df["price"].mean()

col1.metric(
    "Sectors Selected",
    f"{filtered_df['sector'].nunique():,}"
)

col2.metric(
    "Average Property Price",
    f"₹{average_price_crore:.2f} Crore"
)

col3.metric(
    "Average Price per Sq Ft",
    f"₹{property_map_df['price_per_sqft'].mean():,.0f}"
)

st.caption(
    "Property price is displayed in crore rupees (Cr). "
    "Price per sq ft is displayed in rupees (₹)."
)

st.divider()





# ==================================================
# 5. PROPERTY-LEVEL GEO MAP
# ==================================================
st.subheader("Property Prices Across Gurugram")

map_fig = px.scatter_map(
    property_map_df,
    lat="latitude",
    lon="longitude",
    size="built_up_area",
    color="price_per_sqft",
    hover_name="sector",
    hover_data={
        "built_up_area": ":,.0f",
        "price": ":,.2f",
        "price_per_sqft": ":,.0f",
        "latitude": False,
        "longitude": False
    },
    size_max=22,
    color_continuous_scale="Viridis",
    zoom=10,
    center={"lat": 28.4595, "lon": 77.0266},
    map_style="open-street-map",
    height=650,
    labels={
        "built_up_area": "Built-up Area (sq ft)",
        "price": "Property Price (dataset units)",
        "price_per_sqft": "Price per Sq Ft (₹)"
    }
)

map_fig.update_layout(
    margin={"r": 0, "t": 5, "l": 0, "b": 0},
    coloraxis_colorbar_title="Price / Sq Ft"
)

st.plotly_chart(map_fig, use_container_width=True)

st.caption(
    f"Showing {len(property_map_df):,} individual properties. "
    "Larger dots indicate larger built-up areas; "
    "dot color indicates price per square foot."
)


# ==================================================
# 6. SECTOR-WISE PROPERTY ANALYSIS
# ==================================================
st.subheader("Sector-wise Property Analysis")

display_df = filtered_df.sort_values(
    "price_per_sqft",
    ascending=False
).copy()

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)


# ==================================================
# 7. PRICE COMPARISON BY SECTOR
# ==================================================
st.subheader("Average Price per Sq Ft by Sector")

chart_df = filtered_df.sort_values(
    "price_per_sqft",
    ascending=False
)

bar_fig = px.bar(
    chart_df,
    x="sector",
    y="price_per_sqft",
    color="price_per_sqft",
    color_continuous_scale="Viridis",
    labels={
        "sector": "Gurugram Sector",
        "price_per_sqft": "Average Price per Sq Ft (₹)"
    },
    title="Property Price Comparison Across Sectors"
)

st.plotly_chart(bar_fig, use_container_width=True)


# ==================================================
# 8. SCATTER PLOT: AREA VS PRICE
# ==================================================
st.divider()
st.subheader("Area Vs Price")

scatter_df = property_map_df.copy()

if bedroom_col is not None:
    scatter_df = scatter_df.dropna(subset=[bedroom_col])

    if not scatter_df.empty:
        scatter_fig = px.scatter(
            scatter_df,
            x="built_up_area",
            y="price",
            color=bedroom_col,
            hover_data={
                "sector": True,
                "built_up_area": ":,.0f",
                "price": ":,.2f",
                bedroom_col: True
            },
            title="Area Vs Price",
            labels={
                "built_up_area": "Built-up Area (sq ft)",
                "price": "Price (dataset units)",
                bedroom_col: "bedRoom"
            },
            color_continuous_scale="Blues"
        )

        scatter_fig.update_layout(height=600)

        st.plotly_chart(
            scatter_fig,
            use_container_width=True
        )
    else:
        st.warning("No rows with valid bedroom counts are available.")
else:
    st.warning(
        "The bedroom column was not found. "
        "Expected a column such as 'bedRoom'. "
        f"Available columns: {new_df.columns.tolist()}"
    )


# ==================================================
# 9. DOWNLOAD FILTERED PROPERTY DATA
# ==================================================
csv_data = property_map_df.to_csv(
    index=False
).encode("utf-8-sig")

st.download_button(
    label="Download Filtered Property Data (CSV)",
    data=csv_data,
    file_name="gurugram_filtered_properties.csv",
    mime="text/csv"
)


# ==================================================
# 10. AMENITIES WORD CLOUDS
# ==================================================
st.divider()
st.header("Gurugram Property Amenities")

amenity_df = load_amenities_data(str(AMENITIES_PATH))

amenity_required_columns = ["features", "sector"]

missing_amenity_columns = [
    col for col in amenity_required_columns
    if col not in amenity_df.columns
]

if missing_amenity_columns:
    st.error(
        "Missing columns in wordcloud_data.csv: "
        f"{missing_amenity_columns}"
    )
    st.stop()


def extract_amenities(series):
    amenities = []

    for item in series.dropna():
        try:
            parsed = ast.literal_eval(str(item))

            if isinstance(parsed, list):
                amenities.extend(
                    str(feature).strip()
                    for feature in parsed
                    if str(feature).strip()
                )

        except (ValueError, SyntaxError):
            continue

    return amenities


amenity_df["sector"] = (
    amenity_df["sector"]
    .astype("string")
    .str.strip()
)

valid_sectors = sorted(
    amenity_df.loc[
        amenity_df["sector"].notna()
        & (amenity_df["sector"] != ""),
        "sector"
    ].unique().tolist()
)

selected_sector = st.selectbox(
    "Select a sector",
    ["Complete Gurugram"] + valid_sectors,
    key="amenity_sector_dropdown"
)

# All Gurugram amenities.
all_amenities = extract_amenities(
    amenity_df["features"]
)

# Amenities for the selected sector.
if selected_sector == "Complete Gurugram":
    sector_amenities = all_amenities
else:
    sector_rows = amenity_df[
        amenity_df["sector"] == selected_sector
    ]

    sector_amenities = extract_amenities(
        sector_rows["features"]
    )


def show_amenity_cloud(amenities, title):
    st.subheader(title)

    if not amenities:
        st.info("No amenities available for this selection.")
        return

    frequencies = pd.Series(amenities).value_counts().to_dict()

    cloud = WordCloud(
        width=1200,
        height=650,
        background_color="white",
        collocations=False,
        max_font_size=100,
        min_font_size=8
    ).generate_from_frequencies(frequencies)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(cloud, interpolation="bilinear")
    ax.axis("off")

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


left_col, right_col = st.columns(2)

with left_col:
    show_amenity_cloud(
        sector_amenities,
        f"Amenities: {selected_sector}"
    )

with right_col:
    show_amenity_cloud(
        all_amenities,
        "Amenities: Complete Gurugram"
    )


# ==================================================
# 10. ADDITIONAL VISUALIZATIONS
# Existing map, sector analysis, and word clouds stay unchanged
# ==================================================

st.divider()
st.header("Additional Property Analysis")

# --------------------------------------------------
# Find bedroom and property-type columns automatically
# --------------------------------------------------
def find_column(dataframe, possible_names):
    column_lookup = {
        str(column).strip().lower().replace(" ", "_"): column
        for column in dataframe.columns
    }

    for name in possible_names:
        key = name.strip().lower().replace(" ", "_")
        if key in column_lookup:
            return column_lookup[key]

    return None


bedroom_col = find_column(
    new_df,
    ["bedRoom", "bedroom", "bedrooms", "bhk", "bed_room", "BHK"]
)

property_type_col = find_column(
    new_df,
    [
        "property_type",
        "property type",
        "type",
        "propertytype",
        "building_type"
    ]
)

# Make a separate dataframe so the existing data is not modified.
viz_df = new_df.copy()

if bedroom_col is not None:
    viz_df[bedroom_col] = pd.to_numeric(
        viz_df[bedroom_col], errors="coerce"
    )

# --------------------------------------------------
# 10A. BHK PIE CHART WITH SECTOR FILTER
# --------------------------------------------------
st.subheader("BHK Pie Chart")

if bedroom_col is None:
    st.warning(
        "BHK Pie Chart needs a bedroom-count column "
        "(for example, bedRoom or bhk). "
        "That column was not found in the current dataset."
    )
else:
    pie_sectors = sorted(
        viz_df["sector"].dropna().astype(str).unique().tolist()
    )

    selected_pie_sector = st.selectbox(
        "Select Sector for BHK Distribution",
        ["Overall"] + pie_sectors,
        key="bhk_pie_sector"
    )

    pie_df = viz_df.copy()

    if selected_pie_sector != "Overall":
        pie_df = pie_df[
            pie_df["sector"].astype(str) == selected_pie_sector
        ]

    pie_df = pie_df[
        pie_df[bedroom_col].notna()
        & (pie_df[bedroom_col] > 0)
    ]

    if pie_df.empty:
        st.info("No bedroom data is available for this selection.")
    else:
        bhk_counts = (
            pie_df[bedroom_col]
            .value_counts()
            .sort_index()
            .rename_axis("Bedrooms")
            .reset_index(name="Property Count")
        )

        bhk_counts["Bedrooms"] = (
            bhk_counts["Bedrooms"].map(
                lambda value: f"{value:g} BHK"
            )
        )

        pie_fig = px.pie(
            bhk_counts,
            names="Bedrooms",
            values="Property Count",
            title=f"BHK Distribution — {selected_pie_sector}",
            hole=0,
            color_discrete_sequence=px.colors.qualitative.Set2
        )

        pie_fig.update_traces(
            textposition="inside",
            textinfo="percent",
            hovertemplate=(
                "%{label}<br>"
                "Properties: %{value:,}<br>"
                "Share: %{percent}<extra></extra>"
            )
        )

        pie_fig.update_layout(height=500)
        st.plotly_chart(pie_fig, use_container_width=True)


# --------------------------------------------------
# 10B. SIDE-BY-SIDE BHK PRICE BOX PLOT
# --------------------------------------------------
st.divider()
st.subheader("Side-by-Side BHK Price Comparison")

if bedroom_col is None:
    st.warning(
        "The bedroom-count column is required for the BHK price box plot."
    )
else:
    box_df = viz_df[
        viz_df[bedroom_col].notna()
        & viz_df["price"].notna()
        & (viz_df[bedroom_col] > 0)
        & (viz_df["price"] > 0)
    ].copy()

    if box_df.empty:
        st.info("No valid bedroom and price records are available.")
    else:
        box_df["BHK"] = box_df[bedroom_col].map(
            lambda value: f"{value:g}"
        )

        bhk_order = sorted(
            box_df["BHK"].unique(),
            key=lambda value: float(value)
        )

        box_fig = px.box(
            box_df,
            x="BHK",
            y="price",
            category_orders={"BHK": bhk_order},
            points="outliers",
            title="Property Price Distribution by Bedroom Count",
            labels={
                "BHK": "bedRoom",
                "price": "Property Price (dataset units)"
            },
            color="BHK",
            color_discrete_sequence=px.colors.qualitative.Set2
        )

        box_fig.update_layout(
            showlegend=False,
            height=550
        )

        st.plotly_chart(box_fig, use_container_width=True)


# --------------------------------------------------
# 10C. SIDE-BY-SIDE HOUSE AND FLAT PRICE DISTRIBUTION
# --------------------------------------------------
st.divider()
st.subheader("Side-by-Side Price Distribution for Houses and Flats")

if property_type_col is None:
    st.warning(
        "The house-versus-flat distribution needs a property-type column "
        "(for example, property_type or type). "
        "That column was not found in the current dataset."
    )
else:
    dist_df = viz_df[
        viz_df["price"].notna() & (viz_df["price"] > 0)
    ].copy()

    dist_df["_property_type"] = (
        dist_df[property_type_col]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    # Keep only house and flat records.
    dist_df["_property_type"] = dist_df["_property_type"].replace({
        "houses": "house",
        "independent house": "house",
        "independent-house": "house",
        "flats": "flat",
        "apartment": "flat",
        "apartments": "flat"
    })

    dist_df = dist_df[
        dist_df["_property_type"].isin(["house", "flat"])
    ].copy()

    if dist_df.empty:
        st.info(
            "No house or flat records were found. "
            "Check the values in your property-type column."
        )
    else:
        available_types = set(dist_df["_property_type"].unique())

        if len(available_types) < 2:
            st.warning(
                "Both house and flat records are needed to compare "
                "their price distributions."
            )

        else:
            import matplotlib.pyplot as plt

            dist_fig, dist_ax = plt.subplots(figsize=(12, 6))

            sns.kdeplot(
                data=dist_df,
                x="price",
                hue="_property_type",
                hue_order=["house", "flat"],
                common_norm=False,
                fill=True,
                alpha=0.3,
                warn_singular=False,
                ax=dist_ax
            )

            dist_ax.set_title(
                "Side-by-Side Price Distribution: House vs Flat"
            )
            dist_ax.set_xlabel("Property Price (dataset units)")
            dist_ax.set_ylabel("Density")
            dist_ax.grid(alpha=0.2)

            st.pyplot(dist_fig, use_container_width=True)
            plt.close(dist_fig)
