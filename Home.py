

import streamlit as st

st.set_page_config(
    page_title="Gurugram Real Estate Analytics",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 Gurugram Real Estate Analytics")

st.write(
    "Welcome to your real estate analysis and prediction platform."
)

st.markdown("### Explore the application")

st.write("Use the pages in the left sidebar to explore:")

st.markdown("""
- **Price Predictor:** Predict property prices using your trained model.
- **Analysis:** Explore property data through charts and visualizations.
- **Recommender System:** Find properties matching your requirements.
- **Insights:** Understand market trends and important findings.
""")
