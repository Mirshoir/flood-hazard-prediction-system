import streamlit as st
import os
import pandas as pd
import geopandas as gpd

from utils import data_loader, map_utils, model_trainer, evaluator, visualizer

# Streamlit app config
st.set_page_config(page_title="Flood Hazard Prediction System", layout="wide")
st.title("🌊 Flood Hazard Prediction System")

# Sidebar menu
menu = st.sidebar.radio("Navigation", [
    "1️⃣ Load Data",
    "2️⃣ Display Maps",
    "3️⃣ Variable Selection",
    "4️⃣ Train Model",
    "5️⃣ Visualize Prediction Map"
])

# Session state for storing data
if 'df' not in st.session_state:
    st.session_state.df = None
if 'gdf' not in st.session_state:
    st.session_state.gdf = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None

# 1️⃣ Load Data
if menu == "1️⃣ Load Data":
    st.subheader("Upload Your Data")

    uploaded_csv = st.file_uploader("Upload tabular CSV data", type=["csv"])
    uploaded_shp = st.file_uploader("Upload spatial data (GeoJSON or SHP)", type=["geojson", "shp", "zip"])

    if uploaded_csv:
        df = pd.read_csv(uploaded_csv)
        st.session_state.df = df
        st.success("✅ Tabular data loaded.")
        st.write(df.head())

    if uploaded_shp:
        gdf = data_loader.load_shapefile(uploaded_shp)
        st.session_state.gdf = gdf
        st.success("✅ Spatial data loaded.")
        st.write(gdf.head())

# 2️⃣ Display Maps
elif menu == "2️⃣ Display Maps":
    st.subheader("🗺️ Leaflet Map View")

    if st.session_state.gdf is not None:
        map_utils.display_leaflet_map(st.session_state.gdf)
    else:
        st.warning("⚠️ Please upload spatial data in the previous step.")

# 3️⃣ Variable Selection
elif menu == "3️⃣ Variable Selection":
    st.subheader("🧮 Variable Selection and Train/Test Split")

    if st.session_state.df is not None:
        df = st.session_state.df
        target = st.selectbox("Select Target Variable", df.columns)
        features = st.multiselect("Select Independent Variables", [col for col in df.columns if col != target])

        split = st.slider("Test size (%)", min_value=10, max_value=50, value=30, step=5)

        if features and target:
            X, y = df[features], df[target]
            X_train, X_test, y_train, y_test = model_trainer.train_test_split_data(X, y, test_size=split / 100)
            st.session_state.update({
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test
            })
            st.success("✅ Features and train-test split prepared.")
    else:
        st.warning("⚠️ Upload tabular data first.")

# 4️⃣ Train Model
elif menu == "4️⃣ Train Model":
    st.subheader("🧠 Select and Train Model")

    if st.session_state.X_train is not None:
        model_type = st.selectbox("Choose Model", ["Random Forest", "Deep Learning (coming soon)"])

        if st.button("Train Model"):
            model, y_train_pred, y_test_pred = model_trainer.train_model(
                model_type, st.session_state.X_train, st.session_state.y_train,
                st.session_state.X_test
            )
            st.session_state.model = model

            # Evaluation
            st.write("### Training Evaluation")
            evaluator.display_metrics(st.session_state.y_train, y_train_pred)

            st.write("### Testing Evaluation")
            evaluator.display_metrics(st.session_state.y_test, y_test_pred)

            # Save outputs
            os.makedirs("outputs", exist_ok=True)
            pd.DataFrame({"Actual": st.session_state.y_train, "Predicted": y_train_pred}).to_csv(
                "outputs/predictions_train.csv", index=False)
            pd.DataFrame({"Actual": st.session_state.y_test, "Predicted": y_test_pred}).to_csv(
                "outputs/predictions_test.csv", index=False)

            st.success("✅ Predictions saved to /outputs folder.")
    else:
        st.warning("⚠️ Perform variable selection first.")

# 5️⃣ Visualize Prediction Map
elif menu == "5️⃣ Visualize Prediction Map":
    st.subheader("🗺️ Prediction Map")

    if st.session_state.gdf is not None and os.path.exists("outputs/predictions_test.csv"):
        gdf = st.session_state.gdf
        preds = pd.read_csv("outputs/predictions_test.csv")
        visualizer.display_prediction_map(gdf, preds)
    else:
        st.warning("⚠️ You need both spatial data and prediction results.")
