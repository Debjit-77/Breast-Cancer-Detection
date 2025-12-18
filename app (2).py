import streamlit as st
import pandas as pd
import joblib

# Basic page config
st.set_page_config(page_title="Breast Cancer Detection", page_icon="🎗️")

# Load model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('breast_cancer_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except:
        st.error("⚠️ Model files not found!")
        return None, None

model, scaler = load_model()

# Title
st.title("🎗️ Breast Cancer Detection")
st.write("Enter tumor measurements below")
st.divider()

# Two column layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Mean Values")
    radius_mean = st.number_input("Radius (mean)", 0.0, 30.0, 13.080000, 0.000001, format="%.6f")
    texture_mean = st.number_input("Texture (mean)", 0.0, 50.0, 15.710000, 0.000001, format="%.6f")
    compactness_mean = st.number_input("Compactness (mean)", 0.0, 1.0, 0.127000, 0.000001, format="%.6f")
    concavity_mean = st.number_input("Concavity (mean)", 0.0, 1.0, 0.046000, 0.000001, format="%.6f")
    concave_points_mean = st.number_input("Concave Points (mean)", 0.0, 1.0, 0.031000, 0.000001, format="%.6f")
    radius_se = st.number_input("Radius (SE)", 0.0, 5.0, 0.185000, 0.000001, format="%.6f")
    concave_points_se = st.number_input("Concave Points (SE)", 0.0, 0.1, 0.006000, 0.000001, format="%.6f")

with col2:
    st.subheader("Worst Values")
    radius_worst = st.number_input("Radius (worst)", 0.0, 50.0, 14.500000, 0.000001, format="%.6f")
    texture_worst = st.number_input("Texture (worst)", 0.0, 60.0, 20.490000, 0.000001, format="%.6f")
    smoothness_worst = st.number_input("Smoothness (worst)", 0.0, 1.0, 0.131000, 0.000001, format="%.6f")
    compactness_worst = st.number_input("Compactness (worst)", 0.0, 2.0, 0.278000, 0.000001, format="%.6f")
    concavity_worst = st.number_input("Concavity (worst)", 0.0, 2.0, 0.189000, 0.000001, format="%.6f")
    concave_points_worst = st.number_input("Concave Points (worst)", 0.0, 1.0, 0.073000, 0.000001, format="%.6f")
    symmetry_worst = st.number_input("Symmetry (worst)", 0.0, 1.0, 0.318000, 0.000001, format="%.6f")

st.divider()

# Predict button
if st.button("🔬 Predict", type="primary", use_container_width=True):
    if model and scaler:
        # Create input dataframe
        input_data = pd.DataFrame({
            'radius_mean': [radius_mean],
            'texture_mean': [texture_mean],
            'compactness_mean': [compactness_mean],
            'concavity_mean': [concavity_mean],
            'concave points_mean': [concave_points_mean],
            'radius_se': [radius_se],
            'concave points_se': [concave_points_se],
            'radius_worst': [radius_worst],
            'texture_worst': [texture_worst],
            'smoothness_worst': [smoothness_worst],
            'compactness_worst': [compactness_worst],
            'concavity_worst': [concavity_worst],
            'concave points_worst': [concave_points_worst],
            'symmetry_worst': [symmetry_worst]
        })

        # Predict
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]

        # Show result
        st.divider()
        if prediction == 0:
            st.success("✅ **BENIGN** - Non-cancerous tumor")
            st.metric("Benign Probability", f"{probability[0]*100:.1f}%")
        else:
            st.error("⚠️ **MALIGNANT** - Cancerous tumor")
            st.metric("Malignant Probability", f"{probability[1]*100:.1f}%")