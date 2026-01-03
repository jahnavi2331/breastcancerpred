
import streamlit as st
import pickle
import numpy as np

# Load model and scaler
with open("breast_cancer_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

FEATURES = [
    'radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean',
    'compactness_mean','concavity_mean','concave_points_mean','symmetry_mean','fractal_dimension_mean',
    'radius_se','texture_se','perimeter_se','area_se','smoothness_se','compactness_se',
    'concavity_se','concave_points_se','symmetry_se','fractal_dimension_se',
    'radius_worst','texture_worst','perimeter_worst','area_worst','smoothness_worst',
    'compactness_worst','concavity_worst','concave_points_worst','symmetry_worst','fractal_dimension_worst'
]

st.set_page_config(page_title="Breast Cancer Detection", layout="centered")
st.title("🧬 Breast Cancer Detection System")

st.write("Enter the tumor measurements below:")

# Typical medical ranges (approx from dataset)
ranges = {
    'radius_mean': (5.0, 30.0, 14.0),
    'texture_mean': (5.0, 40.0, 20.0),
    'perimeter_mean': (40.0, 200.0, 90.0),
    'area_mean': (100.0, 3000.0, 700.0),
    'smoothness_mean': (0.05, 0.2, 0.1),
    'compactness_mean': (0.02, 0.4, 0.1),
    'concavity_mean': (0.0, 0.5, 0.1),
    'concave_points_mean': (0.0, 0.3, 0.05),
    'symmetry_mean': (0.1, 0.3, 0.18),
    'fractal_dimension_mean': (0.04, 0.1, 0.06),
}

user_input = []

for feature in FEATURES:
    if feature in ranges:
        min_v, max_v, default = ranges[feature]
        value = st.number_input(feature, min_value=min_v, max_value=max_v, value=default)
    else:
        value = st.number_input(feature, value=0.0)
    user_input.append(value)


if st.button("Predict"):
    data = np.array(user_input).reshape(1, -1)
    data = scaler.transform(data)
    prediction = model.predict(data)[0]

    if prediction == 1:
        st.error("⚠️ Malignant Tumor Detected")
    else:
        st.success("✅ Benign Tumor Detected")
