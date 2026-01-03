
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

user_input = []
for feature in FEATURES:
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
