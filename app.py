
import streamlit as st
import pandas as pd
import joblib


# ==========================================
# LOAD MODEL & PREPROCESSING FILES
# ==========================================

model = joblib.load("Best_Model.pkl")
scaler = joblib.load("Scaled.pkl")
expected_columns = joblib.load("columns.pkl")


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="centered"
)

st.title("❤️ Heart Disease Predictor")
st.markdown("Enter the patient's details to generate a model prediction.")


# ==========================================
# USER INPUTS
# ==========================================

age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
chol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
restecg = st.selectbox("Resting ECG", [0, 1, 2])
thalach = st.slider("Maximum Heart Rate", 60, 220, 150)
exang = st.selectbox("Exercise-Induced Angina", [0, 1])
oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0)
slope = st.selectbox("ST Slope", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (ca)", [0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3])


# ==========================================
# PREDICTION
# ==========================================

if st.button("Predict", use_container_width=True):

    raw_input = {
        "age": age,
        "sex": 1 if sex == "Male" else 0,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }

    input_df = pd.DataFrame([raw_input])

    # Ensure exact training column order
    input_df = input_df[expected_columns]

    # Apply the same scaler used during training
    scaled_input = scaler.transform(input_df)

    # Predict
    prediction = model.predict(scaled_input)[0]

    # Display result
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("Model predicted: Heart Disease detected (positive class)")
    else:
        st.success("Model predicted: Heart Disease not detected (negative class)")
