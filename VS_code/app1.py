import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Heart Stroke Prediction", page_icon="❤️", layout="centered")


# ---------- Load model artifacts (cached so this runs once, not on every interaction) ----------
@st.cache_resource
def load_artifacts():
    model = joblib.load("SVM_heart.pkl")
    scaler = joblib.load("scaler.pkl")
    expected_columns = joblib.load("Columns.pkl")
    return model, scaler, expected_columns


try:
    model, scaler, expected_columns = load_artifacts()
except FileNotFoundError as e:
    st.error(f"Missing model file: {e}. Make sure SVM_heart.pkl, scaler.pkl, and Columns.pkl "
             f"are in the same folder as this script.")
    st.stop()


st.title("❤️ Heart Stroke Prediction")
st.markdown("Provide the following details to check your heart disease risk.")

# ---------- Collect input inside a form (prevents a full rerun on every slider tick) ----------
with st.form("patient_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 0, 100, 40)
        sex = st.selectbox("Sex", ["M", "F"])
        chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
        resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
        cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])

    with col2:
        resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
        max_hr = st.slider("Max Heart Rate", 60, 220, 150)
        exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
        oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0, step=0.1)
        st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

    submitted = st.form_submit_button("Predict", use_container_width=True)


if submitted:
    # Start every column at 0, then set only the ones that apply.
    # This avoids creating throwaway dummy columns (e.g. "Sex_F", "ChestPainType_ASY")
    # for the one-hot categories that were dropped as the baseline during training.
    input_row = {col: 0 for col in expected_columns}

    input_row["Age"] = age
    input_row["RestingBP"] = resting_bp
    input_row["Cholesterol"] = cholesterol
    input_row["FastingBS"] = fasting_bs
    input_row["MaxHR"] = max_hr
    input_row["Oldpeak"] = oldpeak

    for dummy in (f"Sex_{sex}", f"ChestPainType_{chest_pain}",
                  f"RestingECG_{resting_ecg}", f"ExerciseAngina_{exercise_angina}",
                  f"ST_Slope_{st_slope}"):
        if dummy in input_row:
            input_row[dummy] = 1

    input_df = pd.DataFrame([input_row])[expected_columns]

    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    # model.probability was False at training time, so predict_proba isn't available.
    # decision_function gives the signed distance from the separating hyperplane instead —
    # useful as a rough margin/confidence indicator, not a calibrated probability.
    margin = model.decision_function(scaled_input)[0]
    confidence = min(abs(margin) / 3, 1.0) * 100  # rough 0-100% scale for display only

    st.divider()
    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")

    st.caption(f"Model margin: {margin:.2f}  •  Approx. confidence: {confidence:.0f}%")
    with st.expander("What does 'margin' mean?"):
        st.write(
            "This SVM wasn't trained with probability estimates enabled, so there's no true "
            "confidence percentage. The margin is the distance from the model's decision "
            "boundary — values further from 0 mean the model is more confident either way."
            "\n\nAI generated content is for informational purposes only and should not be considered medical advice. "
        )