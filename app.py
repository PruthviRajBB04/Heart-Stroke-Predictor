import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Heart Stroke Prediction", page_icon="❤️", layout="centered")

# ---------- Custom CSS ----------
st.markdown("""
<style>
    /* Overall background - dark */
    .stApp {
        background: linear-gradient(180deg, #14181f 0%, #1b2028 100%);
    }

    /* Make default body text light and readable on the dark background */
    .stApp, .stApp p, .stApp label, .stApp span,
    .stApp .stMarkdown, .stApp .stCaption {
        color: #e8eaed;
    }

    /* Hero header */
    .hero {
        text-align: center;
        padding: 1.75rem 1rem 1.25rem 1rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #c0392b 0%, #e67e22 100%);
        color: #ffffff;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
    }
    .hero h1 {
        font-size: 2rem;
        margin-bottom: 0.25rem;
        color: #ffffff;
    }
    .hero p {
        font-size: 0.95rem;
        color: #fdecea;
        opacity: 1;
        margin: 0;
    }

    /* Section labels inside the form */
    .section-label {
        font-weight: 700;
        font-size: 0.85rem;
        color: #ff8a75;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin: 0.5rem 0 0.4rem 0;
    }

    /* Form container card */
    div[data-testid="stForm"] {
        background: #232833;
        padding: 1.75rem;
        border-radius: 16px;
        border: 1px solid #333a47;
        box-shadow: 0 4px 18px rgba(0,0,0,0.3);
    }

    /* Labels for sliders / selectboxes / number inputs */
    div[data-testid="stForm"] label p {
        color: #f1f3f5 !important;
        font-weight: 500;
    }

    /* Input widgets themselves */
    div[data-testid="stForm"] input,
    div[data-testid="stForm"] select,
    div[data-baseweb="select"] > div {
        background-color: #2f3542 !important;
        color: #ffffff !important;
        border-color: #454c5c !important;
    }

    /* Slider value + track */
    div[data-testid="stSlider"] [data-testid="stTickBarMin"],
    div[data-testid="stSlider"] [data-testid="stTickBarMax"] {
        color: #cfd3da !important;
    }

    /* Submit button */
    div[data-testid="stFormSubmitButton"] button {
        background: linear-gradient(135deg, #e74c3c 0%, #f39c12 100%);
        color: #ffffff;
        font-weight: 700;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 0;
        transition: transform 0.15s ease;
    }
    div[data-testid="stFormSubmitButton"] button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(231, 76, 60, 0.45);
    }
    div[data-testid="stFormSubmitButton"] button p {
        color: #ffffff !important;
        font-weight: 700;
    }

    /* Result cards */
    .result-card {
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        margin-top: 1rem;
        margin-bottom: 1rem;
        border: 1px solid;
    }
    .result-high {
        background: #3a1f1d;
        border-color: #e74c3c;
    }
    .result-low {
        background: #1c3a2b;
        border-color: #2ecc71;
    }
    .result-card h2 {
        margin: 0 0 0.4rem 0;
    }
    .result-high h2 { color: #ff6b5e; }
    .result-low h2 { color: #4dd88a; }
    .result-card p {
        color: #e8eaed !important;
        font-size: 0.95rem;
        margin: 0;
    }

    .disclaimer {
        font-size: 0.78rem;
        color: #9aa0aa;
        text-align: center;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)


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


# ---------- Hero header ----------
st.markdown("""
<div class="hero">
    <h1>❤️ Heart Stroke Prediction</h1>
    <p>Fill in the patient details below to estimate heart disease risk using an SVM model.</p>
</div>
""", unsafe_allow_html=True)

# ---------- Collect input inside a form (prevents a full rerun on every slider tick) ----------
with st.form("patient_form"):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-label">Patient Info</div>', unsafe_allow_html=True)
        age = st.slider("Age", 0, 100, 40)
        sex = st.selectbox("Sex", ["M", "F"])
        chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])

        st.markdown('<div class="section-label">Vitals</div>', unsafe_allow_html=True)
        resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
        cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])

    with col2:
        st.markdown('<div class="section-label">ECG &amp; Exercise</div>', unsafe_allow_html=True)
        resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
        max_hr = st.slider("Max Heart Rate", 60, 220, 150)
        exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
        oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0, step=0.1)
        st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

    st.write("")
    submitted = st.form_submit_button("🔍 Predict", use_container_width=True)


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
        st.markdown(f"""
        <div class="result-card result-high">
            <h2>⚠️ High Risk of Heart Disease</h2>
            <p>Model margin: {margin:.2f} • Approx. confidence: {confidence:.0f}%</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-card result-low">
            <h2>✅ Low Risk of Heart Disease</h2>
            <p>Model margin: {margin:.2f} • Approx. confidence: {confidence:.0f}%</p>
        </div>
        """, unsafe_allow_html=True)

    st.progress(min(int(confidence), 100))

    with st.expander("What does 'margin' mean?"):
        st.write(
            "This SVM wasn't trained with probability estimates enabled, so there's no true "
            "confidence percentage. The margin is the distance from the model's decision "
            "boundary — values further from 0 mean the model is more confident either way."
        )

    st.markdown(
        '<p class="disclaimer">AI generated content is for informational purposes only '
        'and should not be considered medical advice.</p>',
        unsafe_allow_html=True
    )