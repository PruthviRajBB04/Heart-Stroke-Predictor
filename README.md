# ❤️ Heart Stroke Prediction

A simple, interactive **Streamlit** web application that predicts the risk of heart
disease based on patient health parameters — age, blood pressure, cholesterol, ECG
results, and more — using a pre-trained **Support Vector Machine (SVM)** model.

> ⚠️ **Disclaimer:** This project is for informational and educational purposes only.
> It is **not** a substitute for professional medical advice, diagnosis, or treatment.

---

## 🚀 Features

- Clean, dark-themed UI built with Streamlit
- Form-based input (no reruns on every slider tick)
- One-hot encoding handled automatically to match the model's expected columns
- Feature scaling applied before prediction
- Risk result shown as a styled card (High Risk / Low Risk)
- Approximate confidence score derived from the SVM decision margin
- Expandable explanation of what the "margin" means

---

## 🧠 Tech Stack

| Component     | Tool/Library     |
|---------------|------------------|
| UI Framework  | Streamlit        |
| Data Handling | Pandas           |
| ML Model      | scikit-learn (SVM) |
| Model Loading | Joblib           |

---

## 📁 Project Structure

```
heart-stroke-prediction/
│
├── heart_stroke_app.py     # Main Streamlit application
├── SVM_heart.pkl           # Trained SVM model
├── scaler.pkl              # Fitted feature scaler
├── Columns.pkl             # Expected input columns (post one-hot encoding)
├── requirements.txt        # Python dependencies
└── README.md                # Project documentation
```

---

## ⚙️ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/heart-stroke-prediction.git
   cd heart-stroke-prediction
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   If you don't have a `requirements.txt` yet, create one with:
   ```
   streamlit
   pandas
   scikit-learn
   joblib
   ```

4. **Ensure model files are present**
   Make sure `SVM_heart.pkl`, `scaler.pkl`, and `Columns.pkl` are in the same
   directory as `heart_stroke_app.py`.

5. **Run the app**
   ```bash
   streamlit run heart_stroke_app.py
   ```

6. Open the local URL shown in your terminal (usually `http://localhost:8501`).

---

## 🖥️ Usage

1. Fill in the patient details in the form — age, sex, chest pain type, resting
   blood pressure, cholesterol, fasting blood sugar, ECG results, max heart rate,
   exercise-induced angina, oldpeak, and ST slope.
2. Click **🔍 Predict**.
3. View the risk result (High Risk / Low Risk) along with the model's decision
   margin and an approximate confidence percentage.

---

## 📊 Input Parameters

| Parameter          | Description                                      |
|---------------------|--------------------------------------------------|
| Age                 | Patient age in years                              |
| Sex                 | M / F                                             |
| Chest Pain Type     | ATA, NAP, TA, ASY                                 |
| Resting BP          | Resting blood pressure (mm Hg)                    |
| Cholesterol         | Serum cholesterol (mg/dL)                         |
| Fasting Blood Sugar | 1 if > 120 mg/dL, else 0                          |
| Resting ECG         | Normal, ST, LVH                                   |
| Max Heart Rate      | Maximum heart rate achieved                       |
| Exercise Angina     | Y / N                                             |
| Oldpeak             | ST depression induced by exercise                 |
| ST Slope            | Up, Flat, Down                                    |

---

## 🔍 Model Notes

- The model was trained **without** probability estimates enabled
  (`probability=False`), so `predict_proba` is not available.
- Instead, the app uses `decision_function()` to get the signed distance from the
  separating hyperplane, which is shown as an approximate confidence score.
  This is a rough indicator of model certainty — **not** a calibrated probability.

---

## 📜 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🙋‍♂️ Author

Built by Prem — AIML student, NIE Mysuru.
