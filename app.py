import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

st.set_page_config(layout="wide", page_title="Heart Disease Prediction App")

st.title("❤️ Heart Disease Prediction")
st.markdown("---")
st.write("Predict the likelihood of heart disease based on various health parameters using your trained model.")

st.subheader("Input Patient Data:")

try:
    model_heart = joblib.load('KNN_heart.pkl')
    scaler_heart = joblib.load('scaler.pkl')
    expected_columns_heart = joblib.load('columns.pkl')
except FileNotFoundError:
    st.error("Error: Model files (KNN_heart.pkl, scaler.pkl, columns.pkl) not found.")
    st.warning("Please ensure these files are in the same directory as heart_disease_app.py after training your model with heart.py.")
    st.stop()

CH_MEAN = 247.35
RESTBP_MEAN = 132.65

col1, col2, col3 = st.columns(3)
with col1:
    age = st.slider("Age", 20, 80, 45)
    sex = st.selectbox("Sex", ["Male", "Female"])
    chest_pain_type = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
    resting_bp = st.slider("Resting Blood Pressure (mmHg)", 0, 200, 120)
with col2:
    cholesterol = st.slider("Cholesterol (mg/dL)", 0, 600, 200)
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", ["No", "Yes"])
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    max_hr = st.slider("Maximum Heart Rate Achieved", 60, 220, 150)
with col3:
    exercise_angina = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
    oldpeak = st.slider("Oldpeak (ST depression induced by exercise)", -2.0, 6.0, 1.0, 0.1)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

if st.button("Predict Heart Disease"):
    if cholesterol == 0:
        cholesterol = CH_MEAN
    if resting_bp == 0:
        resting_bp = RESTBP_MEAN

    raw_input_data = {
        'Age': age,
        'Sex': sex,
        'ChestPainType': chest_pain_type,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': 'Yes' if fasting_bs == 'Yes' else 'No',
        'RestingECG': resting_ecg,
        'ExerciseAngina': 'Yes' if exercise_angina == 'Yes' else 'No',
        'Oldpeak': oldpeak,
        'MaxHR': max_hr,
        'ST_Slope': st_slope
    }
    input_df_raw = pd.DataFrame([raw_input_data])

    processed_input = pd.DataFrame(0, index=[0], columns=expected_columns_heart)

    processed_input['Age'] = input_df_raw['Age']
    processed_input['RestingBP'] = input_df_raw['RestingBP']
    processed_input['Cholesterol'] = input_df_raw['Cholesterol']
    processed_input['FastingBS'] = input_df_raw['FastingBS'].map({'No': 0, 'Yes': 1})
    processed_input['MaxHR'] = input_df_raw['MaxHR']
    processed_input['Oldpeak'] = input_df_raw['Oldpeak']

    if 'Sex_M' in expected_columns_heart:
        processed_input['Sex_M'] = 1 if input_df_raw['Sex'].iloc[0] == 'Male' else 0

    if f'ChestPainType_{input_df_raw["ChestPainType"].iloc[0]}' in expected_columns_heart:
        processed_input[f'ChestPainType_{input_df_raw["ChestPainType"].iloc[0]}'] = 1

    if f'RestingECG_{input_df_raw["RestingECG"].iloc[0]}' in expected_columns_heart:
        processed_input[f'RestingECG_{input_df_raw["RestingECG"].iloc[0]}'] = 1

    if 'ExerciseAngina_Y' in expected_columns_heart:
        processed_input['ExerciseAngina_Y'] = 1 if input_df_raw['ExerciseAngina'].iloc[0] == 'Yes' else 0

    if f'ST_Slope_{input_df_raw["ST_Slope"].iloc[0]}' in expected_columns_heart:
        processed_input[f'ST_Slope_{input_df_raw["ST_Slope"].iloc[0]}'] = 1

    processed_input = processed_input[expected_columns_heart]

    input_data_scaled = scaler_heart.transform(processed_input)

    prediction_proba = model_heart.predict_proba(input_data_scaled)[0]
    prediction = np.argmax(prediction_proba)

    st.subheader("Prediction Result:")
    if prediction == 1:
        st.error(f"**High Likelihood of Heart Disease!** (Confidence: {prediction_proba[1]*100:.2f}%)")
        st.write("Please consult a medical professional for accurate diagnosis.")
    else:
        st.success(f"**Low Likelihood of Heart Disease.** (Confidence: {prediction_proba[0]*100:.2f}%)")
        st.write("Continue to monitor your health.")

