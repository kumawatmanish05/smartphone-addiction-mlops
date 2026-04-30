import streamlit as st
import pandas as pd
import joblib

# Load model + encoder + features
model = joblib.load("models/model.pkl")
le = joblib.load("models/label_encoder.pkl")
features = joblib.load("models/features.pkl")

st.set_page_config(page_title="Smartphone Addiction Predictor", layout="centered")

st.title("📱 Smartphone Addiction Predictor")
st.write("Predict your addiction level based on usage habits")

# 🔹 User Inputs
age = st.slider("Age", 10, 60, 22)
gender = st.selectbox("Gender", ["Male", "Female"])

daily_screen_time = st.slider("Daily Screen Time (hrs)", 0.0, 15.0, 6.0)
social_media = st.slider("Social Media Usage (hrs)", 0.0, 10.0, 3.0)
gaming = st.slider("Gaming Hours (hrs)", 0.0, 10.0, 1.0)
work_study = st.slider("Work/Study Hours (hrs)", 0.0, 12.0, 4.0)
sleep = st.slider("Sleep Hours", 0.0, 12.0, 6.0)

notifications = st.slider("Notifications per Day", 0, 300, 100)
app_opens = st.slider("App Opens per Day", 0, 200, 50)
weekend_screen = st.slider("Weekend Screen Time", 0.0, 15.0, 7.0)

stress = st.selectbox("Stress Level", ["Low", "Medium", "High"])
academic = st.selectbox("Academic Work Impact", ["Low", "Medium", "High"])

# 🔹 Convert inputs into DataFrame
input_data = pd.DataFrame([{
    "age": age,
    "gender": gender,
    "daily_screen_time_hours": daily_screen_time,
    "social_media_hours": social_media,
    "gaming_hours": gaming,
    "work_study_hours": work_study,
    "sleep_hours": sleep,
    "notifications_per_day": notifications,
    "app_opens_per_day": app_opens,
    "weekend_screen_time": weekend_screen,
    "stress_level": stress,
    "academic_work_impact": academic
}])

# 🔥 SAME PROCESSING AS TRAINING

# Ordinal encoding
input_data['stress_level'] = input_data['stress_level'].map({'Low': 1, 'Medium': 2, 'High': 3})
input_data['academic_work_impact'] = input_data['academic_work_impact'].map({'Low': 1, 'Medium': 2, 'High': 3})

# Feature engineering
input_data['screen_to_sleep_ratio'] = input_data['daily_screen_time_hours'] / (input_data['sleep_hours'] + 1)
input_data['social_ratio'] = input_data['social_media_hours'] / (input_data['daily_screen_time_hours'] + 1)
input_data['gaming_ratio'] = input_data['gaming_hours'] / (input_data['daily_screen_time_hours'] + 1)
input_data['activity_score'] = input_data['app_opens_per_day'] + input_data['notifications_per_day']
input_data['weekend_spike'] = input_data['weekend_screen_time'] - input_data['daily_screen_time_hours']

input_data['stress_x_screen'] = input_data['stress_level'] * input_data['daily_screen_time_hours']
input_data['sleep_deficit'] = (7 - input_data['sleep_hours']).clip(lower=0)
input_data['opens_per_notification'] = input_data['app_opens_per_day'] / (input_data['notifications_per_day'] + 1)

# One-hot encoding
input_data = pd.get_dummies(input_data, columns=['gender'], drop_first=True)

# Match training columns
input_data = input_data.reindex(columns=features, fill_value=0)

# 🔹 Predict
if st.button("Predict Addiction Level"):
    prediction = model.predict(input_data)
    result = le.inverse_transform(prediction)[0]

    st.success(f"📊 Predicted Addiction Level: {result}")