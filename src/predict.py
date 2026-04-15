import joblib
import pandas as pd

from src.feature_engineering import create_features


def predict(data: dict):
    # Convert input to DataFrame
    df = pd.DataFrame([data])

    # Apply feature engineering
    df = create_features(df)

    # Encode categorical
    df = pd.get_dummies(df, drop_first=True)

    # Load model
    model = joblib.load("models/model.pkl")

    # Align columns (IMPORTANT)
    model_features = model.feature_names_in_
    df = df.reindex(columns=model_features, fill_value=0)

    # Predict
    prediction = model.predict(df)

    return prediction[0]


if __name__ == "__main__":
    # Example input
    sample_user = {
        "age": 22,
        "gender": "Male",
        "daily_screen_time_hours": 7,
        "social_media_hours": 3,
        "gaming_hours": 2,
        "work_study_hours": 2,
        "sleep_hours": 5,
        "notifications_per_day": 120,
        "app_opens_per_day": 80,
        "weekend_screen_time": 9,
        "stress_level": "High",
        "academic_work_impact": "High"
    }

    result = predict(sample_user)
    label_map = {
    0: "Mild",
    1: "Moderate",
    2: "Severe"
}

print("📱 Predicted Addiction Level:", label_map[result])