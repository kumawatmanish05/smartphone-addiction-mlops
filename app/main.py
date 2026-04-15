from fastapi import FastAPI
import joblib
import pandas as pd

from src.feature_engineering import create_features

app = FastAPI()

# Load model & encoder once
model = joblib.load("models/model.pkl")
le = joblib.load("models/label_encoder.pkl")


@app.get("/")
def home():
    return {"message": "Smartphone Addiction Prediction API is running 🚀"}


@app.post("/predict")
def predict(data: dict):
    # Convert input to DataFrame
    df = pd.DataFrame([data])

    # Feature engineering
    df = create_features(df)

    # Encode categorical
    df = pd.get_dummies(df, drop_first=True)

    # Align columns with training
    df = df.reindex(columns=model.feature_names_in_, fill_value=0)

    # Prediction
    pred = model.predict(df)

    # Decode label
    result = le.inverse_transform(pred)

    return {
        "prediction": result[0]
    }