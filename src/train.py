import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import compute_sample_weight

from src.preprocessing import load_data, clean_data
from src.feature_engineering import create_features


def main():
    print("🚀 Starting training pipeline...")

    # -------- LOAD DATA --------
    df = load_data("data/raw/smartphone_usage.csv")

    # -------- CLEAN DATA --------
    df = clean_data(df)
    print("🧹 Data cleaned")

    # -------- FEATURE ENGINEERING --------
    df = create_features(df)
    print("⚙️ Features created")

    # -------- TARGET --------
    target = 'addiction_level'

    # Encode target
    le = LabelEncoder()
    df[target] = le.fit_transform(df[target])

    # -------- FEATURES --------
    X = df.drop(columns=['addiction_level', 'addicted_label'])
    y = df[target]

    # Encode categorical
    X = pd.get_dummies(X, drop_first=True)

    # -------- SPLIT --------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print("✂️ Data split complete")

    # -------- MODEL --------
    from sklearn.ensemble import GradientBoostingClassifier

    model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
    )
    from sklearn.utils.class_weight import compute_sample_weight

    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

    model.fit(X_train, y_train, sample_weight=sample_weights)
    print("🤖 Model trained")

    # -------- PREDICTION --------
    y_pred = model.predict(X_test)

    # -------- EVALUATION --------
    print("\n📊 Model Performance:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    # -------- SAVE MODEL --------
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")

    print("\n✅ Model saved successfully!")


if __name__ == "__main__":
    main()