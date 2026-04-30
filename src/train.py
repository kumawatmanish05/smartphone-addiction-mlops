import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from xgboost import XGBClassifier

from src.preprocessing import load_data, clean_data
from src.feature_engineering import create_features


def main():
    print("🚀 Starting training pipeline...")

    # 🔹 Load Data
    df = load_data("data/raw/smartphone_usage.csv")

    # 🔹 Clean Data
    df = clean_data(df)
    print("🧹 Data cleaned")

    # 🔹 Feature Engineering
    df = create_features(df)

    # 🔥 Additional Features (IMPORTANT)
    df['screen_to_sleep_ratio'] = df['daily_screen_time_hours'] / (df['sleep_hours'] + 1)
    df['social_ratio'] = df['social_media_hours'] / (df['daily_screen_time_hours'] + 1)
    df['gaming_ratio'] = df['gaming_hours'] / (df['daily_screen_time_hours'] + 1)
    df['activity_score'] = df['app_opens_per_day'] + df['notifications_per_day']
    df['weekend_spike'] = df['weekend_screen_time'] - df['daily_screen_time_hours']

    print("⚙️ Features created")

    # 🔹 Target
    target = "addiction_level"

    # 🔹 Encode target
    le = LabelEncoder()
    df[target] = le.fit_transform(df[target])

    # 🔹 Save encoder
    joblib.dump(le, "models/label_encoder.pkl")

    # 🔹 Features & Labels
    X = df.drop(columns=[target])
    y = df[target]

    # 🔹 One-hot encoding
    X = pd.get_dummies(X, drop_first=True)

    # 🔹 Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("✂️ Data split complete")

    # 🔹 Class balancing
    weights = compute_sample_weight(class_weight='balanced', y=y_train)

    # 🔥 XGBoost Model
    base_model = XGBClassifier(
        objective='multi:softmax',
        num_class=3,
        random_state=42,
        eval_metric='mlogloss'
    )

    # 🔥 Hyperparameter tuning
    params = {
        'n_estimators': [200, 300],
        'max_depth': [4, 6],
        'learning_rate': [0.05, 0.1]
    }

    grid = GridSearchCV(base_model, params, cv=3, scoring='accuracy', n_jobs=-1)

    print("🔍 Tuning model...")
    grid.fit(X_train, y_train, sample_weight=weights)

    model = grid.best_estimator_

    print("🤖 Model trained")

    # 🔹 Predictions
    y_pred = model.predict(X_test)

    # 🔹 Evaluation
    acc = accuracy_score(y_test, y_pred)

    print("\n📊 Model Performance:")
    print("Accuracy:", acc)

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    # 🔹 Save model
    joblib.dump(model, "models/model.pkl")

    print("\n✅ Model saved successfully!")


if __name__ == "__main__":
    main()