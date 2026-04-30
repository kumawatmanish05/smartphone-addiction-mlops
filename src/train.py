import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score
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

    # 🔥 Ordinal Encoding (IMPORTANT)
    df['stress_level'] = df['stress_level'].map({'Low': 1, 'Medium': 2, 'High': 3})
    df['academic_work_impact'] = df['academic_work_impact'].map({'Low': 1, 'Medium': 2, 'High': 3})

    # 🔥 Additional Strong Features
    df['screen_to_sleep_ratio'] = df['daily_screen_time_hours'] / (df['sleep_hours'] + 1)
    df['social_ratio'] = df['social_media_hours'] / (df['daily_screen_time_hours'] + 1)
    df['gaming_ratio'] = df['gaming_hours'] / (df['daily_screen_time_hours'] + 1)
    df['activity_score'] = df['app_opens_per_day'] + df['notifications_per_day']
    df['weekend_spike'] = df['weekend_screen_time'] - df['daily_screen_time_hours']

    df['stress_x_screen'] = df['stress_level'] * df['daily_screen_time_hours']
    df['sleep_deficit'] = (7 - df['sleep_hours']).clip(lower=0)
    df['opens_per_notification'] = df['app_opens_per_day'] / (df['notifications_per_day'] + 1)

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

    # 🔹 One-hot encoding (only for gender)
    X = pd.get_dummies(X, columns=['gender'], drop_first=True)

    # 🔹 Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("✂️ Data split complete")

    # 🔹 Further split for early stopping
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    # 🔹 Class balancing
    weights = compute_sample_weight(class_weight='balanced', y=y_tr)

    # 🔥 XGBoost Model (optimized)
    model = XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        eval_metric='mlogloss'
    )

    print("🔍 Training with early stopping...")

    from xgboost.callback import EarlyStopping

    model.fit(
    X_train, y_train,
    sample_weight=compute_sample_weight(class_weight='balanced', y=y_train)
)

    print("🤖 Model trained")

    # 🔹 Predictions
    y_pred = model.predict(X_test)

    # 🔹 Evaluation
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')

    print("\n📊 Model Performance:")
    print("Accuracy:", acc)
    print("Macro F1 Score:", f1)

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    # 🔹 Save model
    joblib.dump(model, "models/model.pkl")

    print("\n✅ Model saved successfully!")


if __name__ == "__main__":
    main()