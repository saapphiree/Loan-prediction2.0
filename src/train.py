import os
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

TRAIN_CSV_PATH = r"C:/Users/smart view/code/P/loan_prediction/data/processed/train_cleaned.csv"

def load_data():
    if not os.path.exists(TRAIN_CSV_PATH):
        raise FileNotFoundError(f"{TRAIN_CSV_PATH} does not exist.")
    return pd.read_csv(TRAIN_CSV_PATH)


def split_xy(df, target_column):
    x = df.drop(columns=[target_column])
    y = df[target_column]
    return x, y

def build_model():
    return XGBClassifier(
        n_estimators=300,
        max_depth=7,
        learning_rate=0.1,
        objective="binary:logistic",
        eval_metric="logloss"
    )

def main():
    df = load_data()

    target_column = "loan_paid_back"

    x, y = split_xy(df, target_column)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=50
    )

    model = build_model()
    print("Training model...")
    model.fit(x_train, y_train)

    preds = model.predict(x_test)

    print("\nAccuracy:", accuracy_score(y_test, preds))
    print("\nClassification Report:\n", classification_report(y_test, preds))

    MODEL_PATH = r"C:/Users/smart view/code/P/loan_prediction/model/loan_model.pkl"
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    import joblib
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()