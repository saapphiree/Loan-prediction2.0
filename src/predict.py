import os
import numpy as np
import pandas as pd
import joblib

MODEL_PATH = r"C:/Users/smart view/code/P/loan_prediction/model/loan_model.pkl" 
TEST_CSV_PATH = r"C:/Users/smart view/code/P/loan_prediction/data/processed/test_cleaned.csv"   
OUTPUT_PATH = r"C:/Users/smart view/code/P/loan_prediction/data/prediction/prediction.csv"    

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"{MODEL_PATH} does not exist.")
    return joblib.load(MODEL_PATH)

def load_data():
    if not os.path.exists(TEST_CSV_PATH):
        raise FileNotFoundError(f"{TEST_CSV_PATH} does not exist.")
    return pd.read_csv(TEST_CSV_PATH)

def main():
    model = load_model()
    df = load_data()

    if 'id' in df.columns:
        ids = df['id']
    else:
        ids = pd.Series(range(len(df)), name='id')

    target_col = "loan_status"
    if target_col in df.columns:
        X = df.drop(columns=[target_col])
    else:
        X = df

    preds = model.predict(X)

    output_df = pd.DataFrame({
        'id': ids,
        'prediction': preds
    })
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Predictions saved to {OUTPUT_PATH}")
    print(output_df.head())

if __name__ == "__main__":
    main()