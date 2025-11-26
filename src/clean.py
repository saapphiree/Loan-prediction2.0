import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_PATH = os.path.join(BASE_DIR, "data", "raw", "train.csv")
TEST_PATH = os.path.join(BASE_DIR, "data", "raw", "test.csv")

PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

def clean_data(df):

    df = df.drop(columns=["id", "grade_subgrade"], errors='ignore')
    if "gender" in df.columns:
        df["gender"] = df["gender"].map({'Male': 0, 'Female': 1, 'Other': 2})
    if "marital_status" in df.columns:
        df["marital_status"] = df["marital_status"].map({'Single': 0, 'Married': 1, 'Divorced': 2, 'Widowed': 3})
    if "education_level" in df.columns:
        df["education_level"] = df["education_level"].map({'High School':0, "Master's":1, "Bachelor's":2, 'PhD':3, 'Other':4})
    if "loan_purpose" in df.columns:
        df["loan_purpose"] = df["loan_purpose"].map({'Other':0, 'Debt consolidation':1, 'Home':2, 'Education':3, 'Vacation':4, 'Car':5,
                                                 'Medical':6, 'Business':7})
    if "employment_status" in df.columns:
        df["employment_status"] = df["employment_status"].map({'Employed':0, 'Unemployed':1, 'Self-employed':2, 'Retired':3, 'Student':4})

    return df

# Load and clean train dataset
train_df = pd.read_csv(TRAIN_PATH)
train_df = clean_data(train_df)
train_df.to_csv(os.path.join(PROCESSED_DIR, "train_cleaned.csv"), index=False)
print("Train dataset cleaned:", train_df.shape)

# Load and clean test dataset
test_df = pd.read_csv(TEST_PATH)
test_df = clean_data(test_df)
test_df.to_csv(os.path.join(PROCESSED_DIR, "test_cleaned.csv"), index=False)
print("Test dataset cleaned:", test_df.shape)

"""print(train_df.shape)
print(train_df.info())
print(train_df['marital_status'].unique())
print(train_df['education_level'].unique())
print(train_df['loan_purpose'].unique())
print(train_df.isna().sum())"""
print("Train set info", train_df.head())
print(train_df.info())
print("Test set info", test_df.head())
print(test_df.info())

PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

train_df.to_csv(os.path.join(PROCESSED_DIR, "train_cleaned.csv"), index=False)
test_df.to_csv(os.path.join(PROCESSED_DIR, "test_cleaned.csv"), index=False)

print("Cleaned datasets saved in:", PROCESSED_DIR)