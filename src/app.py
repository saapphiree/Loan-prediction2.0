import streamlit as st
import pandas as pd
import joblib
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "loan_model.pkl")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"{MODEL_PATH} not found.")
    return joblib.load(MODEL_PATH)

model = load_model()

st.title("Loan Default Prediction App")
st.write("Enter applicant details below:")

annual_income = st.number_input("Annual Income", min_value=0, value=30000)
debt_to_income_ratio = st.number_input("Debt to Income Ratio", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700)
loan_amount = st.number_input("Loan Amount", min_value=0, value=10000)
interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=100.0, value=12.0)

gender = st.selectbox("Gender", ["Male", "Female", "Other"])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
education_level = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD", "Other"])
employment_status = st.selectbox("Employment Status", ["Employed", "Unemployed", "Self-employed", "Retired", "Student"])
loan_purpose = st.selectbox("Loan Purpose", ["Other", "Debt consolidation", "Home", "Education", "Vacation", "Car", "Medical", "Business"])

input_data = pd.DataFrame({
    "annual_income": [annual_income],
    "debt_to_income_ratio": [debt_to_income_ratio],
    "credit_score": [credit_score],
    "loan_amount": [loan_amount],
    "interest_rate": [interest_rate],
    "gender": [gender],
    "marital_status": [marital_status],
    "education_level": [education_level],
    "employment_status": [employment_status],
    "loan_purpose": [loan_purpose]
})

def encode_input(df):
    df = df.copy()
    df["gender"] = df["gender"].map({'Male': 0, 'Female': 1, 'Other': 2})
    df["marital_status"] = df["marital_status"].map({'Single': 0, 'Married': 1, 'Divorced': 2, 'Widowed': 3})
    df["education_level"] = df["education_level"].map({'High School':0, "Master's":1, "Bachelor's":2, 'PhD':3, 'Other':4})
    df["employment_status"] = df["employment_status"].map({'Employed':0, 'Unemployed':1, 'Self-employed':2, 'Retired':3, 'Student':4})
    df["loan_purpose"] = df["loan_purpose"].map({'Other':0, 'Debt consolidation':1, 'Home':2, 'Education':3, 'Vacation':4, 'Car':5, 'Medical':6, 'Business':7})
    return df

input_encoded = encode_input(input_data)

if st.button("Predict Loan Status"):
    pred = model.predict(input_encoded)[0]
    prob = model.predict_proba(input_encoded)[0][pred]
    if pred == 1:
        st.success(f"This customer is likely to PAY BACK. (Confidence: {prob:.2f})")
    else:
        st.error(f"This customer is likely to DEFAULT. (Confidence: {prob:.2f})")
