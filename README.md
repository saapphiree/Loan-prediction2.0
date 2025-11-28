Loan Prediction Script

A Python-based machine learning application to predict whether a customer will pay back their loan or default. This project demonstrates an end-to-end ML workflow: data preprocessing, model training, and deployment through a Streamlit web interface.

Features

Predicts loan repayment status (paid back or default) using customer financial and demographic data.

Simple web interface for inputting user details.

Uses a pre-trained XGBoost model (loan_model.pkl) for fast predictions.

Handles key features:

Annual income

Debt-to-income ratio

Credit score

Loan amount & interest rate

Gender, marital status, education level, employment status

Loan purpose

Installation

Clone the repository:

git clone <your-repo-url>
cd loan_prediction


Create and activate a conda environment:

conda create -n env1 python=3.10
conda activate env1


Install required packages:

pip install -r requirements.txt


(Alternatively, install manually: streamlit, pandas, numpy, scikit-learn, xgboost.)

Usage

Navigate to the src directory:

cd src


Run the Streamlit app:

streamlit run app.py


Enter the customer details in the interface and click Predict to see the result.

Project Structure
loan_prediction/
├── models/
│   └── loan_model.pkl          # Trained XGBoost model
├── src/
│   ├── app.py                  # Streamlit interface
│   ├── train.py                # Model training script
│   └── clean.py                # Data preprocessing script
├── data/
│   └── train.csv               # Training dataset
├── requirements.txt            # Python dependencies
└── README.md

Dataset

The dataset used in this project was obtained from Kaggle
 for educational purposes.

Demo

App Screenshot Placeholder

Sample Output:

Customer is likely to: DEFAULT

Acknowledgements

Developed with Python, Pandas, Scikit-learn, XGBoost, and Streamlit.

Dataset is for educational purposes.
