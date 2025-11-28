# Loan Prediction Application

[![Python](https://img.shields.io/badge/python-3.10-blue?logo=python)](https://www.python.org/) [![Streamlit](https://img.shields.io/badge/Streamlit-App-green?logo=streamlit)](https://streamlit.io/) [![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

A Python-based machine learning application to predict loan repayment status. Users can input financial and demographic information to see whether a customer is likely to **pay back a loan** or **default**.

---

## Table of Contents

1. [Features](#features)
2. [Dataset](#dataset)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Project Structure](#project-structure)
6. [Demo](#demo)
7. [Acknowledgements](#acknowledgements)
8. [License](#license)

---

## Features

* Predicts loan repayment (`paid back` or `default`) using a pre-trained XGBoost model.
* Web interface powered by **Streamlit** for easy interaction.
* Handles key features such as:

  * Annual income
  * Debt-to-income ratio
  * Credit score
  * Loan amount & interest rate
  * Gender, marital status, education level, employment status
  * Loan purpose

---

## Dataset

* Dataset used is from **[Kaggle](https://www.kaggle.com/)** for educational purposes.
* Contains financial and demographic information for customers along with loan repayment status.

---

## Installation

1. Clone the repository:

```bash
git clone <your-repo-url>
cd loan_prediction
```

2. Create and activate a conda environment:

```bash
conda create -n env1 python=3.10
conda activate env1
```

3. Install required packages:

```bash
pip install -r requirements.txt
```

*(Alternatively, manually install `pandas`, `numpy`, `scikit-learn`, `xgboost`, `streamlit`.)*

---

## Usage

1. Navigate to the source directory:

```bash
cd src
```

2. Run the Streamlit app:

```bash
streamlit run app.py
```

3. Enter the customer details in the app interface.
4. Click **Predict** to see the output.

---

## Project Structure

```
loan_prediction/
├── data/
│   └── train.csv              # Dataset from Kaggle
├── models/
│   └── loan_model.pkl         # Trained XGBoost model
├── src/
│   ├── app.py                 # Streamlit app
│   ├── clean.py               # Data preprocessing script
│   └── train.py               # Model training script
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

## Demo

**Screenshot Placeholder**

![Screenshot](https://via.placeholder.com/600x400.png?text=App+Screenshot)

**Sample Output Example:**

```
Customer is likely to: DEFAULT
```

---

## Acknowledgements

* Built with **Python**, **Pandas**, **Scikit-learn**, **XGBoost**, and **Streamlit**.
* Dataset is used purely for **educational purposes**.
