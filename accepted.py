# Import necessary libraries
import pandas as pd
import numpy as np
import streamlit as st
import joblib

# Title of the App
st.title("Loan Repayment Risk Analysis")
st.write("This application evaluates a borrower’s post-loan repayment behavior using deep learning to predict the likelihood of default based on financial and behavioral features.")

# Load the trained model
model = joblib.load('accepted.pkl')
scaler = joblib.load('scaler.pkl')

# The input features
st.header("Input Features")

loan_amount = st.number_input('Loan Amount', min_value = 1000, max_value = 500000, value = 5000)
int_rate = st.number_input('Interest Rate', min_value = 5, max_value = 40, value = 10)
annual_income = st.number_input('Annual Income', min_value = 10000, max_value = 120000000, value = 120000)
invst_fund = st.number_input('Investor Funded Amount', min_value = 500, max_value = 50000, value = 10000)
debt_to_inc = st.number_input('Debt to Income Ratio', min_value = -5, max_value = 1000, value = 50)
total_late_fee = st.number_input('Late Fee Received', min_value = 1, max_value = 1500, value = 30)
rev_bal = st.number_input('Revolving Bal', min_value = 100, max_value = 2600000, value = 1000)
out_prin = st.number_input('Outstanding Principal', min_value = 0, max_value = 1560000, value = 2000)
old_acct = st.number_input('Old Acct Month', min_value = 0, max_value = 1000, value = 50)
rev_line = st.number_input('Revolving Line Utilization', min_value = 0, max_value = 400, value = 100)
total_pymt = st.number_input('Total Paid', min_value = 0, max_value = 100000, value = 50)
total_amt = st.number_input('Total Amount', min_value = 0, max_value = 15000000, value = 100)
last_paid = st.number_input('Last Amount Paid', min_value = 0, max_value = 50000, value = 200)
installment = st.number_input('Installment', min_value = 0, max_value = 2000, value = 10)
rec_bal_mth = st.number_input('Month Since Recent Bal', min_value = 0, max_value = 700, value = 20)
rev_limit = st.number_input('Revolving High Limit', min_value = 0, max_value = 10000000, value = 200)
cur_bal = st.number_input('Average Current Balance', min_value = 0, max_value = 650000, value = 3000)
bal_ex_mort = st.number_input('Balance Excluding Mortgage', min_value = 0, max_value = 3500000, value = 500)
rec_acct = st.number_input('Month Since Recent Account', min_value = 0, max_value = 200, value = 20)
total_prin = st.number_input('Total Received Principal', min_value = 0, max_value = 500000, value = 1000)
total_int = st.number_input('Total Interest Received', min_value = 0, max_value = 30000, value = 3000)

inputs = {
    'Loan Amount': [loan_amount],
    'Interest Rate': [int_rate],
    'Annual Income': [annual_income],
    'Investment Funds': [invst_fund],
    'Debt To Income': [debt_to_inc],
    'Total Late Fee Received': [total_late_fee],
    'Revolving Balance': [rev_bal],
    'Outstanding Principal': [out_prin],
    'Month Since Old Account': [old_acct],
    'Revolving Line Utilization': [rev_line],
    'Total Payment': [total_pymt],
    'Total Amount': [total_amt],
    'Last Amount Paid': [last_paid],
    'Installment': [installment],
    'Month Since Recent Bal': [rec_bal_mth],
    'Revolving High Limit': [rev_limit],
    'Average Current Balance': [cur_bal],
    'Balance Excluding Mortgage': [bal_ex_mort],
    'Month Since Recent Account': [rec_acct],
    'Total Principal Received': [total_prin],
    'Total Interest Received': [total_int]
}

# Convert the input to dataframe
input_df = pd.DataFrame(inputs)

# Display the input data
st.subheader('Input Data')
st.write(input_df)

# Scale to match training data
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(input_df)
scaled_data = scaler.transform(X_imputed)

# Prediction
if st.button('Predict'):
    reshaped = scaled_data.reshape(1, -1)
    prediction_prob = model.predict(reshaped)[0][0]
    prediction = 1 if prediction_prob >= 0.5 else 0
    
    if prediction == 1:
        st.success(f"The model predicts that this customer will pay back --with {prediction_prob * 100:.2f}% confidence.")
    else:
        st.error(f"The model predicts that this customer will NOT pay back --with {(1 - prediction_prob) * 100:.2f}% confidence.")
