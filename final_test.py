import pickle
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

print(" Loading model and encoders ")

# 1. Load the pre-trained model and encoders from pickle files
with open("churn_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

print(" Model and encoders loaded successfully!\n")
print("=========================================")
print(" Prediction for New Customer Scenario:")
print("=========================================\n")

# 2. Dummy data for a new customer to test the model
new_customer = {
    'gender': 'Female',                         # Options: 'Male', 'Female'
    'SeniorCitizen': 0,                        # 0 = No, 1 = Yes
    'Partner': 'No',                           # Options: 'Yes', 'No'
    'Contract': 'Month-to-month',               # Options: 'Month-to-month', 'One year', 'Two year'
    'tenure': 2,                               # Total months customer has stayed with the company
    'MonthlyCharges': 85.50,                   # Monthly bill amount
    'TotalCharges': 171.00                     # Cumulative bill amount
}

# 3. Label Encoding: Convert text inputs to numerical format and extract the scalar value [0]
gender_encoded = encoders['gender'].transform([new_customer['gender']])[0]
partner_encoded = encoders['Partner'].transform([new_customer['Partner']])[0]
contract_encoded = encoders['Contract'].transform([new_customer['Contract']])[0]

# 4. Prepare input dataframe for model prediction
input_data = pd.DataFrame([{
    'gender': gender_encoded,
    'SeniorCitizen': new_customer['SeniorCitizen'],
    'Partner': partner_encoded,
    'tenure': new_customer['tenure'],
    'MonthlyCharges': new_customer['MonthlyCharges'],
    'TotalCharges': new_customer['TotalCharges'],
    'Contract': contract_encoded
}])

# 5. Perform model prediction and fetch prediction probabilities
prediction = model.predict(input_data)
probability = model.predict_proba(input_data) * 100

# 6. Display prediction results on the output console
print(f" Customer Details: Gender={new_customer['gender']}, Contract={new_customer['Contract']}, Tenure={new_customer['tenure']} Months")
print(f" Churn Probability calculated by AI: {probability[0][1]:.2f}%")

if prediction[0] == 1:
    print("\n Result: This customer is highly likely to leave the company! (High Churn Risk)")
else:
    print("\n Result: This customer is loyal and safe. (Low Churn Risk)")
