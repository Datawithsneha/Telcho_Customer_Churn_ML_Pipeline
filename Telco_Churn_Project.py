import mysql.connector
import pandas as pd
import pickle
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# 1. Suppress warnings for a clean and professional console output
warnings.filterwarnings('ignore')

print("==================================================")
print(" STARTING: END-TO-END TELCO CHURN ML PIPELINE")
print("==================================================\n")

# 2. Step 1: Fetching data directly from your MySQL Database
print(" Step 1: Connecting to MySQL Database and fetching records")
try:
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="Your_Mysql_Password",
        database="telecom_db"
    )
    query = "SELECT gender, SeniorCitizen, Partner, tenure, MonthlyCharges, TotalCharges, Contract, Churn FROM churn_data"
    df = pd.read_sql(query, conn)
    conn.close()
    print(" Data successfully loaded from MySQL relational database.")
except Exception as e:
    print(f" Database Connection Error: {e}")
    print(" Please make sure your MySQL server is running and 'db_import.py' was executed.")
    exit()

# 3. Step 2: Data Preprocessing and Cleaning
print("\n Step 2: Handling missing values and cleaning columns...")
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(' ', None), errors='coerce')
median_value = df['TotalCharges'].median()
df['TotalCharges'] = df['TotalCharges'].fillna(median_value)

# 4. Step 3: Categorical Feature Encoding
print(" Step 3: Transforming text fields into numerical matrices...")
encoders = {}
for col in ['gender', 'Partner', 'Contract']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

X = df.drop(columns=['Churn'])
y = df['Churn']

# 5. Step 4: Resolving Data Imbalance natively using SMOTE
print(" Step 4: Applying SMOTE to balance target classes ")
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# 6. Step 5: Machine Learning Model Training
print(" Step 5: Training Random Forest Classifier model on 80% dataset")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 7. Step 6: Pipeline Evaluation Analysis
print("\n Final ML Pipeline Performance Analysis ")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# 8. Step 7: Local Simulation Testing (Testing on a dummy customer data)
print("\n Step 6: Simulating a Real-Time Prediction Scenario...")
dummy_customer = {
    'gender': 'Female', 'SeniorCitizen': 0, 'Partner': 'No',
    'Contract': 'Month-to-month', 'tenure': 2, 'MonthlyCharges': 85.50, 'TotalCharges': 171.00
}

# Apply identical processing transforms to the standalone input
input_data = pd.DataFrame([{
    'gender': encoders['gender'].transform([dummy_customer['gender']])[0],
    'SeniorCitizen': dummy_customer['SeniorCitizen'],
    'Partner': encoders['Partner'].transform([dummy_customer['Partner']])[0],
    'tenure': dummy_customer['tenure'],
    'MonthlyCharges': dummy_customer['MonthlyCharges'],
    'TotalCharges': dummy_customer['TotalCharges'],
    'Contract': encoders['Contract'].transform([dummy_customer['Contract']])[0]
}])

prediction = model.predict(input_data)
probability = model.predict_proba(input_data)[0][1] * 100

print(f" Customer Context: 2 Months Tenure, High Monthly Bills, Month-to-Month Contract.")
print(f" Simulated Churn Risk Probability: {probability:.2f}%")

if prediction[0] == 1:
    print(" Strategy Alert: High Churn Risk! Proactive retention offer suggested.")
else:
    print(" Strategy Alert: Low Churn Risk! Normal automated customer lifecycle.")


print(" PIPELINE COMPLETED SUCCESSFULLY WITHOUT ANY ERROR")
