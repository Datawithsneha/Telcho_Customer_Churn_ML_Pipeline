import mysql.connector
import pandas as pd
import pickle
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# 1. Suppress unnecessary warning messages for clean output console
warnings.filterwarnings('ignore')

print(" Fetching raw data from MySQL server ")

# 2. Establish connection with MySQL database and load targeted features
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Your_Mysql_Password",
    database="telecom_db"
)
query = "SELECT gender, SeniorCitizen, Partner, tenure, MonthlyCharges, TotalCharges, Contract, Churn FROM churn_data"
df = pd.read_sql(query, conn)
conn.close()

# 3. Data Cleaning: Handle missing/blank spaces in TotalCharges column (Pandas 3.0 Safe)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(' ', None), errors='coerce')
median_value = df['TotalCharges'].median()
df['TotalCharges'] = df['TotalCharges'].fillna(median_value)

# 4. Feature Encoding: Convert categorical text variables into integer categories
encoders = {}
for col in ['gender', 'Partner', 'Contract']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Map the target variable 'Churn' to binary format (Yes=1, No=0)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Splitting independent features (X) and dependent target variable (y)
X = df.drop(columns=['Churn'])
y = df['Churn']

print(" Balancing dataset classes using SMOTE technique ")
# 5. Apply SMOTE to handle class imbalance natively without bias
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# 6. Train-Test Split (80% Training Data, 20% Testing Data)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

print(" Initializing Machine Learning model training")
# 7. Model Training using Random Forest Classifier algorithm
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 8. Model Evaluation: Generate classification analytics metrics
y_pred = model.predict(X_test)
print("\n Machine Learning Model Performance Report ")
print(classification_report(y_test, y_pred))

# 9. Serialization: Export trained model matrix and label encoders to binary pickle files
with open("churn_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

print(" Success! Machine Learning model has been saved as 'churn_model.pkl' in the project workspace.")
