import mysql.connector
import pandas as pd

# 1. Read the CSV file placed inside the project folder
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# 2. Establish connection with MySQL server
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Sneha@963"
)
cursor = conn.cursor()
cursor.execute("USE telecom_db")

# 3. Drop table if it already exists and create a new table structure
cursor.execute("DROP TABLE IF EXISTS churn_data")
create_table_query = """
CREATE TABLE churn_data (
    customerID VARCHAR(50), gender VARCHAR(10), SeniorCitizen INT, Partner VARCHAR(10),
    Dependents VARCHAR(10), tenure INT, PhoneService VARCHAR(10), MultipleLines VARCHAR(20),
    InternetService VARCHAR(20), OnlineSecurity VARCHAR(20), OnlineBackup VARCHAR(20),
    DeviceProtection VARCHAR(20), TechSupport VARCHAR(20), StreamingTV VARCHAR(20),
    StreamingMovies VARCHAR(20), Contract VARCHAR(20), PaperlessBilling VARCHAR(10),
    PaymentMethod VARCHAR(40), MonthlyCharges FLOAT, TotalCharges VARCHAR(20), Churn VARCHAR(10)
)
"""
cursor.execute(create_table_query)

# 4. Load the data into the MySQL table row by row
for _, row in df.iterrows():
    sql = "INSERT INTO churn_data VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
    cursor.execute(sql, tuple(row))

conn.commit()
print(" Success! Data has been successfully imported into MySQL database!")
cursor.close()
conn.close()

