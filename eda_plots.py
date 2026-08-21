import mysql.connector
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# 1. Fetch data from MySQL using pandas read_sql
conn = mysql.connector.connect(host="localhost", user="root", password="Sneha@963", database="telecom_db")
df = pd.read_sql("SELECT * FROM churn_data", conn)
conn.close()

# 2. Data Cleaning: Convert TotalCharges to numeric and handle blank spaces
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(' ', None), errors='coerce')
median_value = df['TotalCharges'].median()
df['TotalCharges'] = df['TotalCharges'].fillna(median_value)

# 3. Data Visualization: Generate Churn Count plot
plt.figure(figsize=(6,4))
sns.countplot(x='Churn', data=df, hue='Churn', palette='Set2', legend=False)
plt.title('Customer Count: Churn vs Retained')
plt.savefig('churn_count.png')
plt.close()

print(" Success! EDA plots generated and saved as 'churn_count.png' in the project folder!")
