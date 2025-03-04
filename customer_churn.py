import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = pd.read_csv("customer_churn.csv")
if 'Churn' in data.columns:
    data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})

# Preprocessing: Identify categorical & numerical columns
categorical_columns = ['gender']  # Update based on dataset
numerical_columns = ['MonthlyCharges', 'TotalCharges']

# One-Hot Encode categorical features
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_data = encoder.fit_transform(data[categorical_columns])

print("Checking for missing values:")
print(data[numerical_columns].isnull().sum())

print("\nChecking for non-numeric values:")
print(data[numerical_columns].applymap(lambda x: isinstance(x, str)).sum())

for col in numerical_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')  # Convert to float, set errors as NaN

data[numerical_columns] = data[numerical_columns].fillna(data[numerical_columns].median())

# Scale numerical features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[numerical_columns])

# Prepare training data
X = np.hstack((encoded_data, scaled_data))
y = data["Churn"].values  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model and preprocessing objects
joblib.dump(model, "churn_model.pkl")
joblib.dump(encoder, "encoder.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model and preprocessing files saved successfully!")
