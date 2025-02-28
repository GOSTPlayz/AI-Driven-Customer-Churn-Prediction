import requests

url = "https://deu9qakbo8.execute-api.us-east-1.amazonaws.com/prod/predict"

# ✅ Ensure the payload is correctly formatted
data = {"features": [0, 1, 80.5, 500.0]}  # Example input
headers = {"Content-Type": "application/json"}

response = requests.post(url, json=data, headers=headers)
print(response.json())

