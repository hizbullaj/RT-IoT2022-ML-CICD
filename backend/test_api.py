import requests

BASE_URL = "http://127.0.0.1:5000"

session = requests.Session()
session.trust_env = False

# Test 1: Health endpoint
response = session.get(f"{BASE_URL}/health")
print("Health Status:", response.status_code)
print(response.json())

# Test 2: Prediction endpoint
with open("backend/test_payload.json", "r") as file:
    payload = __import__("json").load(file)

response = session.post(
    f"{BASE_URL}/predict",
    json=payload
)

print("\nPrediction Status:", response.status_code)
print(response.json())