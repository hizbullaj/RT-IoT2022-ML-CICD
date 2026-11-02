
from pathlib import Path
import json
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# Load our saved model bundle
bundle = joblib.load(ROOT / "backend/models/model_bundle.joblib")

# Get the exact 15 features
features = bundle["feature_names"]

# Read the sample CSV
df = pd.read_csv(ROOT / "data/sample_inputs.csv")

# Check that all required features exist
missing = [f for f in features if f not in df.columns]

if missing:
    raise ValueError(f"Missing features: {missing}")

# Select the first sample row in correct feature order
row = df.loc[0, features]

# Convert row into JSON-compatible values
payload = {}

for key, value in row.items():
    if pd.isna(value):
        payload[key] = None
    elif hasattr(value, "item"):
        payload[key] = value.item()
    else:
        payload[key] = value

# Save JSON payload
output = ROOT / "backend/test_payload.json"

output.write_text(
    json.dumps(payload, indent=2),
    encoding="utf-8"
)

print("JSON payload created successfully.")
print("Number of features:", len(payload))
print("Saved at:", output)
