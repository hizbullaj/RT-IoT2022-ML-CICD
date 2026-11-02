
from pathlib import Path
import json
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# Load model bundle
bundle = joblib.load(ROOT / "backend/models/model_bundle.joblib")

MODEL = bundle["model"]
FEATURES = bundle["feature_names"]
LABEL_ENCODER = bundle["label_encoder"]
CLASS_NAMES = bundle["class_names"]

# Read JSON payload
with open(ROOT / "backend/test_payload.json", encoding="utf-8") as f:
    payload = json.load(f)

# Check feature names
missing = [f for f in FEATURES if f not in payload]
extra = [f for f in payload if f not in FEATURES]

if missing or extra:
    raise ValueError(
        f"Payload mismatch. Missing: {missing}, Extra: {extra}"
    )

# Arrange data in the exact model feature order
frame = pd.DataFrame(
    [[payload[f] for f in FEATURES]],
    columns=FEATURES
)

# Predict
prediction = MODEL.predict(frame)[0]

# Convert predicted number into attack name
predicted_class = LABEL_ENCODER.inverse_transform(
    [int(prediction)]
)[0]

print("\n========== MODEL PREDICTION ==========")
print("Model:", bundle["model_name"])
print("Predicted attack:", predicted_class)

# Show probabilities for each class
if hasattr(MODEL, "predict_proba"):
    probabilities = MODEL.predict_proba(frame)[0]

    print("\n========== CLASS PROBABILITIES ==========")

    for name, probability in zip(CLASS_NAMES, probabilities):
        print(f"{name}: {probability:.4f}")

print("\nPASS: Independent model prediction successful.")
