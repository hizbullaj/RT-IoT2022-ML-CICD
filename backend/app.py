
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

# Project paths
BASE = Path(__file__).resolve().parent

# Load the saved model bundle
BUNDLE = joblib.load(BASE / "models" / "model_bundle.joblib")

MODEL = BUNDLE["model"]
FEATURES = BUNDLE["feature_names"]
LABEL_ENCODER = BUNDLE["label_encoder"]
CLASS_NAMES = BUNDLE["class_names"]
CATEGORY_MAPS = BUNDLE["category_maps"]
FEATURE_INFO = BUNDLE["feature_info"]

app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "model": BUNDLE["model_name"],
        "features": len(FEATURES),
        "classes": len(CLASS_NAMES)
    })


@app.post("/predict")
def predict():

    data = request.get_json(silent=True)

    # Check JSON input
    if not isinstance(data, dict):
        return jsonify({
            "error": "Expected a JSON object"
        }), 400

    # Check missing and extra features
    missing = [f for f in FEATURES if f not in data]
    extra = [f for f in data if f not in FEATURES]

    if missing or extra:
        return jsonify({
            "error": "Input schema mismatch",
            "missing": missing,
            "extra": extra
        }), 400

    try:
        values = {}

        for feature in FEATURES:

            value = data[feature]

            if value is None:
                return jsonify({
                    "error": f"{feature} cannot be empty"
                }), 400

            # Convert service name into its saved numeric code
            if feature == "service" and isinstance(value, str):
                service_map = CATEGORY_MAPS["service"]

                if value not in service_map:
                    return jsonify({
                        "error": "Invalid service category",
                        "allowed": FEATURE_INFO["service"]["choices"]
                    }), 400

                value = service_map[value]

            # Validate numeric features
            try:
                value = float(value)
            except (ValueError, TypeError):
                return jsonify({
                    "error": f"{feature} must be numeric"
                }), 400

            if pd.isna(value):
                return jsonify({
                    "error": f"{feature} must be a valid number"
                }), 400

            # Check values against saved feature information
            info = FEATURE_INFO.get(feature, {})

            if info.get("type") == "number":
                if value < info["min"] or value > info["max"]:
                    return jsonify({
                        "error": f"{feature} is outside the allowed range",
                        "minimum": info["min"],
                        "maximum": info["max"]
                    }), 400

            values[feature] = value

        # Prepare input in the exact model feature order
        frame = pd.DataFrame(
            [[values[f] for f in FEATURES]],
            columns=FEATURES
        )

        # Make prediction
        raw_prediction = MODEL.predict(frame)[0]

        predicted_class = LABEL_ENCODER.inverse_transform(
            [int(raw_prediction)]
        )[0]

        result = {
            "prediction": str(predicted_class),
            "model": BUNDLE["model_name"],
            "task": "classification",
            "disclaimer": "Research and educational prototype only."
        }

        # Calculate class probabilities
        if hasattr(MODEL, "predict_proba"):
            probabilities = MODEL.predict_proba(frame)[0]

            result["probabilities"] = {
                str(name): round(float(probability), 6)
                for name, probability in zip(CLASS_NAMES, probabilities)
            }

        return jsonify(result), 200

    except (ValueError, TypeError, KeyError) as exc:
        return jsonify({
            "error": "Invalid input",
            "detail": str(exc)
        }), 400

    except Exception:
        app.logger.exception("Prediction failed")

        return jsonify({
            "error": "Prediction processing failed"
        }), 500


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
