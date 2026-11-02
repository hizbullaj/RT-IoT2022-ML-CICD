"""Phase 3b: is LightGBM broken at its first settings? Try 3 gentler ones."""
from pathlib import Path
import time
import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
X_train = pd.read_csv(DATA / "X_train.csv")
X_test = pd.read_csv(DATA / "X_test.csv")
y_train = pd.read_csv(DATA / "y_train.csv").iloc[:, 0].values
y_test = pd.read_csv(DATA / "y_test.csv").iloc[:, 0].values
class_names = list(joblib.load(ROOT / "models" / "encoders.joblib")["label_encoder"].classes_)
labels = list(range(len(class_names)))
rare = {"Metasploit": class_names.index("Metasploit_Brute_Force_SSH"),
        "FIN_SCAN": class_names.index("NMAP_FIN_SCAN")}

configs = {
    "A: lr=0.05, 300 trees": dict(n_estimators=300, learning_rate=0.05),
    "B: lr=0.02, 500 trees": dict(n_estimators=500, learning_rate=0.02),
    "C: lr=0.05, 300 trees, min_child_samples=5": dict(n_estimators=300, learning_rate=0.05,
                                                       min_child_samples=5),
}

rows = []
for name, params in configs.items():
    start = time.time()
    model = LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1, **params)
    model.fit(X_train, y_train)
    secs = time.time() - start
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)
    proba = proba / proba.sum(axis=1, keepdims=True)
    rec = recall_score(y_test, pred, average=None, labels=labels, zero_division=0)
    rows.append({
        "config": name,
        "train_accuracy": accuracy_score(y_train, model.predict(X_train)),
        "test_accuracy": accuracy_score(y_test, pred),
        "f1_macro": f1_score(y_test, pred, average="macro", zero_division=0),
        "auc_roc_macro_ovr": roc_auc_score(y_test, proba, multi_class="ovr",
                                           average="macro", labels=labels),
        "recall_Metasploit": rec[rare["Metasploit"]],
        "recall_FIN_SCAN": rec[rare["FIN_SCAN"]],
        "classes_with_zero_recall": int((rec == 0).sum()),
        "seconds": secs,
    })
    print(f"done: {name} ({secs:.1f}s)")

result = pd.DataFrame(rows).set_index("config")
print("\n", result.round(4).T.to_string())
(ROOT / "reports").mkdir(exist_ok=True)
result.to_csv(ROOT / "reports" / "phase3b_lightgbm_fix.csv")
print("\nSaved reports/phase3b_lightgbm_fix.csv")