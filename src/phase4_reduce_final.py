"""Phase 4: rank features, retrain on the top 10/12/15, pick the final model, save the bundle."""
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report)

ROOT = Path(__file__).resolve().parent.parent
DATA, MODELS, REPORTS = ROOT / "data" / "processed", ROOT / "models", ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

lines = []
def say(text=""):
    print(text)
    lines.append(str(text))

# ---- load Phase 2 files ----
X_train = pd.read_csv(DATA / "X_train.csv")
X_test = pd.read_csv(DATA / "X_test.csv")
y_train = pd.read_csv(DATA / "y_train.csv").iloc[:, 0].values
y_test = pd.read_csv(DATA / "y_test.csv").iloc[:, 0].values
enc = joblib.load(MODELS / "encoders.joblib")
class_names = list(enc["label_encoder"].classes_)
labels = list(range(len(class_names)))
RARE = ["Metasploit_Brute_Force_SSH", "NMAP_FIN_SCAN"]
KS = [10, 12, 15]
say(f"Train: {X_train.shape}   Test: {X_test.shape}")

def make_model(name):
    if name == "LightGBM":
        return LGBMClassifier(n_estimators=500, learning_rate=0.02,
                              random_state=42, n_jobs=-1, verbose=-1)
    return XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=6,
                         tree_method="hist", random_state=42, n_jobs=-1,
                         eval_metric="mlogloss")

def evaluate(model, cols):
    """Train on `cols`, score on the 20% test set. Returns metrics + predictions."""
    start = time.time()
    model.fit(X_train[cols], y_train)
    secs = time.time() - start
    pred = model.predict(X_test[cols])
    proba = model.predict_proba(X_test[cols])
    proba = proba / proba.sum(axis=1, keepdims=True)
    # test rows whose (these) features never appear in train
    seen = set(pd.util.hash_pandas_object(X_train[cols], index=False))
    unseen = ~pd.util.hash_pandas_object(X_test[cols], index=False).isin(seen).values
    rec = recall_score(y_test, pred, average=None, labels=labels, zero_division=0)
    m = {
        "accuracy": accuracy_score(y_test, pred),
        "precision_macro": precision_score(y_test, pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, pred, average="macro", zero_division=0),
        "auc_roc_macro_ovr": roc_auc_score(y_test, proba, multi_class="ovr",
                                           average="macro", labels=labels),
        "accuracy_unseen_only": accuracy_score(y_test[unseen], pred[unseen]),
        "unseen_rows": int(unseen.sum()),
        "recall_Metasploit": rec[class_names.index(RARE[0])],
        "recall_FIN_SCAN": rec[class_names.index(RARE[1])],
        "zero_recall_classes": int((rec == 0).sum()),
        "train_seconds": secs,
    }
    return m, pred

# ---- 1) rank features with LightGBM (gain), TRAINING data only ----
say("\n=== Step 1: ranking features with LightGBM (training data only) ===")
rank_model = LGBMClassifier(n_estimators=500, learning_rate=0.02, importance_type="gain",
                            random_state=42, n_jobs=-1, verbose=-1)
rank_model.fit(X_train, y_train)
imp = pd.Series(rank_model.feature_importances_, index=X_train.columns)
imp = (imp / imp.sum()).sort_values(ascending=False)
imp.to_csv(REPORTS / "phase4_feature_ranking.csv", header=["gain_share"])
say("Top 20 features (share of total gain):")
say(imp.head(20).round(4).to_string())
say(f"Top 10 together hold {imp.head(10).sum():.1%} of total gain.")
for port in ["id.orig_p", "id.resp_p"]:
    if port in imp.index[:5]:
        say(f"NOTE: {port} is in the top 5. Port numbers can carry tool-specific "
            f"patterns, so treat that with some care.")

# ---- 2) train both models on all features and on top 10/12/15 ----
all_cols = list(X_train.columns)
results, fitted, preds = {}, {}, {}
for name in ["LightGBM", "XGBoost"]:
    for k in ["all"] + KS:
        cols = all_cols if k == "all" else list(imp.index[:k])
        say(f"\nTraining {name} on {'all 83' if k == 'all' else 'top ' + str(k)} features ...")
        m, pred = evaluate(make_model(name), cols)
        results[(name, k)], preds[(name, k)] = m, pred
        fitted[(name, k)] = (make_model(name), cols)   # placeholder, replaced below
        say(f"   done in {m['train_seconds']:.1f}s  accuracy={m['accuracy']:.4f}  "
            f"macro F1={m['f1_macro']:.4f}")

# ---- 3) comparison table + change versus all features ----
rows = []
for (name, k), m in results.items():
    base = results[(name, "all")]
    rows.append({"model": name, "features": 83 if k == "all" else k, **m,
                 "d_accuracy_vs_all": m["accuracy"] - base["accuracy"],
                 "d_f1_macro_vs_all": m["f1_macro"] - base["f1_macro"]})
table = pd.DataFrame(rows)
table.to_csv(REPORTS / "phase4_comparison.csv", index=False)
say("\n=== Comparison (test set) ===")
say(table.drop(columns=["train_seconds"]).round(4).to_string(index=False))

# ---- 4) choose the feature count, then the model ----
chosen_k = None
for k in KS:
    if all(results[(n, k)]["f1_macro"] >= results[(n, "all")]["f1_macro"] - 0.01
           for n in ["LightGBM", "XGBoost"]):
        chosen_k = k
        break
if chosen_k is None:
    chosen_k = 15
    say("\nNo feature count stayed within 0.01 macro F1 for both models. Using 15 anyway.")
else:
    say(f"\nSmallest feature count within 0.01 macro F1 for both models: {chosen_k}")
final_name = sorted(["LightGBM", "XGBoost"],
                    key=lambda n: (results[(n, chosen_k)]["f1_macro"],
                                   results[(n, chosen_k)]["accuracy"]), reverse=True)[0]
final_cols = list(imp.index[:chosen_k])
say(f"FINAL MODEL: {final_name} on {chosen_k} features")
say("Features: " + ", ".join(final_cols))

# ---- 5) refit the final model once more and save everything ----
final_model = make_model(final_name)
final_metrics, final_pred = evaluate(final_model, final_cols)
say("\n=== Final model: all six metrics (macro-averaged) ===")
for k_, v in final_metrics.items():
    say(f"{k_}: {v:.4f}" if isinstance(v, float) else f"{k_}: {v}")
if final_metrics["accuracy"] > 0.99:
    say("\nWARNING: accuracy is above 99%. Checks done: 'no' dropped; duplicate overlap "
        f"measured. Accuracy on unseen-only rows = {final_metrics['accuracy_unseen_only']:.4f} "
        f"({final_metrics['unseen_rows']} rows). Compare it with the accuracy above.")

rep = classification_report(y_test, final_pred, labels=labels, target_names=class_names,
                            output_dict=True, zero_division=0)
pc = pd.DataFrame(rep).T.loc[class_names, ["precision", "recall", "f1-score", "support"]]
pc.to_csv(REPORTS / "phase4_final_per_class.csv")
say("\nPer-class results (final model):")
say(pc.round(4).to_string())

cm = confusion_matrix(y_test, final_pred, labels=labels)
pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
    REPORTS / "phase4_final_confusion.csv")
share = cm / cm.sum(axis=1, keepdims=True)
fig, ax = plt.subplots(figsize=(10, 9))
ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(labels); ax.set_yticks(labels)
ax.set_xticklabels(class_names, rotation=90, fontsize=8)
ax.set_yticklabels(class_names, fontsize=8)
for i in labels:
    for j in labels:
        if cm[i, j] > 0:
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                    color="white" if share[i, j] > 0.5 else "black")
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title(f"{final_name}, {chosen_k} features (test set)")
fig.tight_layout(); fig.savefig(REPORTS / "phase4_final_confusion.png", dpi=150); plt.close(fig)

# what the web form needs: choices for text fields, min/median/max for numbers
text_cols = enc["text_cols"]
category_maps = {c: {str(v): i for i, v in enumerate(cats)}
                 for c, cats in zip(text_cols, enc["text_encoder"].categories_)}
feature_info = {}
for c in final_cols:
    if c in category_maps:
        feature_info[c] = {"type": "category", "choices": list(category_maps[c].keys())}
    else:
        col = X_train[c]
        feature_info[c] = {"type": "number", "min": float(col.min()),
                           "median": float(col.median()), "max": float(col.max())}

bundle = {
    "model": final_model,
    "model_name": final_name,
    "feature_names": final_cols,            # exact order the model expects
    "class_names": class_names,
    "label_encoder": enc["label_encoder"],
    "text_encoder": enc["text_encoder"],
    "text_cols": text_cols,
    "category_maps": category_maps,
    "feature_info": feature_info,
}
joblib.dump(bundle, MODELS / "final_bundle.joblib")
(REPORTS / "phase4_report.txt").write_text("\n".join(lines), encoding="utf-8")
say("\nSaved models/final_bundle.joblib and reports/phase4_* files")