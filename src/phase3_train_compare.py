"""Phase 3: train LightGBM and XGBoost on ALL features, compare on the 20% test set."""
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # draw pictures without opening a window
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

lines = []
def say(text=""):
    print(text)
    lines.append(str(text))

# ---- load the Phase 2 files ----
X_train = pd.read_csv(DATA / "X_train.csv")
X_test = pd.read_csv(DATA / "X_test.csv")
y_train = pd.read_csv(DATA / "y_train.csv").iloc[:, 0].values
y_test = pd.read_csv(DATA / "y_test.csv").iloc[:, 0].values
class_names = list(joblib.load(ROOT / "models" / "encoders.joblib")["label_encoder"].classes_)
labels = list(range(len(class_names)))
say(f"Train: {X_train.shape}   Test: {X_test.shape}   Classes: {len(class_names)}")

# ---- the two models (sensible settings, no tuning) ----
models = {
    "LightGBM": LGBMClassifier(n_estimators=200, learning_rate=0.1,
                               random_state=42, n_jobs=-1, verbose=-1),
    "XGBoost": XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=6,
                             tree_method="hist", random_state=42, n_jobs=-1,
                             eval_metric="mlogloss"),
}

# ---- which test rows are NOT also in train? (duplicate check from Phase 2) ----
train_hash = set(pd.util.hash_pandas_object(X_train, index=False))
unseen = ~pd.util.hash_pandas_object(X_test, index=False).isin(train_hash).values
say(f"Test rows with no copy in train: {unseen.sum()} of {len(X_test)}")

def save_confusion_png(cm, title, path):
    row_share = cm / cm.sum(axis=1, keepdims=True)       # colour = share of each true class
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.imshow(row_share, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(labels); ax.set_yticks(labels)
    ax.set_xticklabels(class_names, rotation=90, fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    for i in labels:
        for j in labels:
            if cm[i, j] > 0:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                        color="white" if row_share[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)

summary_rows, per_class = [], {}
for name, model in models.items():
    say(f"\n================ {name} ================")
    start = time.time()
    model.fit(X_train, y_train)
    train_secs = time.time() - start
    say(f"Training time: {train_secs:.1f} seconds")

    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)
    proba = proba / proba.sum(axis=1, keepdims=True)      # make rows sum to exactly 1

    row = {
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "precision_macro": precision_score(y_test, pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_test, pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_test, pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_test, pred, average="weighted", zero_division=0),
        "auc_roc_macro_ovr": roc_auc_score(y_test, proba, multi_class="ovr",
                                           average="macro", labels=labels),
        "accuracy_unseen_rows_only": accuracy_score(y_test[unseen], pred[unseen]),
        "f1_macro_unseen_rows_only": f1_score(y_test[unseen], pred[unseen],
                                              average="macro", zero_division=0),
        "train_seconds": train_secs,
    }
    summary_rows.append(row)
    for k, v in row.items():
        if k != "model":
            say(f"{k}: {v:.4f}")

    # per-class results
    report = classification_report(y_test, pred, labels=labels, target_names=class_names,
                                   output_dict=True, zero_division=0)
    pc = pd.DataFrame(report).T.loc[class_names, ["precision", "recall", "f1-score", "support"]]
    per_class[name] = pc
    say("\nPer-class results:")
    say(pc.round(4).to_string())
    pc.to_csv(REPORTS / f"phase3_per_class_{name}.csv")

    # confusion matrix: CSV + picture + top mistakes in words
    cm = confusion_matrix(y_test, pred, labels=labels)
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
        REPORTS / f"phase3_confusion_{name}.csv")
    save_confusion_png(cm, f"{name} confusion matrix (test set)",
                       REPORTS / f"phase3_confusion_{name}.png")
    errors = cm.copy(); np.fill_diagonal(errors, 0)
    say("\nBiggest mistakes (true class -> predicted class: count):")
    flat = sorted(((errors[i, j], i, j) for i in labels for j in labels if errors[i, j] > 0),
                  reverse=True)[:5]
    say("\n".join(f"  {class_names[i]} -> {class_names[j]}: {c}" for c, i, j in flat)
        if flat else "  none")

# ---- side-by-side comparison ----
summary = pd.DataFrame(summary_rows).set_index("model")
summary.to_csv(REPORTS / "phase3_comparison.csv")
say("\n================ SIDE-BY-SIDE ================")
say(summary.round(4).T.to_string())

say("\n--- The two rare classes (recall, test rows) ---")
for rare in ["Metasploit_Brute_Force_SSH", "NMAP_FIN_SCAN"]:
    for name in models:
        r = per_class[name].loc[rare]
        say(f"{rare} | {name}: recall={r['recall']:.3f}, f1={r['f1-score']:.3f}, "
            f"test rows={int(r['support'])}")

# ---- choose the winner: higher macro F1, tie-break on accuracy ----
winner = summary.sort_values(["f1_macro", "accuracy"], ascending=False).index[0]
say(f"\nWINNER (highest macro F1, then accuracy): {winner}")

# ---- leakage warning ----
if (summary["accuracy"] > 0.99).any():
    say("\nWARNING: accuracy is above 99%. Checks so far: 'no' column dropped; "
        "6.75% of test rows have a copy in train. Compare 'accuracy_unseen_rows_only' "
        "to 'accuracy' above before trusting these scores.")

(REPORTS / "phase3_report.txt").write_text("\n".join(lines), encoding="utf-8")
say("\nSaved everything in reports/ (phase3_*)")