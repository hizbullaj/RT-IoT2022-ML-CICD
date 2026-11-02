"""Phase 2: clean, check for leaks, split 80/20, encode. Saves results."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

ROOT = Path(__file__).resolve().parent.parent
RAW = next((ROOT / "data" / "raw").glob("*.csv"))
OUT_DATA = ROOT / "data" / "processed"
OUT_MODELS = ROOT / "models"
OUT_REPORTS = ROOT / "reports"
for folder in (OUT_DATA, OUT_MODELS, OUT_REPORTS):
    folder.mkdir(parents=True, exist_ok=True)

lines = []
def say(text=""):
    print(text)
    lines.append(str(text))

TARGET = "Attack_type"
TEXT_COLS = ["proto", "service"]

df = pd.read_csv(RAW)
say(f"Loaded: {df.shape}")

# 1) Is the 'no' column a leak? If each attack type lives in its own
#    range of 'no', the model could cheat by reading it.
say("\n--- 'no' column: min / max / count per class ---")
say(df.groupby(TARGET)["no"].agg(["min", "max", "count"]).to_string())
say(f"'no' unique values: {df['no'].nunique()} of {len(df)} rows")

# 2) Remove inf / NaN
before = len(df)
df = df.replace([np.inf, -np.inf], np.nan).dropna()
say(f"\nRows removed for NaN/inf: {before - len(df)}")

# 3) Drop the ID column. Target is separated from the features.
y_text = df[TARGET]
X = df.drop(columns=["no", TARGET])
say(f"Features after dropping 'no' and target: {X.shape[1]}")

# 4) Duplicates now that 'no' is gone
dup_mask = X.duplicated(keep=False)
say(f"\nRows that have an identical twin (features only): {dup_mask.sum()}")
dup_df = X[dup_mask].copy()
dup_df["y"] = y_text[dup_mask]
conflict = (dup_df.groupby(list(X.columns))["y"].nunique() > 1).sum()
say(f"Duplicate groups with CONFLICTING labels: {conflict}")

# 5) ONE stratified split, 80/20, random_state=42
X_train, X_test, y_train_text, y_test_text = train_test_split(
    X, y_text, test_size=0.20, stratify=y_text, random_state=42
)
say(f"\nTrain rows: {len(X_train)}   Test rows: {len(X_test)}")

# 6) Leak check: identical feature rows in both train and test
train_hash = set(pd.util.hash_pandas_object(X_train, index=False))
test_hash = pd.util.hash_pandas_object(X_test, index=False)
shared = int(test_hash.isin(train_hash).sum())
say(f"Test rows whose features also appear in train: {shared} "
    f"of {len(X_test)} ({100 * shared / len(X_test):.2f}%)")

# 7) Class counts in each split
counts = pd.DataFrame({
    "train": y_train_text.value_counts(),
    "test": y_test_text.value_counts(),
})
say("\n--- Class counts per split ---")
say(counts.to_string())

# 8) Encode. Encoders learn from TRAIN only.
text_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_train = X_train.copy()
X_test = X_test.copy()
X_train[TEXT_COLS] = text_enc.fit_transform(X_train[TEXT_COLS])
X_test[TEXT_COLS] = text_enc.transform(X_test[TEXT_COLS])

label_enc = LabelEncoder()
y_train = pd.Series(label_enc.fit_transform(y_train_text), name=TARGET, index=X_train.index)
y_test = pd.Series(label_enc.transform(y_test_text), name=TARGET, index=X_test.index)

say("\n--- Text column codes ---")
for col, cats in zip(TEXT_COLS, text_enc.categories_):
    say(f"{col}: {dict((c, i) for i, c in enumerate(cats))}")
say("\n--- Target codes ---")
say(str({c: i for i, c in enumerate(label_enc.classes_)}))

# 9) Save everything
X_train.to_csv(OUT_DATA / "X_train.csv", index=False)
X_test.to_csv(OUT_DATA / "X_test.csv", index=False)
y_train.to_csv(OUT_DATA / "y_train.csv", index=False)
y_test.to_csv(OUT_DATA / "y_test.csv", index=False)
joblib.dump(
    {"text_encoder": text_enc, "label_encoder": label_enc,
     "text_cols": TEXT_COLS, "feature_names": list(X_train.columns)},
    OUT_MODELS / "encoders.joblib",
)
(OUT_REPORTS / "phase2_split_summary.txt").write_text("\n".join(lines), encoding="utf-8")
say("\nSaved data/processed/*.csv, models/encoders.joblib, reports/phase2_split_summary.txt")