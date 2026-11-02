"""Phase 1: look at the data. Changes nothing, only reports."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
csv_files = list((ROOT / "data" / "raw").glob("*.csv"))
if len(csv_files) != 1:
    raise SystemExit(f"Expected exactly 1 CSV in data/raw, found {len(csv_files)}")

df = pd.read_csv(csv_files[0])
out = []                      # we collect lines, then print and save them
def say(text=""):
    print(text)
    out.append(str(text))

say(f"File: {csv_files[0].name}")
say(f"Shape (rows, columns): {df.shape}")

say("\n--- Columns and types ---")
say(df.dtypes.to_string())

say("\n--- Target: Attack_type class counts ---")
if "Attack_type" in df.columns:
    say(df["Attack_type"].value_counts().to_string())
else:
    say("WARNING: no column named Attack_type. Columns are listed above.")

say("\n--- Missing values (only columns with any) ---")
missing = df.isna().sum()
say(missing[missing > 0].to_string() if (missing > 0).any() else "None")

say("\n--- Infinite values (only columns with any) ---")
num = df.select_dtypes(include=[np.number])
inf_counts = np.isinf(num).sum()
say(inf_counts[inf_counts > 0].to_string() if (inf_counts > 0).any() else "None")

say("\n--- Text columns and number of unique values ---")
for col in df.select_dtypes(exclude=[np.number]).columns:
    say(f"{col}: {df[col].nunique()} unique")

say("\n--- Suspicious index/ID columns ---")
found = False
for col in df.columns:
    name_hint = col.lower().startswith("unnamed") or col.lower() in ("no", "id", "index")
    is_counter = False
    if pd.api.types.is_integer_dtype(df[col]) and df[col].is_unique:
        is_counter = df[col].is_monotonic_increasing   # 0,1,2,3... style
    if name_hint or is_counter:
        found = True
        say(f"{col}: name_hint={name_hint}, unique_increasing_integers={is_counter}")
if not found:
    say("None found")

say("\n--- Duplicate rows ---")
say(f"Exact duplicate rows: {df.duplicated().sum()}")

(ROOT / "reports").mkdir(exist_ok=True)
(ROOT / "reports" / "phase1_data_check.txt").write_text("\n".join(out), encoding="utf-8")
say("\nSaved to reports/phase1_data_check.txt")