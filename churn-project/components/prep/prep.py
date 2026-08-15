import argparse
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

parser = argparse.ArgumentParser()
parser.add_argument("--input_data", type=str)
parser.add_argument("--output_data", type=str)
args = parser.parse_args()

df = pd.read_csv(args.input_data)

# Drop the customer ID - it's not a feature
df = df.drop(columns=["customerID"], errors="ignore")

# TotalCharges sometimes has blank strings - coerce and drop bad rows
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()

# Encode the label
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Label-encode remaining categorical columns
cat_cols = df.select_dtypes(include="object").columns
for col in cat_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

Path(args.output_data).mkdir(parents=True, exist_ok=True)
df.to_csv(Path(args.output_data) / "prepped-churn.csv", index=False)

