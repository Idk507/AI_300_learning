import argparse, joblib, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

parser = argparse.ArgumentParser()
parser.add_argument("--training_data", type=str)
parser.add_argument("--model_output", type=str)
args = parser.parse_args()

df = pd.read_csv(Path(args.training_data) / "prepped-churn.csv")
X, y = df.drop(columns=["Churn"]), df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_train, y_train)

Path(args.model_output).mkdir(parents=True, exist_ok=True)
joblib.dump(model, Path(args.model_output) / "model.pkl")
X_test.assign(Churn=y_test).to_csv(Path(args.model_output) / "test-set.csv", index=False)
