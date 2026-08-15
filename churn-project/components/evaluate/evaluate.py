import argparse, joblib, json, pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score

parser = argparse.ArgumentParser()
parser.add_argument("--model_input", type=str)
parser.add_argument("--metrics_output", type=str)
args = parser.parse_args()

model = joblib.load(Path(args.model_input) / "model.pkl")
test_df = pd.read_csv(Path(args.model_input) / "test-set.csv")
X_test, y_test = test_df.drop(columns=["Churn"]), test_df["Churn"]

preds = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]

metrics = {
    "accuracy": accuracy_score(y_test, preds),
    "auc": roc_auc_score(y_test, probs),
}

Path(args.metrics_output).mkdir(parents=True, exist_ok=True)
with open(Path(args.metrics_output) / "metrics.json", "w") as f:
    json.dump(metrics, f)

print(metrics)
