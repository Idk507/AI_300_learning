import json
import logging
import os
import joblib
import pandas as pd

def init():
    global model
    model_dir = os.getenv("AZUREML_MODEL_DIR")
    
    # Locate model.pkl dynamically in case of nested directory paths
    model_path = None
    for root, dirs, files in os.walk(model_dir):
        if "model.pkl" in files:
            model_path = os.path.join(root, "model.pkl")
            break
            
    if not model_path:
        raise FileNotFoundError(f"model.pkl not found under {model_dir}")

    model = joblib.load(model_path)
    logging.info("Model loaded successfully")

def run(raw_data):
    try:
        data = json.loads(raw_data)
        if "input_data" in data and "data" in data["input_data"]:
            input_df = pd.DataFrame(data["input_data"]["data"])
        elif "data" in data:
            input_df = pd.DataFrame(data["data"])
        else:
            input_df = pd.DataFrame(data)

        predictions = model.predict(input_df)
        return predictions.tolist()
    except Exception as e:
        return {"error": str(e)}