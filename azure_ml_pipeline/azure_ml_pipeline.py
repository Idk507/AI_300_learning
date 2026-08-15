"""
Azure Machine Learning end-to-end experimentation pipeline (Python SDK v2).

Covers: workspace client setup, MLTable data registration, AutoML job
submission, MLflow-tracked notebook training (autologging + custom logging),
model registration, and managed online / batch endpoint deployment.

Fixes applied vs. the earlier draft:
  1. mlflow.set_tracking_uri was being *assigned a string* instead of *called*.
     Fixed to mlflow.set_tracking_uri("<uri>").
  2. Several snippets used mlflow.* without importing mlflow. Added imports.
  3. mlflow.autolog() was being called inside the `with mlflow.start_run():`
     block on every run. Moved to run once, before any run starts, which is
     the documented and correct usage pattern.
  4. Unified variable naming (training_data_input) that was inconsistent
     between two earlier snippets (my_training_data_input vs training_data_input).
  5. Added a minimal, self-contained data-loading stub so the training
     section is runnable as-is rather than referencing undefined X_train/y_train.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

import mlflow
from xgboost import XGBClassifier

from azure.ai.ml import MLClient, automl, Input
from azure.ai.ml.entities import (
    Data,
    Model,
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    BatchEndpoint,
    BatchDeployment,
)
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.automl import ClassificationPrimaryMetrics
from azure.identity import DefaultAzureCredential


# ---------------------------------------------------------------------------
# 1. Workspace client
# ---------------------------------------------------------------------------

ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id="<SUBSCRIPTION_ID>",
    resource_group_name="<RESOURCE_GROUP>",
    workspace_name="<WORKSPACE_NAME>",
)


# ---------------------------------------------------------------------------
# 2. Register training data as a versioned MLTable asset
# ---------------------------------------------------------------------------

data_asset = Data(
    path="./data/diabetes-training",
    type=AssetTypes.MLTABLE,
    description="Diabetes classification training data",
    name="input-data-automl",
)
ml_client.data.create_or_update(data_asset)


# ---------------------------------------------------------------------------
# 3. Discover valid primary metrics for the task (do this before configuring)
# ---------------------------------------------------------------------------

print("Valid classification primary metrics:", list(ClassificationPrimaryMetrics))


# ---------------------------------------------------------------------------
# 4. Configure and submit the AutoML classification job
# ---------------------------------------------------------------------------

training_data_input = Input(
    type=AssetTypes.MLTABLE,
    path="azureml:input-data-automl:1",
)

classification_job = automl.classification(
    compute="aml-cluster",
    experiment_name="auto-ml-class-dev",
    training_data=training_data_input,
    target_column_name="Diabetic",
    primary_metric="accuracy",
    n_cross_validations=5,
    enable_model_explainability=True,
)

classification_job.set_limits(
    timeout_minutes=60,
    trial_timeout_minutes=20,
    max_trials=10,
    max_concurrent_trials=4,
    enable_early_termination=True,
)

returned_job = ml_client.jobs.create_or_update(classification_job)
print("Monitor at:", returned_job.studio_url)

# Block until the job finishes before touching its outputs downstream.
ml_client.jobs.stream(returned_job.name)


# ---------------------------------------------------------------------------
# 5. MLflow tracking setup for notebook-based experimentation
#    (only needed when NOT running on an Azure ML compute instance,
#    where the tracking URI is already pre-configured)
# ---------------------------------------------------------------------------

mlflow_tracking_uri = ml_client.workspaces.get(ml_client.workspace_name).mlflow_tracking_uri
mlflow.set_tracking_uri(mlflow_tracking_uri)   # was incorrectly `mlflow.set_tracking_uri = "..."`

mlflow.set_experiment(experiment_name="heart-condition-classifier")


# ---------------------------------------------------------------------------
# 6. Minimal data stub so the training section below is runnable as-is.
#    Replace with your real feature/label extraction.
# ---------------------------------------------------------------------------

df = pd.read_csv("./data/diabetes-training/diabetes.csv")
X = df.drop(columns=["Diabetic"])
y = df["Diabetic"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ---------------------------------------------------------------------------
# 7. Train with autologging (call autolog() once, before the run starts)
# ---------------------------------------------------------------------------

mlflow.autolog()

with mlflow.start_run(run_name="xgb-autolog"):
    model = XGBClassifier(eval_metric="logloss")
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)


# ---------------------------------------------------------------------------
# 8. Train with explicit custom logging (autolog disabled for this run
#    so metrics are not double-logged)
# ---------------------------------------------------------------------------

mlflow.autolog(disable=True)

with mlflow.start_run(run_name="xgb-manual-tuned"):
    params = {
        "eval_metric": "logloss",
        "max_depth": 6,
        "n_estimators": 300,
        "learning_rate": 0.05,
    }
    mlflow.log_params(params)

    model = XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    y_pred = model.predict(X_test)

    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred))
    mlflow.sklearn.log_model(model, artifact_path="model")


# ---------------------------------------------------------------------------
# 9. Register the winning model (AutoML job output shown here; swap the
#    `path` for a notebook run's artifact URI if that run wins instead)
# ---------------------------------------------------------------------------

registered_model = ml_client.models.create_or_update(
    Model(
        path=f"azureml://jobs/{returned_job.name}/outputs/artifacts/paths/model/",
        name="diabetes-classifier",
        type=AssetTypes.MLFLOW_MODEL,
        description="Best classification model selected after AutoML and notebook comparison",
    )
)


# ---------------------------------------------------------------------------
# 10. Deploy to a managed online endpoint (real-time inference)
# ---------------------------------------------------------------------------

endpoint = ManagedOnlineEndpoint(name="diabetes-classifier-endpoint", auth_mode="key")
ml_client.online_endpoints.begin_create_or_update(endpoint).result()

deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name="diabetes-classifier-endpoint",
    model=registered_model.id,
    instance_type="Standard_DS3_v2",
    instance_count=2,  # at least 2 for zero-downtime rolling updates
)
ml_client.online_deployments.begin_create_or_update(deployment).result()

# Route 100 percent of traffic to the new deployment once validated.
endpoint.traffic = {"blue": 100}
ml_client.online_endpoints.begin_create_or_update(endpoint).result()


# ---------------------------------------------------------------------------
# 11. Optional: batch endpoint for scheduled / offline scoring
# ---------------------------------------------------------------------------

batch_endpoint = BatchEndpoint(name="diabetes-classifier-batch")
ml_client.batch_endpoints.begin_create_or_update(batch_endpoint).result()

batch_deployment = BatchDeployment(
    name="default",
    endpoint_name="diabetes-classifier-batch",
    model=registered_model.id,
    compute="aml-cluster",
    instance_count=2,
    max_concurrency_per_instance=2,
)
ml_client.batch_deployments.begin_create_or_update(batch_deployment).result()
