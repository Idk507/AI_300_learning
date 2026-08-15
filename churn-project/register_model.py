from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Model
from azure.identity import DefaultAzureCredential
from pipelines.churn_pipeline import pipeline_job

# 1. Initialize client
ml_client = MLClient(
    credential=DefaultAzureCredential(),
    # subscription_id="<SUBSCRIPTION_ID>",
    # resource_group_name="<RESOURCE_GROUP>",
    # workspace_name="<WORKSPACE_NAME>",
)

# 2. Reference the job name/ID returned when submitting your pipeline
pipeline_job = ml_client.jobs.create_or_update(pipeline_job)
job_name = pipeline_job.name

# 3. Define the Model asset from the pipeline output
# "trained_model" matches the key defined in your @pipeline function return dictionary
model = Model(
    path=f"azureml://jobs/{job_name}/outputs/trained_model",
    name="churn-model",
    type=AssetTypes.CUSTOM_MODEL,
    description="RandomForest churn classifier trained via churn_pipeline",
)

# 4. Save/register the model in your Azure ML Workspace
registered_model = ml_client.models.create_or_update(model)
print(f"Registered Model: {registered_model.name}, Version: {registered_model.version}")