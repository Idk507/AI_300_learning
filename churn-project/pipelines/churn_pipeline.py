from azure.ai.ml.dsl import pipeline
from azure.ai.ml import Input, MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Model
from azure.identity import DefaultAzureCredential
from azure.ai.ml import load_component

ml_client = MLClient(
    DefaultAzureCredential(),
    # subscription_id="<SUBSCRIPTION_ID>",
    # resource_group_name="<RESOURCE_GROUP>",
    # workspace_name="<WORKSPACE_NAME>",
)

prep = load_component(source="components/prep/prep.yml")
train = load_component(source="components/train/train.yml")
evaluate = load_component(source="components/evaluate/evaluate.yml")

@pipeline()
def churn_pipeline(raw_data):
    prep_step = prep(input_data=raw_data)
    train_step = train(training_data=prep_step.outputs.output_data)
    eval_step = evaluate(model_input=train_step.outputs.model_output)
    return {
        "trained_model": train_step.outputs.model_output,
        "metrics": eval_step.outputs.metrics_output,
    }

pipeline_job = churn_pipeline(
    Input(type=AssetTypes.URI_FILE, path="azureml:churn-data:1")
)
pipeline_job.settings.default_compute = "churn-cluster"
pipeline_job.settings.default_datastore = "workspaceblobstore"

pipeline_job = ml_client.jobs.create_or_update(
    pipeline_job, experiment_name="churn-training"
)
print(pipeline_job.studio_url)



# # Create and submit pipeline
# pipeline_job = ml_client.jobs.create_or_update(pipeline_job, experiment_name="churn-training")
# print(f"Submitted run: {pipeline_job.name}")

# # Wait for completion before registering
# ml_client.jobs.stream(pipeline_job.name)

# # Register using the dynamic run name
# model = Model(
#     path=f"azureml://jobs/{pipeline_job.name}/outputs/trained_model",
#     name="churn-model",
#     type=AssetTypes.CUSTOM_MODEL
# )
# ml_client.models.create_or_update(model)