from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

ml_client = MLClient(
    credential=DefaultAzureCredential(),
    # subscription_id="<SUBSCRIPTION_ID>",
    # resource_group_name="<RESOURCE_GROUP>",
    # workspace_name="<WORKSPACE_NAME>",
)
print("Connected to Azure ML Workspace:", ml_client.workspace_name)
print(dir(ml_client))
