from azure.ai.ml import MLClient
from azure.ai.ml import load_component
from azure.identity import DefaultAzureCredential

# Initialize MLClient
ml_client = MLClient(
    credential=DefaultAzureCredential(),
    # subscription_id="<SUBSCRIPTION_ID>",
    # resource_group_name="<RESOURCE_GROUP>",
    # workspace_name="<WORKSPACE_NAME>",
)

# Load component YAML definitions
prep = load_component(source="components/prep/prep.yml")
train = load_component(source="components/train/train.yml")
evaluate = load_component(source="components/evaluate/evaluate.yml")

# Register or update components in the Azure ML Workspace
for c in [prep, train, evaluate]:
    ml_client.components.create_or_update(c)
    print(f"Registered component: {c.name}")