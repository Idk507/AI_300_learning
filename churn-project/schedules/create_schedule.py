import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from azure.ai.ml import MLClient
from azure.ai.ml.entities import JobSchedule, RecurrenceTrigger
from azure.identity import DefaultAzureCredential
from pipelines.churn_pipeline import pipeline_job

# 1. Initialize client
ml_client = MLClient(
    credential=DefaultAzureCredential(),
    # subscription_id="<SUBSCRIPTION_ID>",
    # resource_group_name="<RESOURCE_GROUP>",
    # workspace_name="<WORKSPACE_NAME>",
)

# 2. Define recurrence (e.g., run every 1 week)
trigger = RecurrenceTrigger(
    frequency="week",
    interval=1,
)

# 3. Create the schedule bound to your constructed `pipeline_job` object
schedule = JobSchedule(
    name="weekly-churn-retrain-schedule",
    trigger=trigger,
    create_job=pipeline_job,
    description="Trigger weekly churn pipeline retraining job",
)

# 4. Save and start the schedule in Azure ML Workspace
saved_schedule = ml_client.schedules.begin_create_or_update(schedule=schedule).result()
print(f"Created Schedule: {saved_schedule.name}")