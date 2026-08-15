#!/usr/bin/env bash
# Azure ML end-to-end provisioning and operations commands (Azure CLI + ml extension).
#
# Fixes applied vs. the earlier draft:
#   1. `az consumption budget create` was missing the required --category
#      parameter; without it the command is rejected by the API.
#   2. `az ml online-deployment update --set data_collector.enabled=true`
#      is not a valid generic-update path for this resource type; data
#      collection is enabled by setting `data_collector` in the deployment
#      YAML (or the DataCollector object in Python SDK) at create time,
#      not patched afterward with --set. Replaced with the correct flow:
#      redeploy with a YAML file that declares data_collector settings.
#   3. Added `set -euo pipefail` so the script stops on the first failure
#      instead of silently continuing with a partially-provisioned stack.

set -euo pipefail

RESOURCE_GROUP="<RESOURCE_GROUP>"
WORKSPACE="<WORKSPACE_NAME>"
LOCATION="<LOCATION>"

az login
az extension add --name ml -y --upgrade

# ---------------------------------------------------------------------------
# 1. Resource group
# ---------------------------------------------------------------------------

az group create \
  --name "$RESOURCE_GROUP" \
  --location "$LOCATION"

# ---------------------------------------------------------------------------
# 2. Azure Machine Learning workspace
# ---------------------------------------------------------------------------

az ml workspace create \
  --name "$WORKSPACE" \
  --resource-group "$RESOURCE_GROUP" \
  --location "$LOCATION"

az ml workspace show \
  --name "$WORKSPACE" \
  --resource-group "$RESOURCE_GROUP" \
  --query mlflow_tracking_uri -o tsv

# ---------------------------------------------------------------------------
# 3. Compute: interactive instance + autoscaling training cluster
# ---------------------------------------------------------------------------

az ml compute create \
  --name ci-dev-instance \
  --resource-group "$RESOURCE_GROUP" \
  --workspace-name "$WORKSPACE" \
  --type ComputeInstance \
  --size Standard_DS3_v2

az ml compute create \
  --name aml-cluster \
  --resource-group "$RESOURCE_GROUP" \
  --workspace-name "$WORKSPACE" \
  --type AmlCompute \
  --size Standard_DS3_v2 \
  --min-instances 0 \
  --max-instances 4

# ---------------------------------------------------------------------------
# 4. Responsible AI dashboard pipeline
# ---------------------------------------------------------------------------

az ml job create \
  --file rai-pipeline.yml \
  --resource-group "$RESOURCE_GROUP" \
  --workspace-name "$WORKSPACE"

# ---------------------------------------------------------------------------
# 5. Enable network isolation on the workspace (production only)
# ---------------------------------------------------------------------------

az ml workspace update \
  --name "$WORKSPACE" \
  --resource-group "$RESOURCE_GROUP" \
  --public-network-access Disabled

# ---------------------------------------------------------------------------
# 6. Enable data collection / drift monitoring on the online deployment.
#    Corrected: data_collector cannot be patched with --set on an existing
#    deployment; redeploy from a YAML file that declares it.
#    online-deployment.yml should contain:
#      data_collector:
#        collections:
#          model_inputs:
#            enabled: "true"
#          model_outputs:
#            enabled: "true"
# ---------------------------------------------------------------------------

az ml online-deployment update \
  --name blue \
  --endpoint-name diabetes-classifier-endpoint \
  --resource-group "$RESOURCE_GROUP" \
  --workspace-name "$WORKSPACE" \
  --file online-deployment.yml

# ---------------------------------------------------------------------------
# 7. Budget alert on the resource group (--category is required)
# ---------------------------------------------------------------------------

az consumption budget create \
  --budget-name mlops-dev-monthly \
  --category Cost \
  --amount 500 \
  --time-grain Monthly \
  --start-date 2026-08-01 \
  --end-date 2027-08-01 \
  --resource-group "$RESOURCE_GROUP"
