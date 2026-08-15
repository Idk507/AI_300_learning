import json
from pathlib import Path

path = Path('pieline_demo/demo.ipynb')
nb = json.loads(path.read_text(encoding='utf-8'))
cell_index = None
for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') != 'code':
        continue
    src = cell['source']
    if any('ml_client.models.create_or_update(' in line for line in src):
        cell_index = i
        break
if cell_index is None:
    raise SystemExit('No model registration cell found')
cell = nb['cells'][cell_index]
old_src = cell['source']
new_src = []
for line in old_src:
    if 'model_uri = mlflow.get_artifact_uri("model")' in line:
        # drop this line entirely by skipping it
        continue
    if 'print("MLflow artifact URI:", model_uri)' in line:
        continue
    if 'except Exception as ex:' in line and 'mlflow.get_artifact_uri' in ''.join(old_src[old_src.index(line):old_src.index(line)+4]):
        new_src.append(line)
        continue
    new_src.append(line)

# Replace the block to use a strict runs:/<run_id>/model path.
updated_lines = []
skipping = False
for line in new_src:
    if 'if run_id is None:' in line and '        raise RuntimeError("Could not locate an MLflow run for the manual training model.")' in line:
        updated_lines.append(line)
        skipping = True
        continue
    if skipping:
        if line.strip().startswith('print(') and 'Using MLflow run id:' in line:
            updated_lines.append('    model_uri = f"runs:/{run_id}/model"\n')
            updated_lines.append('    print("Registering model from MLflow run URI:", model_uri)\n')
            skipping = False
            continue
        # skip any lines until we get past the print or except block inserted earlier
        continue
    updated_lines.append(line)

# fallback if the earlier logic didn't find injection point
if updated_lines == new_src:
    updated_lines = []
    for line in old_src:
        if '    if run_id is None:' in line:
            updated_lines.append(line)
            updated_lines.append('        raise RuntimeError("Could not locate an MLflow run for the manual training model.")\n')
            updated_lines.append('    model_uri = f"runs:/{run_id}/model"\n')
            updated_lines.append('    print("Registering model from MLflow run URI:", model_uri)\n')
            # skip the next few lines in the old block until registered_model
            skip = True
            continue
        if 'registered_model = ml_client.models.create_or_update(' in line:
            updated_lines.append(line)
            skip = False
            continue
        if skip:
            continue
        updated_lines.append(line)

cell['source'] = updated_lines
path.write_text(json.dumps(nb, indent=1), encoding='utf-8')
print(f'Patched cell {cell_index}')