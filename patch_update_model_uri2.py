import json
from pathlib import Path

path = Path('pieline_demo/demo.ipynb')
nb = json.loads(path.read_text(encoding='utf-8'))
for cell in nb['cells']:
    if cell.get('cell_type') != 'code':
        continue
    src = cell['source']
    if any('ml_client.models.create_or_update(' in line for line in src):
        break
else:
    raise SystemExit('Registration cell not found')

new_src = []
skipped = False
for idx, line in enumerate(src):
    if 'if run_id is None:' in line and 'raise RuntimeError("Could not locate an MLflow run for the manual training model.")' not in line:
        # keep the if line
        new_src.append(line)
        continue
    if '        raise RuntimeError("Could not locate an MLflow run for the manual training model.")' in line:
        new_src.append(line)
        new_src.append('    model_uri = f"runs:/{run_id}/model"\n')
        new_src.append('    print("Registering model from MLflow run URI:", model_uri)\n')
        skipped = True
        continue
    if skipped:
        if line.strip().startswith('registered_model = ml_client.models.create_or_update('):
            new_src.append(line)
            skipped = False
        continue
    new_src.append(line)

cell['source'] = new_src
path.write_text(json.dumps(nb, indent=1), encoding='utf-8')
print('Updated model registration cell')
