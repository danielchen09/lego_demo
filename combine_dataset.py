import ndjson
import os
from pathlib import Path

new_dataset = []

for p in Path('datasets/').glob('*.ndjson'):
    with open(p) as f:
        data = ndjson.load(f)
    for x in data:
        if 'annotations' in x and len(x['annotations']['boxes']) == 42:
            new_dataset.append(x)

print(len(new_dataset))
new_dataset.insert(0, data[0])

with open('combined_connect_4_dataset.ndjson', 'w') as f:
    ndjson.dump(new_dataset, f)