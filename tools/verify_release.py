"""Static release and aggregate consistency checks; no private input required."""
from pathlib import Path
import ast, csv, hashlib, json, re

root = Path(__file__).resolve().parents[1]
for p in root.rglob('*.py'): ast.parse(p.read_text(encoding='utf-8-sig'), filename=str(p))
manifest = json.loads((root/'SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
for row in manifest['files']:
    assert hashlib.sha256((root/row['file']).read_bytes()).hexdigest() == row['release_sha256']
with (root/'data_sources/datasets.csv').open(encoding='utf-8', newline='') as f: rows = list(csv.DictReader(f))
assert len(rows) == 23
assert sum(int(r['project_query_records']) for r in rows if r['dataset'] != 'In-house') == 1731
table = json.loads((root/'results/manuscript_tables.json').read_text(encoding='utf-8'))['tables'][-1]
from statistics import mean, stdev
for row in table[1:]:
    counts = [int(s.replace(',', '').strip()) for s in row[1].split('/')]
    accuracies = [v / 1731 * 100 for v in counts]
    expected = [float(s.strip()) for s in row[2].split('±')]
    assert abs(mean(accuracies)-expected[0]) <= .0051
    assert abs(stdev(accuracies)-expected[1]) <= .0051
for p in root.rglob('*'):
    if not p.is_file() or '.git' in p.parts or '__pycache__' in p.parts: continue
    assert p.suffix not in {'.pt','.pth','.npz','.npy','.joblib','.pkl','.xlsx','.pem','.key','.log'}, str(p)
    text = p.read_text(encoding='utf-8-sig')
    for pattern in [r'gh[pousr]_[A-Za-z0-9]{25,}', r'github_pat_[A-Za-z0-9_]{20,}', r'-----BEGIN [A-Z ]*PRIVATE KEY-----']:
        assert not re.search(pattern, text), 'Potential secret in '+str(p.relative_to(root))
    assert not re.search(r'/home/[A-Za-z0-9_-]+/', text), str(p)
print(json.dumps({'status':'passed','source_files':len(manifest['files']),'datasets':len(rows),'external_query_records':1731,'controlled_table_mean_and_sample_sd':'passed','secret_and_private_artifact_scan':'passed'}))
