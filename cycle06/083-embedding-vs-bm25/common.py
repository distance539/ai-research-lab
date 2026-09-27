"""Independent experiment utilities; no upstream source copied."""
from pathlib import Path
import csv
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parent

def read_json(path):
    return json.loads(Path(path).read_text())

def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines()]

def qrels(path):
    out = {}
    for r in csv.DictReader(Path(path).open(), delimiter='\t'):
        q, d, score = r['query-id'], r['corpus-id'], int(r['score'])
        assert d not in out.setdefault(q, {}), 'duplicate qrel'
        out[q][d] = score
    return out

def ranked(result):
    return sorted(result, key=lambda d: (result[d], d), reverse=True)

def metrics(gold, order):
    relevant = {d for d, score in gold.items() if score > 0}
    assert relevant
    out = {}
    for k in [1, 10, 100]:
        hit = [int(d in relevant) for d in order[:k]]
        dcg = sum(v / math.log2(i + 2) for i, v in enumerate(hit))
        ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(relevant))))
        out[f'ndcg_cut_{k}'] = dcg / ideal
        out[f'recall_{k}'] = sum(hit) / len(relevant)
    out['mrr_10'] = next((1 / i for i, d in enumerate(order[:10], 1) if d in relevant), 0.0)
    return out

def verify_resources(cache):
    cache = Path(cache)
    lock = read_json(ROOT / 'resources.lock.json')
    for rel, expected in lock['data_files'].items():
        assert sha(cache / rel) == expected, f'data hash mismatch: {rel}'
    upstream = cache / 'upstream' / ('beir-' + lock['beir_commit'])
    for rel, expected in lock['beir_files'].items():
        assert sha(upstream / rel) == expected, f'upstream hash mismatch: {rel}'
    return lock, upstream
