"""Finite-sample LAC with an explicit overflow rule and capped-rank control."""
from fractions import Fraction
from pathlib import Path
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parent

def load(name):
    return json.loads((ROOT / name).read_bytes())

def save(name, value):
    (ROOT / name).write_text(json.dumps(value, separators=(',', ':'), allow_nan=False) + '\n')

def scores(row):
    logits = row['logits']
    weights = [math.exp(z - max(logits)) for z in logits]
    return [1 - w / sum(weights) for w in weights]

def main():
    protocol = load('protocol.json')
    calibration = load('calibration_predictions.json')
    audit = load('run_predictions.json')
    manifest = load('split_manifest.json')
    assert [r['idx'] for r in calibration] == manifest['calibration']
    assert [r['idx'] for r in audit] == manifest['protocol_audit']
    assert len(set(manifest['calibration'] + manifest['protocol_audit'])) == 616
    true_scores = [scores(r)[r['label']] for r in calibration]
    audit_scores = [scores(r) for r in audit]
    summaries, masks = [], []
    for n in protocol['sizes']:
        rank = math.ceil((n + 1) * (1 - Fraction(protocol['alpha'])))
        ordered = sorted(true_scores[:n])
        for rule in ('conformal', 'capped'):
            threshold = None if rule == 'conformal' and rank > n else ordered[min(n, rank) - 1]
            sizes = [0, 0, 0]
            hits = 0
            decisions = []
            for row, pair in zip(audit, audit_scores):
                labels = [y for y, score in enumerate(pair) if threshold is None or score <= threshold]
                sizes[len(labels)] += 1
                hits += row['label'] in labels
                decisions.append([row['idx'], labels])
            summaries.append(dict(n=n, rank=rank, rule=rule, q=threshold, sizes=sizes,
                                  hits=hits, total=len(audit), mean_size=(sizes[1]+2*sizes[2])/len(audit)))
            masks.append(dict(n=n, rule=rule, decisions=decisions))
    save('results.json', summaries)
    save('memberships.json', masks)
    names = 'protocol.json config.json split_manifest.json resources.lock.json calibration_predictions.json audit_predictions.json run_predictions.json run.py audit.py baseline.py model_run.py'.split()
    save('lock.json', {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names})
    for row in summaries:
        print(row)

if __name__ == '__main__':
    main()
