"""Compare calibrated probabilities and selection at equal coverage."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def read(name):
    return json.loads((ROOT / name).read_bytes())

def write(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2, allow_nan=False))

def evaluate(rows, temperature):
    scores = []
    bins = [[] for _ in range(10)]
    for row in rows:
        z = row['logits']
        weights = [math.exp((v - max(z)) / temperature) for v in z]
        p = [v / sum(weights) for v in weights]
        prediction = int(p[1] > p[0])
        confidence = max(p)
        wrong = int(prediction != row['label'])
        bins[min(9, int(10 * confidence))].append(confidence - (1 - wrong))
        scores.append(dict(idx=row['idx'], confidence=confidence, wrong=wrong,
                           nll=-math.log(p[row['label']])))
    scores.sort(key=lambda r: (-r['confidence'], r['idx']))
    curve = [dict(k=0, errors=0, risk=None)]
    errors = 0
    for k, sample in enumerate(scores, 1):
        errors += sample['wrong']
        curve.append(dict(k=k, errors=errors, risk=errors/k))
    n = len(scores)
    accepted = sum(r['confidence'] >= .99 for r in scores)
    summary = dict(temperature=temperature, nll=sum(r['nll'] for r in scores)/n,
                   ece=sum(abs(sum(b)) for b in bins)/n,
                   aurc=sum(c['risk'] for c in curve[1:])/n,
                   ties=n-len({r['confidence'] for r in scores}),
                   fixed_threshold=curve[accepted], fixed_budget=curve[270])
    return scores, curve, summary

def main():
    for name, expected in read('lock.json').items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected, name
    rows = read('audit_predictions.json')
    results = {}
    details = {}
    for name, temperature in [('raw', 1), ('scaled', read('fit.json')['temperature'])]:
        scores, curve, results[name] = evaluate(rows, temperature)
        details[name] = dict(scores=scores, curve=curve)
    results['rank_mismatches'] = sum(a['idx'] != b['idx'] for a,b in zip(
        details['raw']['scores'], details['scaled']['scores']))
    results['unequal_prefixes'] = sum(a != b for a,b in zip(
        details['raw']['curve'], details['scaled']['curve']))
    write('paired.json', details)
    write('results.json', results)
    print(json.dumps(results))

if __name__ == '__main__':
    main()
