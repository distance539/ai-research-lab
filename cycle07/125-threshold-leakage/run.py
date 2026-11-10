"""Select on calibration, then score the locked rule and a deliberately leaked control."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GRID = (0.50, 0.90, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999)


def read(name):
    return json.loads((ROOT / name).read_bytes())


def write(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2) + '\n')


def counts(rows, threshold):
    accepted = [r for r in rows if threshold is not None and r['confidence'] >= threshold]
    n = len(accepted)
    errors = sum(1-r['correct'] for r in accepted)
    return dict(threshold=threshold, accepted=n, errors=errors, total=len(rows),
                risk=errors/n if n else None, coverage=n/len(rows))


def choose(points):
    eligible = [p for p in points if p['accepted'] and 20*p['errors'] <= p['accepted']]
    if eligible:
        return max(eligible, key=lambda p: (p['accepted'], -p['threshold']))['threshold']
    return None


def select():
    rows = read('calibration_predictions.json')
    assert [r['idx'] for r in rows] == read('split_manifest.json')['calibration']
    points = [counts(rows, t) for t in GRID]
    selected = dict(threshold=choose(points), calibration=points)
    write('selected.json', selected)
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['select','all'], default='all', nargs='?')
    args = parser.parse_args()
    lock = read('lock.json')
    for name, sha in lock.items():
        if args.phase == 'select' and name == 'audit_predictions.json':
            continue
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == sha, name
    selected = select()
    print('calibration threshold', selected['threshold'])
    if args.phase == 'select':
        return
    audit = read('audit_predictions.json')
    assert [r['idx'] for r in audit] == read('split_manifest.json')['protocol_audit']
    clean = counts(audit, selected['threshold'])
    points = [counts(audit, t) for t in GRID]
    leaked = counts(audit, choose(points))
    write('results.json', dict(locked=clean, leaked=leaked, audit_grid=points))
    print('locked', clean, 'leaked', leaked)


if __name__ == '__main__':
    main()
