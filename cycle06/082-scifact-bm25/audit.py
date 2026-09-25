"""Offline replay, including deliberately missing/empty/tied run fixtures."""
import argparse
from pathlib import Path
from common import metrics, ranked, read_json, write_json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    args = ap.parse_args()
    result = read_json(args.run / 'results.json')
    raw = read_json(args.run / 'raw_results.json')
    gold = read_json(args.run / 'qrels_used.json')
    recorded = read_json(args.run / 'per_query.json')
    summary = read_json(args.run / 'summary.json')
    assert set(result) == set(gold) == set(recorded) == set(raw)
    error = 0.0
    for q in gold:
        assert len(result[q]) <= 100
        assert list(result[q]) == ranked(raw[q])[:100]
        for d, value in result[q].items():
            assert value == raw[q][d]
        m = metrics(gold[q], ranked(result[q]))
        for name in m:
            error = max(error, abs(m[name] - recorded[q][name]))
    for name, value in summary['metrics'].items():
        assert abs(sum(x[name] for x in recorded.values()) / len(gold) - value) < 1e-12
    assert metrics({'a': 1}, [])['ndcg_cut_10'] == 0
    assert metrics({'a': 1}, ['b', 'a'])['mrr_10'] == .5
    assert ranked({'a': 1, 'b': 1}) == ['b', 'a']
    # A missing query is rejected, not averaged away.
    corrupted = dict(result)
    corrupted.pop(next(iter(corrupted)))
    assert set(corrupted) != set(gold)
    assert error < 1e-10
    report = {'passed': True, 'queries': len(gold), 'max_error': error,
              'fixtures': ['empty run -> zero', 'MRR rank2', 'descending lexical tie', 'missing query rejected'],
              'scope': 'independent scalar metric formulas; no reretrieval or training'}
    write_json(args.run / 'audit.json', report)
    print(report)

if __name__ == '__main__':
    main()
