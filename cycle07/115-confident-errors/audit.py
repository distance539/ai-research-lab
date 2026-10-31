"""Independent stdlib-only metric reconstruction, no imports from run.py."""
import argparse
import copy
import json
import math
from pathlib import Path


def check(rows, summary, expected):
    assert [r["idx"] for r in rows] == expected
    assert len(set(expected)) == len(expected)
    n = len(rows)
    correct = conf = nll = brier = 0.0
    bins = [[0, 0.0, 0.0] for _ in range(10)]
    high_n = high_ok = 0
    for r in rows:
        z = r["logits"]
        assert all(math.isfinite(x) for x in z) and r["label"] in (0, 1)
        exps = [math.exp(x - max(z)) for x in z]
        probs = [v / sum(exps) for v in exps]
        pred = int(z[1] > z[0])
        confidence = probs[pred]
        ok = int(pred == r["label"])
        loss = max(z) + math.log(sum(exps)) - z[r["label"]]
        assert r["prediction"] == pred and r["correct"] == ok
        for actual, expected_p in zip(r["probabilities"], probs):
            assert abs(actual - expected_p) < 1e-12
        assert abs(r["confidence"] - confidence) < 1e-12
        assert abs(r["nll"] - loss) < 1e-12
        correct += ok
        conf += confidence
        nll += loss
        brier += 2 * (probs[1] - r["label"]) ** 2
        index = next((i for i in range(10) if confidence < (i + 1) / 10), 9)
        bins[index][0] += 1
        bins[index][1] += confidence
        bins[index][2] += ok
        if confidence >= 0.9:
            high_n += 1
            high_ok += ok
    values = {"accuracy": correct / n, "mean_confidence": conf / n, "signed_gap": (conf - correct) / n, "nll": nll / n, "brier_two_class_sum": brier / n, "ece10": sum(abs(c - a) for _, c, a in bins) / n}
    for key, value in values.items():
        assert abs(summary[key] - value) < 1e-12, key
    assert summary["n"] == n and summary["correct"] == correct
    assert summary["high_confidence"]["n"] == high_n
    assert summary["high_confidence"]["correct"] == high_ok
    for index, (count, confidence, accuracy) in enumerate(bins):
        saved = summary["bins"][index]
        assert saved["n"] == count
        if count:
            assert abs(saved["confidence"] - confidence / count) < 1e-12
            assert abs(saved["accuracy"] - accuracy / count) < 1e-12
        else:
            assert saved["accuracy"] is None and saved["confidence"] is None
    return values


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prefix", default="formal")
    a = p.parse_args()
    here = Path(__file__).resolve().parent
    rows = json.loads((here / f"{a.prefix}_predictions.json").read_text())
    summary = json.loads((here / f"{a.prefix}_summary.json").read_text())
    ids = json.loads((here / "split_manifest.json").read_text())["exploration"][:len(rows)]
    values = check(rows, summary, ids)
    mutations = {}
    for field in ["idx", "label", "prediction", "correct", "confidence", "nll"]:
        bad = copy.deepcopy(rows)
        bad[0][field] += 1
        try:
            check(bad, summary, ids)
        except AssertionError:
            mutations[field] = "rejected"
        else:
            raise AssertionError("Mutation escaped: " + field)
    result = {"passed": True, "n": len(rows), "independent_metrics": values, "mutation_tests": mutations, "human_review": False}
    (here / f"{a.prefix}_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
