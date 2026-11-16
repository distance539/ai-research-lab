"""Paired scoring with unchanged upstream modules loaded in their package layout."""
import json
import hashlib
import shutil
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent

def read(name):
    return json.loads((ROOT / name).read_text())

def write(name, data):
    (ROOT / name).write_text(json.dumps(data, indent=2) + '\n')

def compare(cal, audit, methods, quantile, protocol):
    labels = torch.tensor([r["label"] for r in cal])
    inputs = [torch.tensor([r["logits"] for r in rows], dtype=torch.float64) for rows in (cal, audit)]
    outcomes = {}
    pairs = [dict(idx=r["idx"], label=r["label"]) for r in audit]
    for name, method in methods.items():
        threshold = quantile(method(inputs[0], labels), protocol["alpha"])
        scores = method(inputs[1])
        sizes = [0, 0, 0]
        hits = 0
        for row, score in zip(pairs, scores):
            selected = (score <= threshold).nonzero().flatten().tolist()
            row[name] = selected
            sizes[len(selected)] += 1
            hits += int(row["label"] in selected)
        outcomes[name] = dict(q=float(threshold), sizes=sizes, hits=hits, slots=sizes[1]+2*sizes[2])
    outcomes["different"] = sum(p["LAC"] != p["APS"] for p in pairs)
    outcomes["paired_hits"] = {str((a,b)):sum((p["label"] in p["LAC"])==a and (p["label"] in p["APS"])==b for p in pairs) for a in (False,True) for b in (False,True)}
    write("results.json", outcomes)
    write("memberships.json", pairs)
    print(json.dumps(outcomes))

def main():
    for name, expected in read("lock.json").items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected, name
    cal, audit = read("calibration_predictions.json"), read("run_predictions.json")
    manifest = read("split_manifest.json")
    assert [r["idx"] for r in cal] == manifest["calibration"]
    assert [r["idx"] for r in audit] == manifest["protocol_audit"]
    with tempfile.TemporaryDirectory() as tmp:
        parent = Path(tmp)/"torchcp/classification/score"
        parent.mkdir(parents=True)
        for name in ["base","lac","aps"]:
            shutil.copyfile(ROOT/(name+".py"), parent/(name+".py"))
        sys.path.insert(0,tmp)
        from torchcp.classification.score.aps import APS
        from torchcp.classification.score.lac import LAC
        from common import calculate_conformal_value
        compare(cal, audit, {"LAC":LAC(), "APS":APS(randomized=False)}, calculate_conformal_value, read("protocol.json"))

if __name__ == "__main__":
    main()
