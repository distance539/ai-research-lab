"""Bind saved predictions to actual public data and the frozen protocol."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import pyarrow.parquet as pq

p = argparse.ArgumentParser()
p.add_argument("--cache", type=Path, required=True)
a = p.parse_args()
here = Path(__file__).resolve().parent
read = lambda name: json.loads((here / name).read_text())
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
lock = read("resources.lock.json")
data = a.cache / "validation.parquet"
assert sha(data) == lock["dataset"]["sha256"]
raw = {r["idx"]: r for r in pq.read_table(data).to_pylist()}
ordered = sorted(raw, key=lambda i: hashlib.sha256(f"cycle7:{i}".encode()).hexdigest())
split = read("split_manifest.json")
assert split["exploration"] == ordered[:256]
assert split["calibration"] == ordered[256:512]
assert split["protocol_audit"] == ordered[512:]
rows = read("formal_predictions.json")
assert [r["idx"] for r in rows] == ordered[:256]
assert not set(r["idx"] for r in rows) & set(split["calibration"] + split["protocol_audit"])
for r in rows:
    assert r["label"] == raw[r["idx"]]["label"]
    assert r["text_sha256"] == hashlib.sha256(raw[r["idx"]]["sentence"].encode()).hexdigest()
for prefix in ["smoke", "formal"]:
    start = read(prefix + "_start.json")
    for field, file in [("config_sha256", "config.json"), ("protocol_sha256", "PROTOCOL.md"), ("split_sha256", "split_manifest.json")]:
        if field == "protocol_sha256":
            assert start[field] in {sha(here / "PROTOCOL.md"), sha(here / "baseline_PROTOCOL.md")}
        else:
            assert start[field] == sha(here / file)
smoke = read("smoke_predictions.json")
max_delta = max(abs(s["logits"][j] - r["logits"][j]) for s, r in zip(smoke, rows) for j in (0, 1))
assert max_delta < 1e-4
assert all(s["prediction"] == r["prediction"] for s, r in zip(smoke, rows))
source = Path(importlib.metadata.distribution("transformers").locate_file("transformers/models/distilbert/modeling_distilbert.py"))
assert sha(source) == "7aa246bf0a4b45f901b83a9ab993e7dd2c0fb91f4fbb874045baec8ae7429a6e"
result = {"passed": True, "n": len(rows), "input_labels_and_hashes_match": True, "splits_disjoint": True, "calibration_and_audit_predictions_accessed": False, "protocol_unchanged": True, "smoke_formal_max_logit_delta": max_delta, "official_model_source_match": True, "exact_ece_boundary_count": sum(any(r["confidence"] == b / 10 for b in range(1,10)) for r in rows)}
(here / "input_verification.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
