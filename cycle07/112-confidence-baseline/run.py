"""Frozen real-model baseline; download resources separately from inference."""
import argparse
import hashlib
import json
import math
import os
import platform
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def fetch(url, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["curl", "--fail", "--location", "--retry", "2", "--connect-timeout", "20", "--max-time", "600", "--output", str(path), url], check=True)


def prepare(cache, cfg):
    cache.mkdir(parents=True, exist_ok=True)
    model_dir = cache / cfg["model_revision"]
    api = cache / "model_api.json"
    fetch(f'https://huggingface.co/api/models/{cfg["model_id"]}/revision/{cfg["model_revision"]}?blobs=true', api)
    info = json.loads(api.read_text())
    assert info["sha"] == cfg["model_revision"]
    files = {x["rfilename"]: x for x in info["siblings"]}
    locks = []
    for name in ["config.json", "tokenizer_config.json", "vocab.txt", "model.safetensors", "README.md"]:
        url = f'https://huggingface.co/{cfg["model_id"]}/resolve/{cfg["model_revision"]}/{name}'
        path = model_dir / name
        if not path.exists():
            fetch(url, path)
        raw = path.read_bytes()
        entry = files[name]
        if "lfs" in entry:
            assert hashlib.sha256(raw).hexdigest() == entry["lfs"]["sha256"]
        else:
            git_blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
            assert git_blob == entry["blobId"]
        locks.append({"name": name, "sha256": digest(path), "bytes": len(raw), "official": entry, "url": url})
    data = cache / "validation.parquet"
    url = f'https://huggingface.co/datasets/{cfg["dataset_id"]}/resolve/{cfg["dataset_revision"]}/data/validation-00000-of-00001.parquet'
    if not data.exists():
        fetch(url, data)
    existing_lock = HERE / "resources.lock.json"
    if existing_lock.exists():
        expected = json.loads(existing_lock.read_text())
        assert digest(data) == expected["dataset"]["sha256"], "Data differs from the published lock"
    save(HERE / "resources.lock.json", {"created_at": datetime.now(timezone.utc).isoformat(), "model": cfg["model_id"], "revision": cfg["model_revision"], "files": locks, "dataset": {"url": url, "revision": cfg["dataset_revision"], "sha256": digest(data), "bytes": data.stat().st_size}})
    print("Resources verified", flush=True)


def metrics(records, bins=10, threshold=0.9):
    n = len(records)
    if not n:
        raise ValueError("Empty predictions")
    groups = [[] for _ in range(bins)]
    for r in records:
        groups[min(int(r["confidence"] * bins), bins - 1)].append(r)
    table = []
    for i, group in enumerate(groups):
        c = sum(x["confidence"] for x in group) / len(group) if group else None
        a = sum(x["correct"] for x in group) / len(group) if group else None
        table.append({"bin": i, "left": i / bins, "right": (i + 1) / bins, "n": len(group), "confidence": c, "accuracy": a})
    high = [r for r in records if r["confidence"] >= threshold]
    acc = sum(r["correct"] for r in records) / n
    conf = sum(r["confidence"] for r in records) / n
    return {"n": n, "correct": sum(r["correct"] for r in records), "accuracy": acc, "mean_confidence": conf, "signed_gap": conf - acc,
            "ece10": sum(b["n"] / n * abs(b["confidence"] - b["accuracy"]) for b in table if b["n"]),
            "nll": sum(r["nll"] for r in records) / n,
            "brier_two_class_sum": sum(sum((r["probabilities"][k] - int(r["label"] == k)) ** 2 for k in range(2)) for r in records) / n,
            "high_confidence": {"threshold": threshold, "n": len(high), "correct": sum(r["correct"] for r in high), "accuracy": sum(r["correct"] for r in high) / len(high) if high else None}, "bins": table}


def infer(cache, cfg, smoke):
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    started = time.perf_counter()
    prefix = "smoke" if smoke else "formal"
    torch.set_num_threads(cfg["threads"])
    torch.manual_seed(cfg["seed"])
    torch.use_deterministic_algorithms(True)
    data = cache / "validation.parquet"
    lock = json.loads((HERE / "resources.lock.json").read_text())
    assert digest(data) == lock["dataset"]["sha256"]
    model_dir = cache / cfg["model_revision"]
    for entry in lock["files"]:
        assert digest(model_dir / entry["name"]) == entry["sha256"]
    rows = pq.read_table(data).to_pylist()
    assert len(rows) == 872 and len({r["idx"] for r in rows}) == 872
    rows.sort(key=lambda r: hashlib.sha256(f'cycle7:{r["idx"]}'.encode()).hexdigest())
    manifest = {"rule": cfg["selection"], "data_sha256": digest(data), "exploration": [r["idx"] for r in rows[:256]], "calibration": [r["idx"] for r in rows[256:512]], "protocol_audit": [r["idx"] for r in rows[512:]], "historically_reused_dataset": True}
    split_path = HERE / "split_manifest.json"
    if split_path.exists():
        assert json.loads(split_path.read_text()) == manifest
    else:
        save(split_path, manifest)
    selected = rows[:8 if smoke else cfg["n"]]
    save(HERE / f"{prefix}_start.json", {"time": datetime.now(timezone.utc).isoformat(), "config_sha256": digest(HERE / "config.json"), "protocol_sha256": digest(HERE / "PROTOCOL.md"), "split_sha256": digest(split_path), "n": len(selected), "smoke": smoke})
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True, use_safetensors=True, attn_implementation=cfg["attention"]).to("cpu").eval()
    assert model.config.id2label == {0: "NEGATIVE", 1: "POSITIVE"}
    records = []
    shapes = []
    inference_start = time.perf_counter()
    for offset in range(0, len(selected), cfg["batch_size"]):
        batch = selected[offset:offset + cfg["batch_size"]]
        texts = [r["sentence"] for r in batch]
        lengths = [len(tokenizer(t, truncation=False)["input_ids"]) for t in texts]
        tokens = tokenizer(texts, padding=True, truncation=True, max_length=cfg["max_length"], return_tensors="pt")
        with torch.inference_mode():
            logits = model(**tokens).logits
        assert logits.shape == (len(batch), 2) and torch.isfinite(logits).all()
        lp = torch.log_softmax(logits.double(), dim=-1)
        probs = lp.exp().numpy()
        shapes.append({"input": list(tokens["input_ids"].shape), "logits": list(logits.shape)})
        for j, r in enumerate(batch):
            pred = int(np.argmax(probs[j]))
            records.append({"idx": r["idx"], "text_sha256": hashlib.sha256(r["sentence"].encode()).hexdigest(), "label": r["label"], "logits": logits[j].tolist(), "probabilities": probs[j].tolist(), "prediction": pred, "confidence": float(probs[j, pred]), "correct": int(pred == r["label"]), "nll": -float(lp[j, r["label"]]), "raw_tokens": lengths[j], "retained_tokens": int(tokens["attention_mask"][j].sum())})
        print(f"{prefix}: {len(records)}/{len(selected)}", flush=True)
        if time.perf_counter() - started > cfg["formal_budget_seconds"]:
            raise RuntimeError("Walltime budget exceeded")
    save(HERE / f"{prefix}_predictions.json", records)
    summary = metrics(records, cfg["ece_bins"], cfg["high_confidence_threshold"])
    summary["inference_seconds"] = time.perf_counter() - inference_start
    summary["total_seconds_after_imports"] = time.perf_counter() - started
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    summary["peak_rss_bytes"] = rss if sys.platform == "darwin" else rss * 1024
    assert summary["peak_rss_bytes"] <= cfg["rss_budget_bytes"]
    summary["truncated"] = sum(r["raw_tokens"] > r["retained_tokens"] for r in records)
    save(HERE / f"{prefix}_summary.json", summary)
    save(HERE / f"{prefix}_environment.json", {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__, "transformers": transformers.__version__, "numpy": np.__version__, "device": "cpu", "dtype": "float32", "attention": "eager", "threads": cfg["threads"], "model_parameters": sum(p.numel() for p in model.parameters()), "shapes": shapes, "cuda_memory_measured": False})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["prepare", "smoke", "run"])
    p.add_argument("--cache", type=Path, default=Path(".cache112"))
    a = p.parse_args()
    cfg = json.loads((HERE / "config.json").read_text())
    if a.action == "prepare":
        prepare(a.cache.resolve(), cfg)
    else:
        infer(a.cache.resolve(), cfg, a.action == "smoke")
