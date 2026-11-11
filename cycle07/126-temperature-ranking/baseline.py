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
    expected = json.loads((HERE / "resources.lock.json").read_text())
    assert digest(data) == expected['dataset']['sha256']
    assert {f['name']: f['sha256'] for f in locks} == {f['name']: f['sha256'] for f in expected['files']}
    print("Frozen resources verified")
