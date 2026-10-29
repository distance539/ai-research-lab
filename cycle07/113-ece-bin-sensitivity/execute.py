"""Capture exit status and raw output for a reproducible local command."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

here = Path(__file__).resolve().parent
name, *args = sys.argv[1:]
assert name in {"prepare", "smoke", "formal", "audit", "smoke_audit"}
if name in {"audit", "smoke_audit"}:
    command = [sys.executable, "audit.py", "--prefix", "smoke" if name == "smoke_audit" else "formal"]
else:
    command = [sys.executable, "-u", "run.py", "run" if name == "formal" else name, *args]
started = time.time()
env = dict(os.environ)
if name != "prepare":
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
with (here / f"{name}_execution.log").open("w") as log:
    try:
        result = subprocess.run(command, cwd=here, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1900)
        code = result.returncode
    except subprocess.TimeoutExpired:
        code = 124
public_command = ["python", *command[1:]]
if "--cache" in public_command:
    public_command[public_command.index("--cache") + 1] = "<RESOURCE_CACHE>"
record = {"command": public_command, "cache_path_redacted": "--cache" in public_command, "started_unix": started, "elapsed_seconds": time.time() - started, "exit_code": code, "output": f"{name}_execution.log"}
(here / f"{name}_execution.json").write_text(json.dumps(record, indent=2) + "\n")
print((here / f"{name}_execution.log").read_text())
print(json.dumps(record))
sys.exit(code)
