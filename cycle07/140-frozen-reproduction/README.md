# Frozen reproduction package (140)
Python 3.12. Install `python -m pip install -r requirements.txt`.
Minimal offline entry: `python reproduce.py`. It verifies final_lock.json, copies into a fresh temporary directory, refits the predeclared 128/1000 calibration conditions, recomputes exploration metrics, runs independent arithmetic verification and eight damage tests. Predictions/results must be byte-identical. Fits wall time is measured, not compared. Outputs reproduction_result.json, audit_result.json, damage_result.json.

Fresh real model smoke: `python run.py fetch` then `python run.py smoke`; optional `--cache PATH --archive PATH` uses hash-verified existing resources. About 350MB download, 1GB disk, 2GB RAM. No API key. Only exploration is allowed; no confirmation inference entry exists in this release. The 141 entry must preserve final_config.json and all metric definitions; any adapter changes must be recorded before inference.

final_lock.json freezes all source/config/data and deterministic reference outputs; its external SHA is in the article metadata. Do not regenerate it to make verification pass. This detects accidental corruption, not an attacker who replaces both package and trusted hash. Seven damage tests exercise file integrity, one exercises independent arithmetic. They are not seven semantic model tests.

Derived rows from 132/135 and protocol137 are retained; inherited four-cell analysis/audit from139 fixed commit b6eb890e51f34877e572bd8daafa1e5a609ff0f3. No raw review text, weights, secrets or confirmation predictions distributed. All self-authored modules included. Source/map/license details in SOURCE_MAP.md and LICENSE.
