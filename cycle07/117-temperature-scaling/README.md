# Temperature scaling with isolated calibration (117)

Python3.12.14. Install `python -m pip install -r requirements.txt`; full observed dependency versions are in requirements-observed.txt. CPU, ~270MB weights, <6GiB memory budget; real run peak968359936 bytes. No network training or paid service. SST-2 is historically exposed exploratory data, never an unseen test claim.

Run in this directory:
```sh
python run.py prepare --cache .cache117
python run.py smoke --cache .cache117
python run.py run --cache .cache117
python audit.py
```
For offline arithmetic only: `python audit.py` uses bundled derived logits. Full run downloads fixed official resources via prepare, verifies model/tokenizer/data identities, uses256 calibration then360 audit. Original112 exploration256 is unused today. No IMDb predictions. Download reuses existing locked files. Cache and weights are not distributed. No raw text is redistributed because data-card license is unknown.

PROTOCOL.md was frozen before inference. Fit minimizes calibration NLL only; audit never selects parameters. Bounded inverse-temperature bisection100iterations; independent golden-section audit. Outputs: calibration_predictions.json, audit_predictions.json, fit.json (trace), results.json (bins), run_environment.json (shapes/timing), audit_result.json. Expected T2.4819242151, audit330/360 unchanged; NLL.384113→.237136. No confidence interval/generalization claim.

baseline.py is bundled unchanged from112 run.py at commit a57b8bcb005ae9914c2c95f2d855449b7665fd1b/cycle07/112-confidence-baseline, used for prepare, metrics and hash I/O; run.py is the new entry. Config explicitly retains original selection description for provenance, but117 pool selection comes ONLY from split_manifest.json. SOURCE_MAP.md maps implementations and changes. Formula code is independent educational code; upstream neural network runs in installed official Transformers. No hidden local modules or absolute machine paths required.

Published execution.json records commands with portable cache name, actual exit codes and raw stdout; local verification retains full command paths. Prior missing dependency/stalled environment failures are recorded outside distribution. Run in a fresh copy to preserve archived evidence: inference commands overwrite corresponding outputs. Cached inputs include exact resource hashes, model and tokenizer revision, config, split, run source and protocol identities; no prompt or decoding for this classification head.
