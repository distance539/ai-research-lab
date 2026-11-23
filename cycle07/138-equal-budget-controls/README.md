# Equal-budget controls (138)
Python3.12; install `python -m pip install -r requirements.txt`.
Run from this directory: `python analyze.py && python audit.py`.
This refits on archived1000 IMDb calibration outputs and evaluates1000 archived exploration outputs; it overwrites fits/results/predictions/audit records. Confirmation outputs are never used.
`python plot.py` writes ../images/risk_controls.png (create ../images first).
For real model replay: `python run.py fetch`, then `python run.py smoke`.
Optional validated caches: `python run.py smoke --cache MODEL_CACHE --archive IMDB_V1`.
MODEL_CACHE contains revision714eb0fa89d2f80546fda750413ed43d93601a13; archive is aclImdb_v1.tar.gz. Defaults download about350MB; allow2GB RAM and1GB disk. No GPU needed. Fixed resource hashes checked before use. No hidden local modules.
Actual run: CPU8-review smoke, full1000 calibration refit per method,1000 exploration paired evaluation,10000 bootstrap replicates. No new1000-review inference, no training, no confirmation. Tolerance1e-5 vs historicalbatch16; actual smoke batch8.
Protocol copied unchanged from137 (SHA in protocol.sha256). Length<=256/>256 retained tokens; fallback if either group<50.100 bisection iterations plus2 boundary checks: each method102000 sample-gradient evaluations, not identical scalar calls or wall time. Temperatures2.389598859 global;2.424306257/2.346367586 grouped. DeltaNLL+0.000168592; exploratory CI crosses zero. Risk80 errors45 vs46. This is exploratory, not independent confirmation.
Data provenance: pools locked in132 before any forward; exploration from132, calibration from135, copied through137. Only hashes, official binary labels, retained lengths and logits redistributed. No raw text, weights or secrets. See SOURCE_MAP.md and resources.lock.json. Source/model licenses retained; IMDb raw redistribution not assumed. Independent audit is arithmetic verification, not human semantic review.
