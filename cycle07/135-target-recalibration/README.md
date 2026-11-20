# Target recalibration (135)
Frozen IMDb calibration pool, seven nested prefixes, scalar temperature fit. Exploration logits reused unchanged from132; no confirmation outputs.
Python3.12.14, torch2.6.0, transformers4.49.0. Install `pip install -r requirements.txt`.
```
python run.py fetch --cache cache --archive aclImdb_v1.tar.gz
python run.py smoke --cache cache --archive aclImdb_v1.tar.gz
python run.py formal --cache cache --archive aclImdb_v1.tar.gz
python analyze.py
python audit.py
python plot.py
```
Minimal offline replay: `python analyze.py && python audit.py`. Outputs results.json and audit_result.json. plot.py writes budget_curve.png in current directory. About268MB model plus84MB compressed archive, CPU4 threads, float32, eager, max_length512, batch16, seed135. Registration is frozen; do not regenerate pools. SHA256 of inputs and code is recorded in results and environment. Runtime details in formal_env.json. Only first8 calibration rows for smoke, then1000 calibration rows formal; formal stage checks smoke resource estimate. No training, no new exploration forward, no final confirmation. Raw review text and weights excluded. Actual logs and derived per-row outputs included.

Nested prefixes16/32/64/128/256/512/1000 are one correlated trajectory, not independent repetitions or a population sample-size guarantee. beta bounded [.05,20],100 bisections, all boundary flags preserved. Primary exploration NLL; Brier sums both classes; ECE10 equal-width bins. Source T2.481924215086295 was fitted on historical SST2. All conditions890/1000 hits. No budget selected by exploration performance. Independent audit uses golden search and two-class logsumexp without importing analyze.py. This is automatic arithmetic review, not human semantic review. Local code MIT; third party attributions in SOURCE_MAP.
