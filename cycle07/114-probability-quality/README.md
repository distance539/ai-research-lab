# Probability quality under fixed positive temperatures

Run `python analyze.py && python check_analysis.py` (Python standard library, observed 3.9.6). Outputs: temperature_predictions.json (1024 rows), temperature_results.json, temperature_audit.json, analysis_environment.json. No fitting or parameter selection: all T=0.5,1,2,4 reported on the same historical exploratory SST-2 predictions.

## Real model reproduction
Install requirements.txt with Python3.12. `python execute.py prepare --cache .cache112` downloads exact revisions and hashes. `python execute.py smoke --cache .cache112` runs eight real examples; `python execute.py formal --cache .cache112` reproduces256 in a fresh copy. `python verify_inputs.py --cache .cache112` audits identities. No implicit local module dependencies. CPU, approximately270MB model; full dependencies/cache require additional storage. No training.

Formal inference files are inherited byte-for-byte from112 at a57b8bcb005ae9914c2c95f2d855449b7665fd1b/cycle07/112-confidence-baseline. This article adds four-temperature analysis and fresh8 example replay. Original PROTOCOL.md remains the baseline protocol; ANALYSIS_PROTOCOL.md is the new frozen protocol. SST-2 was used historically. Neither other SST-2 pools nor IMDb are predicted. Data license unknown: raw sentences are not distributed.

Plot: install requirements-plot.txt, run `python plot.py temperature_results.png`. No artificial error bars. All numerical verification is automated arithmetic, not human semantic review. See SOURCE_MAP.md and LICENSE.
