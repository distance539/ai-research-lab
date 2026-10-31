# High-confidence error audit (115)

Run `python analyze.py && python check_analysis.py` with Python standard library (observed3.9.6). Outputs group_results.json (12 groups and top5 errors), group_audit.json, analysis_environment.json. No fitting or semantic relabeling. Protocol: ANALYSIS_PROTOCOL.md; fixed input hash: analysis_config.json.

## Real model
Install requirements.txt with Python3.12. Run `python execute.py prepare --cache .cache112`, then `python execute.py smoke --cache .cache112`. For256 rows use `python execute.py formal --cache .cache112` in a fresh copy. Run `python verify_inputs.py --cache .cache112`. Approx270MB model, additional dependencies and cache storage. CPU float32, no training. Cache locks include exact model/tokenizer/data versions and config/protocol. No hidden local module dependencies.

Formal256 predictions and baseline modules are inherited byte-for-byte from112 at a57b8bcb005ae9914c2c95f2d855449b7665fd1b/cycle07/112-confidence-baseline. New work: grouped analysis and fresh8 real-model replay. SST-2 is historically exposed exploration, not unseen testing. Other SST pools and IMDb not predicted. Original PROTOCOL.md remains unchanged. No raw data text distributed (data license unknown); download fixed official resource.

Plot: install requirements-plot.txt then `python plot.py output.png`; reads group_results.json. Numeric graph is not image generated. See SOURCE_MAP.md, LICENSE, resource and environment records.
