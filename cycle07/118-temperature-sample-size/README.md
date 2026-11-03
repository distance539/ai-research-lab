# Small calibration sample stability (118)

Python 3.9+ standard library for analysis: `python run.py` then `python audit.py`.
Expected outputs runs.json (250 fits), summary.json, analysis_environment.json and audit_result.json. Fixed real117 logits: calibration256 / already observed audit360. Fifty hash permutations, nested16/32/64/128/256 subsets. No audit-selected temperature. Expected T SD1.42835/.70605/.36192/.19940/~0;16-row boundary10/50,32-row1/50. Finite-pool conditional dispersion, not population confidence intervals.

Real model replay requires Python3.12.14 and `python -m pip install -r requirements.txt`:
```
python model_run.py prepare --cache .cache
python model_run.py smoke --cache .cache
python audit.py
```
Optional `python model_run.py run --cache .cache` regenerates all616 frozen predictions and inherited117 fit/results; then `python run.py` checks their exact registered identities. Approx270MB model, CPU4threads,6GiB/30minute budget. Cached resources must match resources.lock.json. This run reused616 previously measured logits and freshly replayed8 calibration examples; not616 new forwards. No IMDb inference. Raw SST text not redistributed (dataset license unknown); prepare downloads fixed official revision.

model_run.py bundles117 code with only source-identity pathname corrected; baseline.py bundles112 helper. Config seed117 applies to unchanged model replay; protocol.json seeds11800..11849 apply to new subsampling. No hidden modules. SOURCE_MAP.md records exact provenance. Same checkpoint/tokenizer/split/truncation/float32/eager maintained. Archived results will be overwritten when rerun; use a copy. requirements-observed.txt records actual model runtime packages, analysis_environment.json the separate stdlib interpreter. Download timing not measured in118 because verified cache reused. No training or semantic relabeling.
