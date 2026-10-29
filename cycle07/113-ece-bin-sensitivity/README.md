# ECE bin sensitivity on fixed real predictions

Minimal analysis: `python analyze.py && python check_analysis.py`. Python standard library only (observed3.9.6); 256 real frozen DistilBERT predictions are included without raw text. Outputs: analysis_results.json, every paired draw and ECE, runtime and independent audit. This is a measurement experiment, not a new model benchmark. All four bin counts are reported, no selection.

For plotting install matplotlib3.10.8 and run `python plot.py ece_results.png`. See requirements-plot.txt. Wilson intervals are pointwise descriptive; bootstrap percentiles are conditional empirical sensitivity, not guaranteed coverage.

## Real model replay
See baseline_README.md and requirements.txt (Python3.12). `python execute.py prepare --cache .cache112` downloads pinned model/data, then `python execute.py smoke --cache .cache112` replays the first8 exploratory inputs. The cache is verified with `python verify_inputs.py --cache .cache112`. Full inference is available with `python execute.py formal --cache .cache112`, but would overwrite inherited baseline outputs; use a fresh copy. It is not required for the fixed-prediction comparison.

The original256 inference and its formal records are inherited byte-for-byte from112 fixed commit a57b8bcb005ae9914c2c95f2d855449b7665fd1b. This article executed a fresh8-row CPU replay and new paired bootstrap analysis. The remaining SST-2 pools and all IMDb data are untouched. Prior SST-2 exposure is disclosed; no claim of fresh confirmation.

## Evidence
PROTOCOL.md and analysis_config.json were frozen before analysis. Formal baseline files retain original timestamps; smoke files reflect this article's replay. analyze.py.log and check_analysis.py.log plus execution JSON record new commands and exit status. Source map distinguishes official implementation, inherited implementation and new independent analysis. No API keys, training, manual semantic labels or paid resources are needed.
