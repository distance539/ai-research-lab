# Independent confirmation (141)
Python 3.12. Install `python -m pip install -r requirements.txt`.
Minimal offline entry: `python confirm.py` then `python audit141.py`.
Outputs confirmation_results.json, confirmation_predictions.json and audit141_result.json. Real saved1000 IMDb logits are provided; no fitting or model inference occurs in the offline entry.

To reproduce model outputs in a fresh copy, first move the distributed formal.json and formal_env.json to a separate reference directory (do not discard them), then run `python run.py fetch`, `python run.py smoke`, `python run.py formal`. Fetch downloads hash-pinned resources (~350MB); allow2GB RAM and1GB disk. Optional `--archive PATH --cache PATH` accepts verified existing resources. Formal refuses to overwrite existing output. Original run used one1000-review confirmation pass and8 exploration smoke rows. CPU float32/eager, batch16,4threads,512tokens.

final_config.json/final_lock.json are unchanged from140 public commit d8dccb42173346f6299f71b187f15d75301864a5. release141.json records the before-confirmation adapter/source hashes. The historical config says confirmation disabled for140; only141 releases the fixed confirmation pool. No temperature refit. analyze.py is retained unchanged for its evaluate function; do not run its historical development-only main (development data intentionally not bundled).

`python plot.py` creates ../images/confirmation_results.png (create ../images first). Quantitative graph reads saved actual results only. Independent audit checks2000 probability/loss rows and10000 bootstrap replicates. First smoke failed before inference due to adapter indentation; failure evidence retained, fixed before confirmation. Dependencies and measured environment are included. No human semantic review claimed.

Raw reviews and weights are excluded. Only SHA identities, official binary labels, token lengths, model floats and derived metrics are distributed. Model Apache2; self-authored code MIT. SOURCE_MAP.md gives pinned upstream attribution. Original confirmation must not be used again for method selection.
