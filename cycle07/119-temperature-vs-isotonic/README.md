# Temperature versus isotonic (119)

Python3.12: `python -m pip install -r requirements.txt`, then `python run.py` and `python audit.py`.
Outputs: results.json (all probabilities, knots, IDs, metrics), analysis_environment.json, audit_result.json.
Frozen real DistilBERT/SST2 calibration256 and already-observed audit360 logits inherited from117/118. Same hash-prefix32/256 IDs for both methods. Primary class-summed Brier; NLL clipped at preregistered1e-6. Endpoint counts and zero true probability counts retained. Stable logit-space raw/temperature NLL additionally distinguishes float saturation from mathematical infinities. No audit selection, no new semantics, no IMDb.

Optional fresh real model replay:
```
python model_run.py prepare --cache .cache
python model_run.py smoke --cache .cache
```
Approx270MB model, CPU4threads,6GiB and30minute budget. Fixed revisions and all resource hashes checked. No raw dataset text redistributed; official license unknown. This run freshly replayed8 examples and reused616 archived model predictions; no training. Cache key includes model/tokenizer files, raw text hashes, configuration, split and model_run code hashes in smoke_start.json; classifier has no prompt or decoder. Analysis inputs locked in protocol.json. Keep copies because commands overwrite results.

All self-authored modules included. Optional model_run.py run regenerates616 inherited117 predictions; do this in a separate copy because exact-byte registered input checks may reject platform rounding. run.py requires archived identities. Independent audit uses min-max block means, not the library fitter. Different fitting objectives (NLL versus squared error) mean this is a method comparison, not solely parameter-count intervention. Single32 subset does not estimate sampling variance.
