# Classifier versus finite candidate confidence

Python3.12.14; install `pip install -r requirements.txt`. CPU float32/eager, 4threads. Model approx270MB; allow 3GB disk plus dependencies and 2GB RAM (measured process peak1.585GB). No training or API key.

```
python prepare.py --cache .cache
python run.py --model-dir .cache/model --data .cache/validation.parquet --smoke
python audit.py
python check_forward.py --model-dir .cache/model --data .cache/validation.parquet
```

Remove --smoke for formal256. Outputs formal_candidate_predictions.json, formal_comparison.json, formal_start.json; smoke uses separate filenames. Run in a copy to preserve frozen evidence. `audit.py` independently checks the bundled256 formal records and aggregates; `check_forward.py` reruns8 prompts through a distinct full-vocab softmax path. Expected candidate correct130/256, all256negative, mean confidence0.87164434, candidate mass0.46836851. This is a failed raw-prompt interface, not optimal chat performance. No open generation or semantic annotation.

DistilBERT predictions, split manifest and resource lock inherited exactly from112 commit a57b8bcb005ae9914c2c95f2d855449b7665fd1b, cycle07/112-confidence-baseline. All other candidate records newly executed. SST2 historical exposure; no other SST pools/IMDb inference. Unknown data license: no raw review text redistributed. Download fixed official data with hashes. Model Apache2.0. Our scripts independent MIT orchestration; external models/libraries remain separately licensed.

Config and protocol frozen before scoring. `execution.json` records actual commands, return codes and measured scope. Start records hash model/tokenizer, prompt/config, code, protocol; input hash binds dataset identities. Late addition of model-lock assertion only validates previously hashed files; final code rerun after this guard. No model weights/cache/credentials in source distribution.
