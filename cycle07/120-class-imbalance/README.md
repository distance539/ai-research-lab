# Class imbalance and global temperature (120)

Run `python run.py` then `python audit.py` (standard library, Python3.9+). Output results.json contains60fits with100IDs, full optimization traces and calibration/audit/by-label metrics; summary.json contains mean/sampleSD/min/max. audit_result.json independently verifies all fits by golden-section minimization and scalar identities. Protocol frozen before analysis; all outcomes retained. Positive counts10/50/90 with100total,20paired hash orderings12000..12019. This estimates sensitivity within a fixed finite pool, not population confidence intervals. No model selection, new semantic labels or IMDb use.

Real model replay (Python3.12, CPU, about270MB weights,6GiB/30minute budget):
```
python -m pip install -r requirements.txt
python model_run.py prepare --cache .cache
python model_run.py smoke --cache .cache
```
The8row smoke checks the actual classifier;616formal predictions are inherited from117, not newly inferred. Fixed model/tokenizer/data revisions and SHA256 are in config.json/resources.lock.json. Existing validated cache can be reused. Do not run the inherited optional `model_run.py run` in the release directory: it reproduces117 and overwrites input/output files. Use a separate copy. Analysis verifies exact archived input bytes.

SST2 has historical exposure; audit360 already observed development evidence, not fresh confirmation. No raw dataset text/weights/cache distributed. Download fixed data revision; official data license unknown. All self-authored modules included under LICENSE. SOURCE_MAP.md identifies inherited modules, official sources and intentional solver differences. requirements-observed.txt records the replay environment. Analysis uses system Python separately, recorded in analysis_environment.json. Runtime excludes imports as named; service memory not measured.

Cache identity: smoke_start.json records model/tokenizer/data hash lock, split/config/model_run code hashes and protocol; no prompt/decoder for classifier. protocol.json locks analysis input artifacts. Commands overwrite generated results; copy directory for repeat runs.
