# Paired IMDb truncation budgets
Python 3.12 / CPU; install `python -m pip install -r requirements-observed.txt`. Allow 6 GiB RAM and about 1 GiB disk beyond the environment. Run from this directory.

Smallest offline reproduction: `python run.py analyze && python audit.py`.

Fresh end-to-end reproduction:
```sh
python common.py fetch
python run.py smoke
python run.py formal
python run.py analyze
python audit.py
python plot.py results.png
```
Both common.py and run.py accept --cache MODEL_CACHE --archive IMDB_V1 to reuse verified resources. Defaults: cache/ and aclImdb_v1.tar.gz. Do not rebuild pools.json: this is the immutable registration from article 132, created before its first inference. Only exploration IDs are scored. No calibration or confirmation inference. Archive checksum and model five-file hashes are checked before loading. common.py is the complete self-contained inherited loader, not an external path dependency; its old analyze/register/formal actions belong to 132 and are not part of this workflow.

New intervention: max_length 128/256/512, including two special tokens. All three conditions freshly inferred, 8 smoke and 1000 formal per condition. Full untruncated tokenization is used only for accounting, never fed to the model. run.py asserts actual input equals the appropriate token prefix. Classification uses original full-review labels, not newly validated prefix labels. Right truncation, FP32/eager, eval mode, 4 CPU threads, batch 16 dynamic padding, no temperature fitting or model selection. Untruncated tokenizer warning is expected during counting. Timings cover tokenization/batching/forward loops once, with fixed ascending order; peak RSS is process cumulative, no per-condition memory claim. No human semantic annotations.

formal.json stores 3000 identity/label/logit/input-digest/token-accounting rows; formal_env.json records shapes, versions, time and memory. results.json and independent audit.log are reproducible from this package without network. execution.json records commands, exit codes and actual execution timestamps. plot.py reads results.json and formal_env.json, no fabricated chart inputs. Original reviews/weights/caches are excluded. Source distribution is MIT; upstream attribution in SOURCE_MAP.md and THIRD_PARTY_LICENSE.txt.
