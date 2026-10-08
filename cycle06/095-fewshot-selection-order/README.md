# Few-shot selection and order: reproducible experiment

Article095. Original SmolLM2-135M-Instruct, GLUE SST2, five frozen conditions on the same64 development questions. Small exploratory experiment, not a full benchmark, new method or held-out confirmation. Weights unchanged.

## Environment and commands

Tested Python3.12.14, macOS26.6 arm64, CPU float32, four threads. requirements.txt pins main dependencies; environment.lock.txt records the installed environment. Harness installed from a fixed Git commit. Other platforms may differ numerically; likelihood parity tolerance2e-5.

~~~sh
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python prepare.py --cache ./cache
python run.py --cache ./cache --plan-only
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false python run.py --cache ./cache --out smoke-new --private private-smoke --smoke
python audit.py --cache ./cache --out smoke-new
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false python run.py --cache ./cache --out full-new --private private-full
python audit.py --cache ./cache --out full-new
python -m unittest test_audit -v
~~~

Output/private directories must be new; scripts refuse overwrite. Included plan.json is checked against regenerated selection. All local modules included, no implicit prior-article dependencies. prepare.py downloads pinned public files and verifies hashes; no API key. Weights269060552bytes, plus tokenizer/data/dependencies. Reserve4GiB process RSS and1200s on the measured class of CPU, not a hardware-independent guarantee. No GPU required.

## Actual execution

Conditions: zero; A_forward/A_reverse; B_forward/B_reverse. Two train examples per label/set, matched42-token blocks, no normalized exact text overlap with any validation row. Actual whole contexts191-233tokens for few-shot,19-61forzero. Each few-shot condition scores26522 candidate input tokens versus4506forzero.

Correct counts32/31/37/33/44 out of64. A order reversal changes6predictions (all wrong-to-right), B reversal11; zero-to-A_forward has1gain/2losses. All640 official/independent candidate scores agree exactly. Formal runtime102.767s after imports, peakRSS1994588160bytes, details in results/environment.json. No inference on128reserved SST2 or prior64AGNews questions; repeatedly used development data is not an unseen test.

Three smoke attempts each evaluated2questions x5conditions: first two exited1 due resource projections, not score disagreement. Final smoke passed after estimator correction and600->1200s CPU budget amendment. See DEVIATIONS.md and retained failed folders. No prompt/condition selected by smoke accuracy.

## Files

- PROTOCOL.md: design and operational amendments before formal inference.
- plan.json/config.json/resources.lock.json: identities, order, resource versions/hashes.
- run.py: official sampler subclass/simple_evaluate and independent Transformer likelihood.
- audit.py: independent arithmetic, pairing, source identity and token-boundary replay.
- test_audit.py:8tests with deliberate corruption, uses included formal results.
- results/: full derived records, summary, runtime, stages and identity.
- smoke/: successful real small run; two failed folders retained separately.
- SOURCE_MAP.md/upstream_verification.json/THIRD_PARTY.md/licenses: provenance.
- commands.json/VALIDATION.md: execution records. Raw logs containing filesystem paths stay in local article verification area, not public distribution.

Original reviews, full prompts/context IDs, raw harness returns, weights, caches, credentials and environments are not redistributed. Public derived candidate scores, answer token IDs, labels, hashes and row indices support audits/regeneration from pinned downloads. Human semantic annotation was not performed or claimed; deterministic checks are machine recomputation, not human gold labels.
