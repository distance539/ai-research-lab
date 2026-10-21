# Local circuit evidence audit — GPT-2 / 256 synthetic IOI prompts

This package asks whether the three frozen END nodes L8H10/L8H6/L9H9 approximate the full model and its response to further ablations. All non-END positions, MLPs, residual paths, biases and normalization remain active. **This is not a complete circuit extraction or original-paper reproduction.** All256 prompts were previously observed in104; this is exploratory and uses no fresh confirmation data.

## Setup and minimum real run

Python3.12.14 was tested on macOS arm64, CPU float32, four threads. Create an isolated environment and install the exact tested dependency set:

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r environment.lock.txt
python run.py --cache ./cache --out smoke-new --smoke
```

`requirements.txt` names the three direct dependencies; `environment.lock.txt` pins all25 installed packages, with equality evidence in environment_verification.json. Cross-platform packages/bitwise output are not guaranteed. Source uses only torch/transformers/numpy plus standard library; no implicit author paths or parent imports.

First run downloads eight SHA-checked files at fixed GPT2/tokenizer revision607a30d783dfa663caf39e06633721c8d4cfcd7e (about550MB; allow1GB disk and4GiB RAM). A previously verified cache can be passed by `--cache`; every entry rehashes it. No inference output cache is consumed. No API key or private credentials needed. Avoid reusing an existing output directory; the entry intentionally refuses overwrite.

For the full run and separate validation:

```sh
python run.py --cache ./cache --out full-new
python audit.py --results full-new --out full-new/arithmetic_audit.json
python replay.py --cache ./cache --results full-new --out full-new/replay.json
python test_checks.py
python rebuild_samples.py --out rebuilt_samples.json
```

`rebuild_samples.py` downloads fixed official literals, checks their SHA and reconstructs all256 samples without importing upstream runtime. Alternatively pass `--source` with a downloaded byte-identical ioi_dataset.py. `test_checks.py` uses distributed smoke evidence and temporary mutated copies.

## Protocol and outputs

Read PROTOCOL.md and immutable plan.json first. Universe =144 END attention outputs; background remains active. H has three fixed nodes. Two references (zero, six distinct-name permutations per template mean) × eight removed subsets K × two systems (full-minus-K vs background-plus-H-minus-K). The empty full arm repeats clean so there are31 unique scored forwards per row, though33 score objects are saved.24 reference forwards; four controls per row. Kept nodes recompute after upstream changes; clean caches are used only for self-controls.

`records.json` includes every prompt, IDs, IO/S scores, probability, argmax, controls and timing; `means.json` includes four[12,768] END banks; `references.json` holds24 unlabelled ABC prompts. `summary.json` reports all subset comparisons, MAE, top1 counts and prespecified largest-residual examples. `identity.json`, `stages.json`, `runtime.json`, `environment.json`, `execution.json` record identities, stages and measured costs. This is synthetic public-template material, not private or natural corpus data.

Formal run:256 rows,7936 scored forwards +1024controls +24references;230.898s process wall time,1465073664B peak RSS. All model tokens15/18/19/20, complete logits[1,T,50257]. No generation/training/truncation/BOS. Existing model cache reused and rehashed. Smoke estimated192.405s; actual full cost was higher and remains under900s budget.

Full means by old_old/new_old/old_new/new_new:4.446/2.947/3.877/2.479; retained zero:0.125/0.061/0.115/0.046, all top1 zero. Retained mean:2.608/1.688/1.976/1.287, top1 24/0/15/0. Under mean replacement, largest incompleteness gap occurs at removing C in all four cells;old_old gap1.838 at empty K grows to3.559 at C. Small/reference-specific, dependent synthetic design; no iid confidence interval or general completeness claim.

## Reuse and verification

Parent104 fixed commitc22a61b35fd75eae8c753be427f3f2c89117efbb is recorded in config.json. prepare/resources/samples/rebuilder/licenses/dependencies are included, so no parent runtime download is needed. parent_config/summary and parent_verification preserve ancestry. SOURCE_MAP.md identifies official lines and adaptations. Numerical and negative checks are independent from analyze.py; replay.py uses postprojection dense deltas instead of preprojection slices and independently verifies all argmaxes and means. See VALIDATION.md. All results are real runs, not paper scores.
