# 099 做了十种设置才找到提升：如何处理选择偏差？

Scope: real SmolLM2-135M-Instruct, ten protocols×64 fixed development rows, then one frozen winner versus zero baseline×128 previously reserved SST2 validation rows. No training. Historical nine protocols were already seen; the ten-candidate list is retrospective, the confirmation is prospective. This is not the official full SST2 test, and not a ten-model comparison.

## Fast independent offline audit

```sh
python audit.py
```

Uses only Python standard library, validates all640 development/256 confirmation rows, token likelihood sums, argmax/semantic label mapping, selection counts, disjoint split and frozen source hashes; derives an exact empirical paired-bootstrap distribution. Does not provide human semantic labels.

## Exact environment and statistical replay

Use Python3.12.14; direct versions requirements.txt, complete resolved versions environment.lock.txt. Tested macOS/arm64 CPUfloat32/eager4threads. Hardware/library differences may change numerical results and require inspection.

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r environment.lock.txt
python analyze.py --out analysis-replay.json
python test_checks.py
```

Expected: analysis-replay.json with paired delta, gains/losses and percentile interval; test_results.json with eight corruption/identity tests. The test command rewrites its deterministic test evidence only.

## Real minimal end-to-end entry

```sh
python prepare.py --cache ./cache
TOKENIZERS_PARALLELISM=false python run.py --cache ./cache --phase development --smoke --out smoke-new
```

Downloads only fixed public model/data files and checks12SHA256s. Cache approximately280MB plus environment; memory limit4GiB per process, runtime limit1200s, max input512 with truncation refusal. Smoke uses only2development rows×10conditions×2candidate answers. Outputs records.json, summary.json, identity.json, stages.json and environment.json; stdout contains condition timings. Candidate scores computed through official HFLM.loglikelihood and independent direct forward pass; ConfigurableTask.process_results checks correctness. No hidden parent modules, no API keys, no private resources.

## Full recorded workflow

```sh
python run.py --cache ./cache --phase development --out development-new
python select.py --development development-new --out selection-replay.json
python run.py --cache ./cache --phase confirmation --out confirmation-replay
python analyze.py --records confirmation-replay/records.json --out confirmation-replay/analysis.json
```

Full confirmation command uses shipped immutable selection.json and validates source/config/protocol identities. Compare selection-replay.json winner/counts and input hashes with original; its timestamp naturally differs. select.py refuses overwrite. Changing scientific code requires a NEW protocol, development search and an independently reserved set, not editing shipped hashes to pass. Re-running confirmation only reproduces a now-consumed set; it cannot be described as a new blind test. No confirmation smoke is allowed. Winner selection occurs only in select.py, never in confirmation.

## Evidence

PROTOCOL.md/config.json/plan.json precede real inference. selection.json includes original freeze time, ten results and all frozen hashes. development/ and confirmation/ preserve per-row source identity, scores, predictions and runtime; commands.json and stdout.log preserve actual execution and exit codes. No source reviews/full prompts/weights/caches or private paths are distributed. See SOURCE_MAP.md for fixed upstream functions, THIRD_PARTY.md/licenses, DEVIATIONS.md for failures and environment repairs. Files listed in MANIFEST.json; local article code.zip is an archive, public browsable source is the reader-facing delivery.

Bootstrap uses SciPy1.13.1 paired=True, percentile,20000,Generator seed9900. Direct shared-index resamples match every SciPy replicate; audit.py independently convolves the empirical difference distribution. Confidence interval conditions on fixed selected protocol and independent rows, not the full search process, model-seed population or unknown movie clustering. Differences between development and confirmation are not automatically attributable entirely to selection bias. News64 remains unused. No097human gold statistics are fabricated or used.
