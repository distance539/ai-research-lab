# END-only ablation baselines: 16 real GPT-2 IOI inputs

This is an exploratory mechanism experiment, not a complete circuit or natural benchmark. Frozen GPT-2 small(124,439,808parameters) runs through official Transformers4.49.0, torch2.6.0, CPUfloat32/eager,4threads,batch1. No training/generation. Raw records,12ABC reference activations,21conditions/sample and all controls are included.

## Reproduce

Python3.12.14 was used. A fresh environment and roughly1GBdownload/4GiBRAM are sufficient for the measured scope; other machines' speed is unverified.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r environment.lock.txt
python run.py --cache ./cache --out smoke-new --smoke
python audit.py smoke-new
python run.py --cache ./cache --out results-new
python audit.py results-new
python test_checks.py
```

Minimal model entry: `python run.py --cache ./cache --out smoke-new --smoke` after installing dependencies. It downloads missing immutable public files, verifies all8hashes, runs12ABC reference forwards and21interventions on one target plus implementation controls. Out directory must not already exist. Complete run is16targets/336interventions. Records,reference_bank,summary,identity,runtime,environment,stages JSON appear in output; terminal prints progress. Included execution.json/stdout.log record the author's actual subprocess and exit status. Runtime is measured inside run.py after standard-library imports; subprocess wall time is separately recorded. Shared model cache was reused with SHA verification; downloading again is unnecessary.

Offline independent check: `python audit.py results` (standard library only). It rechecks inputs, reference permutations/vectors, scalar logits,all condition means/counts and paired interaction arithmetic. `python test_checks.py` verifies7deliberate corruptions are caught. Full-vocabulary argmax cannot be reconstructed from two candidate scores alone; the fresh-package/public model replay checks saved argmax/probability and candidate outputs.

Public template reconstruction: `python rebuild_samples.py --out samples-rebuilt.json`; optional `--source` accepts a previously downloaded SHA-checked official file. prepare.py is also a standalone downloader: `python prepare.py --cache ./cache`.

## Design and interpretation

Read PROTOCOL.md then SOURCE_MAP.md. A=L8H10,B=L8H6,C=L9H9 are prior102's three highest s2_flip END restoration heads. Selection was on these same developmental rows. Seven nonempty subsets x zero/mean_abc/resample_abc,END only. References enumerate all6permutations of3distinct names per template. Fixed resample mapping(seed103) is one assignment, not a Monte Carlo average. No heldout names/templates are inferred. A,B,C in the factorial code denote heads; ABC in reference bank denotes three distinct name slots, a separate convention.

Primary result: cleanD4.295346; ABdrops2.340314/2.492823/2.452782 for zero/mean/resample. I_AB negative in all3(-.293791/-.275505/-.261606), so effects are subadditive on this metric. Conditional C afterAB is-.059575/.044491/.024631, compared with singleC .320334/.702054/.666513. Nonadditivity is not proof of a backup circuit. Layernorm/nonlinearity/saturation and downstream changes are competing explanations. Full table/paired cells in results/summary.json and records.json. No IID CI or population claim.

## Integrity and distribution

plan.json was frozen before smoke/formal inference, covers code/config/resources/protocol/parent evidence. Cache identity includes plan+resources+mode; no output cache reuse. MANIFEST.json covers every distributed file except itself. All necessary selfwritten modules,licenses,configs,versions and evidence are included. No weights,venvs,caches,credentials or private machine paths. Synthetic public-template prompts and derived activations are distributable with attribution. Original full upstream repository is not repackaged. LICENSE.txt applies only to our original code. See THIRD_PARTY.md and VALIDATION.md.
