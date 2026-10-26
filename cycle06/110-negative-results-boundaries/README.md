# 110 — Development uncertainty and failure boundaries

100 previously observed SciFact development queries;5183documents;50fixed candidates; one actual fresh MiniLM CE scoring pass, pureRRF/pureCE/fixed equalweight rank blend, followed by paired component and query bootstrap. No training, new model, query exclusion or confirmation evaluation.

## Run

Tested Python3.12.14 on macOSarm64, CPUfloat32. `requirements.txt` pins direct dependencies; `requirements-observed.txt` contains the full tested freeze. Use a fresh environment and allow about1GB resident memory plus environment/data/model space. Saved measured environment is authoritative, not an installation guarantee on every platform.

```bash
python -m pip install -r requirements.txt
python prepare.py --cache ./cache
python run.py --cache ./cache --out fresh-smoke --smoke
python run.py --cache ./cache --out fresh-results
python audit.py --out fresh-results
python ce_audit.py --out fresh-results
python audit_statistics.py --out fresh-results
python test_audit.py --out fresh-results
python test_statistics.py --out fresh-results
```

Minimum real model command after setup: `python run.py --cache ./cache --out fresh-smoke --smoke` runs5queries×10candidates. Output directory must be NEW. `prepare.py` retrieves fixedrevision model/tokenizer and hash-identified BEIR data, or verifies an existing cache. No hidden author paths. Model files ~91MB plus tokenizer; dataset archive2.8MB. Initial retrieval is not repeated: verified historical ranking inputs are bundled with licenses and source URLs. Confirmation qrels are not loaded by experiment code.

Saved `results/` contains this run's5000pair CSV, runs, metric/fusion traces, resource and identity files, all20000draws,60000replicate means and uncertainty. To reproduce statistics alone: `python statistics_report.py --out results` (rewrites statistical outputs; copy results if preservation matters). Arithmetic-only audit uses Pythonstdlib and needs no weights. Live inference requires pinned dependencies and cache.

Model/tokenizer revision233902d25c440f23af6f7d6e94d2946bac0bee0a, cross-encoder/ms-marco-MiniLM-L6-v2; no prompt or decoding, Identitylogits. Titles plus abstracts, joint512tokens,16batch,4CPUthreads,seed110. Score ties resolve by descending documentID string. CE andblend share one model pass; pureRRF adds noCEcost. Four auxiliary shape-check forwardpairs outside formal5000pairs;50pair preflight separate. Budget ledger explains scope.

## Results and limits

Means RRF.6709061302/CE.6691325921/blend.7025205892. All5000fresh CE scores equal108.94shared-positive-document components (88singletons,6pairs),10000draws seed110. Blend-minusCE mean.0333879971, component percentile95% interval[.0008831395,.0663188730]; query sensitivity[.0013120488,.0680479516]. Development selection bias is not corrected; no population significance, innovation or confirmation claim. Threshold.01 remains a111practical criterion, not declared passed here. Inspect all three comparisons and harm cases, not only means.

`EXPERIMENT_LEDGER.md` inventories106–110, including negative oracle controls. `PROTOCOL.md` freezes group rule and111analysis before opening confirmation.300confirmation stays sealed until111. `prior/107_summary.json` retains old pending wording as historical output; `prior/107_confirmation.json` is actual user approval2026-10-24. No semantic judgments invented and no fresh ones required.

## Evidence / license

See SOURCE_MAP.md, upstream_verification.json, input/model/resources locks, prior_manifest.json, execution_history.json, audit_execution.json, VALIDATION.md. Apache2.0 local code and BEIR/ST attribution; SciFact annotations CC-BY4.0 and abstracts ODC-By1.0 detailed in SCIFACT-LICENSE.md. Full corpus,modelweights,venvs,credentials and privatepaths are excluded. No external shared local modules.
