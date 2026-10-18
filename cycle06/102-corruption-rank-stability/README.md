# 102 — Counterfactual corruption and head-rank stability

Real GPT-2 small, 16 controlled IOI development samples, two name counterfactuals, 78,336 single-node interventions. This is not natural data, a full benchmark, a full circuit reconstruction, or a Gaussian-noise study. See PROTOCOL.md for frozen choices; SOURCE_MAP.md for official fixed commits and attribution. All own helper modules are included.

## Reproduce

Use Python3.12 (tested3.12.14), CPU and at least4GiB available RAM. Model weights are548MB; allow approximately1GB disk for assets and results. Runtime depends on hardware; see results/runtime.json for this measured run. Do not compare this smoke subset with a full-paper benchmark.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r environment.lock.txt
python run.py --cache ./cache --out replay-smoke --smoke
python audit.py --results replay-smoke
python run.py --cache ./cache --out replay-full
python audit.py --results replay-full
python test_checks.py
```

The output directory must not exist. All scripts accept --help except the fixed corruption-test script. No API key or authentication. prepare.py fetches eight public model/tokenizer/card files from immutable revision URLs and checks SHA256 on EVERY run, including cached files. The actual author run reused an already verified cache; the exact local command is archived outside the distributable package, while execution.json records the portable equivalent. requirements.txt lists direct dependencies; environment.lock.txt pins all25 packages used in the reused environment. GPU/MPS and other versions are untested. No training or generation.

`python rebuild_samples.py --out reconstructed.json` fetches the fixed public IOI Python source, verifies its hash and parses literals without executing it; result must exactly equal samples.json. It also accepts --source for a previously verified file. No hidden absolute paths or imports from prior article directories.

## Outputs and auditing

results/ contains32 condition-records, all78,336 patch records, summary/rankings, runtime/environment/stages/cache identity, execution command/exit code and raw stdout. smoke/ contains one sample x2conditions x4320 patches. Both have independent audit.json. parent_summary.json is101's fixed published ranking, used to check the rerun baseline. parent_baseline.json and parent_samples.json preserve100 provenance; only samples.json selected16 are inferred. Heldout names/templates are not inferred.

Key=SHA256(plan hash + resources lock hash + smoke/full mode); plan binds model/tokenizer/data via resource/config hashes, exact prompts via samples hash, code/scoring/corruption and no-generation policy. Output reuse is disabled. The plan hash is an execution identity, not a claim that every document was written before results; README/VALIDATION are post-run documentation.

Scoring always uses ORIGINAL IO minus ORIGINAL S, including when corruption makes the opposite recipient correct. s2_flip changes one token; global_swap changes all three name tokens simultaneously. Both full-design prompt multisets equal clean, controlling average baseline difficulty, not per-pair information paths. All positions are saved; END was prespecified. Raw gain is primary; ratio of means secondary. Negative values retained. Spearman uses deterministic tie ordering, with near ties reported. No IID confidence interval or universal component importance claimed.

`python analyze.py --results results` recomputes summaries without a model. `audit.py` independently checks original candidates, complete unique grid, controls, arithmetic, all END rankings/ratios, corruption mappings and multiset balance, paired focal gains and ordinal correlation. Eleven mutation tests must be rejected. Official TransformerLens specification is ported; TransformerLens runtime itself is NOT installed/run. Third-party licenses and implementation distinctions are in THIRD_PARTY.md and SOURCE_MAP.md.
