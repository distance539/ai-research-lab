# 101 GPT-2 attention head activation patching

Actual16controlled synthetic clean/corrupt pairs, official GPT-2 small124,439,808parameters. Scan12layers x12heads x each15/19token position:39,168isolated interventions. Changed token is second subject occurrence; score retains original clean IO and subject. This is a restricted development experiment, not a natural-data benchmark or complete circuit claim.

## Minimal offline evidence audit

```sh
python audit.py
python test_checks.py
```

Expected: all16rows/39,168grid cells and144END rankings/ratios pass; eight evidence corruptions rejected. No torch needed for this deterministic recomputation. It is not human review.

## Real end-to-end smoke

Python3.12.14, torch2.6.0/Transformers4.49.0; exact25runtime requirements in environment.lock.txt. Disk~551MB model assets plus environment. Preset budget4GiB RSS/1800s; actual formal190.646s wall/1,600,782,336B peak on macOSarm64 CPU4threads. Model/tokenizer assets may be reused only if all8hashes match.

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r environment.lock.txt
TOKENIZERS_PARALLELISM=false python run.py --cache ./cache --smoke --out smoke-new
python audit.py --results smoke-new
```

run.py downloads pinned public assets when missing, verifies their hashes, checks frozen scientific code/config/protocol identities, and runs one complete2,160node scan. Expected outputs: records.json, patches.json, summary.json, runtime.json, environment.json, stages.json, identity.json. Smoke originally15.239s wall; crosshardware numeric identity not promised. Cache keys cover model/tokenizer resources plus source/config/prompt/data/score/no-generation settings; output caches are not reused. No author-machine paths or hidden modules required.

For full16pair run omit --smoke; choose new output directory. Read config.json/PROTOCOL.md first, then run.py cache/patch/controls, analyze.py, audit.py and SOURCE_MAP.md. Layer/head indices are zero-based. Full-state positive control replaces the entire post-embedding residual; it does not imply that patching all attention heads alone reconstructs clean behavior. Negative selfpatch uses every layer's corrupt z cache. Independent projected-delta control at presetL9H9END verifies the preprojection hook against postprojection algebra.

## Results and limits

Mean clean/corrupt difference4.295332/-4.295332; denominator8.590665. TopEND L8H10 raw gain2.454803, recovery0.285752, resulting mean still-1.840530. L10H7END has gain-3.840752. All cells saved, including negative effects; do not sum individual recovery ratios. No joint patch, training, generation, new-name transfer or confirmation inference.16rows share3names/two templates; no IID confidence interval.

plan.json was frozen before smoke/formal. results/ and smoke/ include raw logs, exit codes, run identities, scores and independent audits. Batched baseline and padded batches32 keep scan arithmetic shape constant. Clean cache is batch1; observed maxbatch discrepancy0.0002213, official full-readout discrepancy0.0001907, independent delta discrepancy0.0000458, causal pre-change residual gain0.0000611. Full-state/selfpatch are exact. Read VALIDATION.md for implementation boundaries.

rebuild_samples.py retrieves the fixed public IOI file, SHAchecks and AST-parses literals to reproduce samples.json; optional --source allows an already verified download. No full Easy-Transformer or TransformerLens runtime required. Parent baseline samples/outputs are copied locally from the fixed100 version; see SOURCE_MAP.md. Held-out names/templates remain unused.
