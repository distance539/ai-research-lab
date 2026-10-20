# Frozen-head transfer: 256 controlled GPT-2 IOI inputs

Real GPT-2 small (124,439,808 parameters), official Transformers4.49.0/torch2.6.0, CPUfloat32/eager, batch1,4threads. No training/generation. This package tests a frozen three-head set, not a complete circuit. All source dependencies are included; no author's absolute paths are required.

## Run

Python3.12.14 used. Approximately550MB model download and4GiBRAM budget; measured formal62.11s and1,466,384,384B peakRSS on the recorded Mac. Other hardware speed/bitwise equality unverified.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r environment.lock.txt
python run.py --cache ./cache --out smoke-new --smoke
python audit.py smoke-new
python run.py --cache ./cache --out results-new
python audit.py results-new
python test_checks.py
python replay.py --cache ./cache --results results --out replay-check.json
```

Minimal entry after dependencies: `python run.py --cache ./cache --out smoke-new --smoke`. It fetches missing immutable model/tokenizer files and checks8SHA256s; existing verified caches are reusable. If public download fails, retry prepare.py --cache ./cache; never disable TLS validation. Smoke uses only old-old row0: four scored conditions plus five control forwards. Complete run256rows,1024scored conditions plus1280control forwards. Output directory must not already exist. records.json contains per-row candidate logits/probability/full-vocabulary argmax, IDs, clean activations, controls and shapes; summary/runtime/environment/identity/stages JSON record aggregation/resources/versions. Terminal progress and exit status appear in saved stdout.log/execution.json for the author's run. No nonexistent full-benchmark score is claimed.

Offline: `python audit.py results` independently recomputes every scalar difference, means, counts, contrasts, IDs, corruption masks and ratios. Seven deliberate corruptions must fail in test_checks.py. `replay.py` independently implements all1024conditions as postprojection deltas and checks full-vocabulary argmax and probability; it never imports run.py or analyze.py. This is machine arithmetic/model checking, not human review. `python rebuild_samples.py --out samples-rebuilt.json` reconstructs all256prompts from SHA-verified official literals; --source can reuse an already downloaded file. No upstream Python is executed by reconstruction.

## Frozen design

A=L8H10,B=L8H6,C=L9H9 from103's previously selected102development ranking. All3jointly intervened at END. Four64row cells: old_old,new_old,old_new,new_new (name group first). Official names first8 versus next8; templates0,1 versus2,3; cyclic pairs/two swaps/two name orders; store/ring fixed. New names/templates were reserved by100 and first inferred after plan.json freeze. Old-old is development reference, not independent testing. All192transfer prompts are now consumed. New names are not claimed absent from pretraining.

Primary joint zero-ablation drop:2.3871,1.7200,1.9474,1.3604. CleanD:4.4460,2.9471,3.8775,2.4795; clean top1:64/64,64/64,56/64,53/64. Secondary s2_flip restoration gains5.3874,3.8338,4.2066,2.9967; normalized recovery60.59%,65.04%,54.24%,60.43%. All256raw drops/gains positive, but effect magnitude and answer recovery differ. Ratios are ratios of means, not means of per-row ratios. No clipping or exclusion of clean failures; no IID interval or head reselection. Templates change length and wording together, not a pure syntactic manipulation. See PROTOCOL.md for frozen hypothesis and budgets.

## Integrity

plan.json frozen before smoke/formal covers scientific scripts/config/protocol/data/resource identity. Supplementary replay/reporting files were added afterward, do not select heads or alter original predictions. MANIFEST.json covers all distributed files except itself. Source map links immutable official code and licenses. Original modules MIT, upstream IOI MIT, Transformers Apache2.0, GPT-2 MIT. No weights,cache,venv,secrets,private configs or restricted natural data in distribution. Synthetic attributed template outputs and derived activations are included. Local ZIP and public directory carry identical distributed bytes.
