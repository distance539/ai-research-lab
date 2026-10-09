# 096 — Option order and candidate scoring

A real CPU experiment with frozen SmolLM2-135M-Instruct, 64 pre-existing SST-2 development questions and five prespecified conditions. No training or generation. This is a controlled custom protocol, not the standard full GLUE benchmark. Inherited095 assets are fixed at commit2522a45e1d2efbb2ff1692be4fc734627ea00b55; all needed self-written modules are included here.

## Setup and minimal real run

Python3.12, CPU; measured peak RSS1.57GB; budget4GiB/1200s. Allow additional disk space for the Python environment and approximately273MB locked resources. Network is needed only for initial dependencies/resources; verified caches may be reused.

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt -c environment.lock.txt
python prepare.py --cache ./cache
python run.py --cache ./cache --out ./replay-smoke --smoke
python audit.py ./replay-smoke
```

The constraints file pins the exact transitive versions used. requirements.txt pins core versions and the actual official commit; environment.lock.txt records the entire measured environment. Do not replace the pinned commit with main.

Full run: omit --smoke, choose a new output directory. The program refuses an existing output directory and refuses truncation. It checks all12 locked resource hashes before inference. Model/tokenizer revision12fd25f77366fa6b3b4b768ec3050bf629380bac; GLUE revisionbcdcba79d07bc864c1c254ccfcedcce55bcc9a8c; harness commitad3f4d0cad1cfcdb815f1e795f7947e49ed9f2e9. No author-machine paths required.

## Outputs and actual scope

records.json: all320 item/condition rows with640 candidate likelihoods, continuation token IDs/logprobs, semantic mapping, gold position, context hashes, tensor shapes and official agreement. summary.json: every condition and paired contrast. environment.json: measured CPU/runtime/RSS. identity.json: resource/source/config/protocol identities, cache key, selected IDs; inference cache disabled. stages.json: resource, preprocessing, inference and evaluation status. run.log/execution.json retain actual successful command evidence with machine paths replaced by explicit cache placeholder. Local private logs retain exact invocation.

Actual2026-10-09 full run:43.556648s after imports,47.121229s external wall time,1568227328B peak RSS. Two-question/five-condition smoke:4.773033s after imports,1369161728B peak RSS,280.124071s conservative full estimate. CPUfloat32/eager/4threads, seed96, max512, batch1. Every scored continuation was one token. No BOS/chat/EOS/calibration. No128SST-2 or64news reserved confirmation inference.

Correct counts (sum/token mean/character mean are identical): official_text32; listed_text_forward31; listed_text_reverse32; listed_letter_forward32; listed_letter_reverse33, each out of64. Text-order flip1 (gain1/loss0); letter-order flips63 (gain32/loss31). The latter retains A/first-position preference but changes the semantic binding. This design cannot separate symbol identity from physical position. Constant-positive predicts33/64; no ability improvement claim. Listed text/letter share identical contexts and all four listed arms have5530 candidate-input tokens; official context baseline4506.

Official HFLM.loglikelihood plus ConfigurableTask.process_results are used directly, independently checked by shifted direct forwards. This is not the simple_evaluate entry point. Official acc_norm divides by choice characters; token mean is a separate local diagnostic. Both text choices have8characters/1token, letters1character/1token: normalization invariance is algebraic for this setting only.

## Independent verification

```bash
python audit.py results
python test_audit.py
python -m compileall -q .
```

All640 scores agree exactly with the official path. audit.py does not import run.py; it recomputes score sums, normalization, first argmax, mapping, correctness and all pairs, and records input/script SHA256. Six mutation tests verify rejection of corrupt scores, mapping, token counts, predictions, missing and duplicated records. This is automated arithmetic, not human semantic review.

See PROTOCOL.md for estimands and frozen rules; DEVIATIONS.md for two failed target-field implementations and environment recovery; SOURCE_MAP.md for permanent official source links, distinctions and licenses. Failed folders retain identity/stage/failure evidence, never count as successful runs. All raw source reviews and full prompt/input tokens are excluded from distribution; use pinned resource downloads for reconstruction. See THIRD_PARTY.md. No model weights, cache, environment or credentials are bundled.
