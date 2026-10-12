# 097 — Answer extraction with owner-confirmed review

Frozen SmolLM2-135M-Instruct generates64 existing SST-2 development responses. Three frozen scoring rules inspect identical outputs: canonical JSON exact match, strict one-key JSON parser, official harness first-label regex. **On 2026-10-12 the user confirmed that review passed, waived the per-row form, and explicitly authorized publication. No per-row labels were supplied, so no quantitative human-gold error rate is reported.** See human_review_confirmation.json and ACCEPTANCE_POLICY_2026-10-12.md. Original pending forms and execution records remain historical evidence.

## Install and minimal real run

Python3.12; CPU; measured peak RSS1.28GB,4GiB budget. Allow space for the Python environment and approximately273MB fixed resources. Reuse an already hash-verified cache if available.

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt -c environment.lock.txt
python prepare.py --cache ./cache
python run.py --cache ./cache --out ./replay-smoke --smoke
python audit.py ./replay-smoke
```

Omit `--smoke` for the64-row run. Existing output directories are refused. All12 fixed resource files are checked before inference; no implicit author-machine path or parent-project import exists. Model/tokenizer revision12fd25f77366fa6b3b4b768ec3050bf629380bac; GLUE revisionbcdcba79d07bc864c1c254ccfcedcce55bcc9a8c; harness commitad3f4d0cad1cfcdb815f1e795f7947e49ed9f2e9. Exact89-version environment captured in environment.lock.txt; requirements pin core dependencies and official commit. No training, inference cache, calibration, few-shot or confirmation inference. KV caching within generation is enabled.

## Actual execution and results

2026-10-10:2-row smoke then64-row formal generation, CPUfloat32/eager/4threads/greedy32/seed97. Original088 JSON system/task/contract/chat prompt reproduced; the new research variable is scoring. The previous096 used candidate likelihoods; scores across those protocols are not equivalent.

Formal10.369964s after imports,14.690279s external wall,9.914543s generation/probe,1280081920B peak RSS. Smoke0.720739s after imports,1279655936B RSS; conservative projected23.399700s<1200. Input66–108 tokens,5250total;422generated tokens includingEOS;1length stop(ID553),63EOS. First logits[1,83,49152]. Canonical exact0/64; strictJSONformat0/64 and correctness0/64; regex50/64. Paired gains50/losses0 reflect a changed measurement, not50 proven scoring errors or improved model ability. All64 extracted labels match official RegexFilter/TakeFirstFilter. Official filters are called directly, generation uses Transformers; this is not a standard harness SST2 benchmark run.

Records contain full responses/hashes, shapes, stop causes, scores and official extraction. No output in this run has an8-word source overlap; all64 can be reparsed publicly. The export script applies the distribution policy to future runs. Original source reviews, full prompts, input/generated token IDs, model weights and cache are not distributed. Reference dataset labels are existing annotations, not this project's human-output gold.

## Human review

The following form workflow remains available for readers who want reproducible per-row semantic metrics; the owner waived it for this article acceptance. The owner confirmation is not a completed label dataset.

The16 IDs were fixed at09:01:16+08 before smoke/formal generation.8 per gold class;31negative/33positive in development. Included review_sample.json is immutable; do not rerun selection into that filename. To inspect reproducibility, run select_review.py with a NEW output filename and compare identities/items, ignoring creation time.

```bash
python review.py build --cache ./cache --results ./your-full-run --out ./human
# Open human/review.html, personally complete16 rows, export JSON.
python review.py validate --file ./human_review_completed_097.json --results ./your-full-run
```

HTML hides scorer predictions, starts every human field empty, and exports reviewer/time/attestation plus response hashes. Raw packet content remains local. Validator checks all16 decision/content/format/reason/reference-dispute fields, identities, consistency, explicit missingness and class weights. A valid-looking JSON alone does not prove a human reviewed it: actual owner confirmation is also required. Blank forms fail. Human metrics remain null because individual judgments were not supplied, despite the owner confirmation. See HUMAN_REVIEW.md and PROTOCOL.md. Do not mistake parser tests for annotation.

## Independent verification

```bash
python audit.py results
python test_checks.py
python -m compileall -q .
```

Independent audit does not import production scorer or runner; verifies64 records, exact/JSON/regex rules, sums, identities and paired IDs.13 tests:6 software fixtures,6 corruption rejection tests,1blank human-review rejection. All are software checks, not human labels. SOURCE_MAP.md gives fixed official files/functions/lines, licensing, local correspondence. VALIDATION.md records archive/replay validation and the dated acceptance amendment.
