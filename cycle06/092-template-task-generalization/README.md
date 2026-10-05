# 092 — 模板换一下还会吗：96条开发样本的探索实验

**Technical reproduction passed; human user confirmed label-preservation review passed on2026-10-05.** No human gold, unseen-task or general improvement claim.

Same32SST2 training rows and64development rows as090/091, plus32balanced AG News development rows. Two fixed model states, six templates,640generations.192confirmation rows never inferred. Frozen091selected target_only checkpoint. News is absent from its incremental training but already used for091selection; not blind new-task evaluation.

## Minimum reproducible entry

Python3.12.14/macOS26.6arm64,torch2.6.0/transformers4.49.0; exact direct dependencies and full environment recorded. CPU, at least6GiB available memory, about3GBdisk plus Python environment. Model269MB, regenerated state538MB; data resources in lock. No parent blog directory dependency.

```sh
python3.12 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python prepare.py --cache ./cache
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false python run.py --cache ./cache --out smoke-new --private ./private-new --smoke
python audit.py --cache ./cache --out smoke-new
```

Smoke regenerates all64training steps to preserve checkpoint identity, evaluates2sentiment+4news rows across6templates before/after (32generations). On tested platform27.4139s/2,847,473,664B peakRSS;64losses/gradient norms equal091 and numeric stateSHA25610af612eb88611e2a211b6315efb5ea6f97030346b58fb3dc55d20dc181b28df matches saved091checkpoint. Serialization hashes differ with filename despite identical tensors. For full640outputs remove`--smoke`, use new output/private dirs. Optional`--checkpoint PATH` accepts ONLY091original fileSHA256fc7e2dd5554793b719019872d47df076e67df677bcd3f5723bc4fd5e7d626efb; no implicit local lookup.

prepare.py obtains20fixed-revision resources and verifies bytes. Existing correct cache reused. run.py stores public evidence in out and full raw predictions/blinded review packet/checkpoint in private. It refuses an existing output directory. Model/weights/source data are not bundled. For public released results run`python audit.py --cache ./cache --out results`. For complete raw reparse add`--raw ./private-new/results-new-raw-predictions.jsonl`using the actual run output name.

## Fixed conditions and results

Model/tokenizer SmolLM2-135M-Instruct12fd25f77366fa6b3b4b768ec3050bf629380bac,134515008parameters. CPUfloat32/eager/4threads,eval seed92,greedy32,no truncation/max512. Training regeneration exactly32rows×4epochs,64SGDsteps,seed91,lr.001,batch2,assistant8tokens/example,clip1,1024supervisedtokens; no news training. templates.py holds all six complete templates, original chat wrapper/label order unchanged. PROTOCOL.md predates inference; primary comparison originalJSON/paraphrase, order change secondary, plain/news diagnostics. No prompt selection by092results.

| Template | n | Before F/C/J | After F/C/J |
|---|---:|---|---|
| json_original |64|0/50/0|64/58/58|
| json_paraphrase |64|0/54/0|64/58/58|
| json_reordered |64|0/53/0|64/55/55|
| plain_original |64|64/57/57|64/33/33|
| news_original |32|30/8/8|29/8/8|
| news_paraphrase |32|31/8/8|29/7/7|

F=strict format,C=unique explicit label matches dataset label,J=both; C is automatic proxy, not semantic human truth. After original→paraphrase J1win/62ties/1loss; reordered1/59/4. Same aggregate does not mean same predictions. News baseline near constant8/32floor. No population CI, task-family replication or full benchmark claim.

Actual formal run reused hash-verified091checkpoint:92.6810s after imports/1,712,799,744B peakRSS,640generations, no updates. GPU memory unmeasured. Prompt lengths differ (paraphrase+2tokens/review; reordered aggregate same); decoder upper cap same but realized outputs differ. Neither equalFLOPs nor pure punctuation effect. results/identity.json includes precise data/model/tokenizer, templates,prompts,decoding,checkpoint state,split and executed source hashes. Key shape[1,83]→[1,83,576]→[1,83,49152], firstargmax504 checked.

## Audits and human review

Independent audit.py reconstructs every prompt/gold from pinned raw data, parses F/C/J independently, recomputes pairs/aggregates, checks source/cache identity and no192reserved rows. Full local640row raw audit passed. Public632full outputs+8redactions (>=8consecutive source words) due unknown data license;8rows keep hash/derived counts but cannot be independently reparsed from public file alone. Regenerate for full raw audit. No raw source/input tokenIDs distributed. Review packet generated before inference with320prompt/sample combinations and no model outputs. The user subsequently confirmed the human review passed. human_review_confirmation.json records the direct statement and packet hashes. Original empty row forms and execution-time pending logs remain historical evidence; no individual annotations or exact review-completion time were invented. Guide HUMAN_REVIEW.md. This retrospective review does not create a blind final test.

SOURCE_MAP.md pins5official files with licenses and local mappings. All needed self-written helpers copied from091fixed public commit7bd49bd7d6c1cd950d58506cfe7d4e962d161c85; newrun/templates/audit independent. Only execution identity files existed during inference; audit/docs added afterward, no inference source modified. Archive/remote validations maintained outside source package to avoid circular hashes.
