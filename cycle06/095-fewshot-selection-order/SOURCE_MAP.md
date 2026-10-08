# Implementation map

Verified2026-10-08. lm-evaluation-harness commit ad3f4d0cad1cfcdb815f1e795f7947e49ed9f2e9, version0.4.9.2, MIT. Fixed checkout and installed files compared byte-for-byte; hashes in upstream_verification.json. No latest-version claim.

| Module/formula | Permanent official source | Local use and differences |
|---|---|---|
| prompt/choices/gold | [SST2 YAML L1](https://github.com/EleutherAI/lm-evaluation-harness/blob/ad3f4d0cad1cfcdb815f1e795f7947e49ed9f2e9/lm_eval/tasks/glue/sst2/default.yaml#L1) | official_sst2.yaml verbatim MIT copy |
| selection | [ContextSampler L17](https://github.com/EleutherAI/lm-evaluation-harness/blob/ad3f4d0cad1cfcdb815f1e795f7947e49ed9f2e9/lm_eval/api/samplers.py#L17) | FixedSampler overrides only sample(n), frozen train indices instead of per-question random draws |
| demo concatenation | [get_context L78](https://github.com/EleutherAI/lm-evaluation-harness/blob/ad3f4d0cad1cfcdb815f1e795f7947e49ed9f2e9/lm_eval/api/samplers.py#L78) | inherited formatting, independent block()/question() character comparison |
| task composition | [fewshot_context L1095](https://github.com/EleutherAI/lm-evaluation-harness/blob/ad3f4d0cad1cfcdb815f1e795f7947e49ed9f2e9/lm_eval/api/task.py#L1095) | explicit train split, fixed sampler metadata, no chat/BOS, zero or4examples |
| data transport | [download L982](https://github.com/EleutherAI/lm-evaluation-harness/blob/ad3f4d0cad1cfcdb815f1e795f7947e49ed9f2e9/lm_eval/api/task.py#L982) | supported custom_dataset, full fixed train/validation, no monkeypatch |
| official entry | [simple_evaluate L50](https://github.com/EleutherAI/lm-evaluation-harness/blob/ad3f4d0cad1cfcdb815f1e795f7947e49ed9f2e9/lm_eval/evaluator.py#L50) | sorted sample positions preserve094 identity fix, no bootstrap/cache |
| token boundary | [_encode_pair L358](https://github.com/EleutherAI/lm-evaluation-harness/blob/ad3f4d0cad1cfcdb815f1e795f7947e49ed9f2e9/lm_eval/api/model.py#L358) | independent full-string tokenization/prefix assertion, refuse truncation |
| likelihood sum | [_loglikelihood_tokens L1131](https://github.com/EleutherAI/lm-evaluation-harness/blob/ad3f4d0cad1cfcdb815f1e795f7947e49ed9f2e9/lm_eval/models/huggingface.py#L1131) | independent_scores direct forward/context shift/sum,640candidate comparisons |
| correctness/mean | [process_results L1546](https://github.com/EleutherAI/lm-evaluation-harness/blob/ad3f4d0cad1cfcdb815f1e795f7947e49ed9f2e9/lm_eval/api/task.py#L1546), [mean L35](https://github.com/EleutherAI/lm-evaluation-harness/blob/ad3f4d0cad1cfcdb815f1e795f7947e49ed9f2e9/lm_eval/api/metrics.py#L35) | first-max, mean correct; independently audit gains/losses/flips with64paired units |

run.py/audit.py/test_audit.py are independent teaching orchestration/audits, not the cited papers' optimization implementations. prepare.py/resource locks/dependency/license materials reused in full from [094 public package](https://github.com/distance539/ai-research-lab/tree/22bce0ae26f5c320fa37ad74b394f2dc3114fa33/cycle06/094-harness-task-protocol); no hidden modules.

Model/tokenizer: [fixed HuggingFaceTB card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/blob/12fd25f77366fa6b3b4b768ec3050bf629380bac/README.md), Apache2.0. Transformers4.49.0 source commit a22a4378d97d06b7a1d9abad6e0086d30fdea199, Apache2.0. Data: [fixed GLUE card](https://huggingface.co/datasets/nyu-mll/glue/blob/bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c/README.md), license unknown, originals not distributed. Inherited lock retains prior Stanford snapshot for provenance; inference reads GLUE only.
