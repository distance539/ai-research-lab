# 099 SOURCE_MAP

Access/verification:2026-10-14. `upstream_verification.json` records actual anonymous HTTP200, SHA256 and byte equality to installed source for every official file below. Code is an independently written teaching experiment, not a reimplementation of the Cawley/Talbot kernel experiments or a reusable-holdout/privacy algorithm.

## Official harness, MIT

Repository https://github.com/EleutherAI/lm-evaluation-harness ; actual pinned commit `ad3f4d0cad1cfcdb815f1e795f7947e49ed9f2e9` (0.4.9.2). LICENSE-HARNESS.txt retained.

| Module/formula | Verified permanent source | Local use |
|---|---|---|
| SST2 text/choice/target | [default.yaml L1](https://github.com/EleutherAI/lm-evaluation-harness/blob/ad3f4d0cad1cfcdb815f1e795f7947e49ed9f2e9/lm_eval/tasks/glue/sst2/default.yaml#L1) | official_sst2.yaml is an unmodified, byte-equal copy. spec.py zero matches doc_to_text; variants explicitly change context/choices. |
| Requests and IDs | [Instance L11](https://github.com/EleutherAI/lm-evaluation-harness/blob/ad3f4d0cad1cfcdb815f1e795f7947e49ed9f2e9/lm_eval/api/instance.py#L11) | run.py constructs explicit idx/position, avoiding log-order mapping errors. |
| Token boundaries | [_encode_pair L358](https://github.com/EleutherAI/lm-evaluation-harness/blob/ad3f4d0cad1cfcdb815f1e795f7947e49ed9f2e9/lm_eval/api/model.py#L358) | score_direct independently tokenizes concatenation and asserts prefix alignment. |
| Official score dispatch | [loglikelihood L407](https://github.com/EleutherAI/lm-evaluation-harness/blob/ad3f4d0cad1cfcdb815f1e795f7947e49ed9f2e9/lm_eval/api/model.py#L407) | inherited HFLM.loglikelihood called directly; not simple_evaluate. |
| Conditional log-likelihood sum | [_loglikelihood_tokens L1131](https://github.com/EleutherAI/lm-evaluation-harness/blob/ad3f4d0cad1cfcdb815f1e795f7947e49ed9f2e9/lm_eval/models/huggingface.py#L1131) | official outputs checked against independent causal shift/log_softmax/gather/sum, tolerance2e-5. |
| Argmax and gold | [process_results L1546; argmax L1601](https://github.com/EleutherAI/lm-evaluation-harness/blob/ad3f4d0cad1cfcdb815f1e795f7947e49ed9f2e9/lm_eval/api/task.py#L1601) | run.py uses a callable semantic-to-position gold mapping; audit.py independently sums and maps scores. |

## Real Transformer, Apache2.0

Transformers4.49.0, actual commit `a22a4378d97d06b7a1d9abad6e0086d30fdea199`: [LlamaForCausalLM.forward L787](https://github.com/huggingface/transformers/blob/a22a4378d97d06b7a1d9abad6e0086d30fdea199/src/transformers/models/llama/modeling_llama.py#L787). Used as installed library; no model implementation copied. PyTorch2.6.0 license also retained. Model/tokenizer fixed revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`, Apache2.0, exact files and hashes in resources.lock.json. No weights redistributed.

## Paired bootstrap, BSD3

SciPy1.13.1, actual commit `44e4ebaac992fde33f04638b99629d23973cb9b2`: [bootstrap L279](https://github.com/scipy/scipy/blob/44e4ebaac992fde33f04638b99629d23973cb9b2/scipy/stats/_resampling.py#L279), [_bootstrap_resample L62](https://github.com/scipy/scipy/blob/44e4ebaac992fde33f04638b99629d23973cb9b2/scipy/stats/_resampling.py#L62). analyze.py invokes official paired percentile20000/Generator9900; shared-index computation independently verifies every sample. audit.py independently derives the exact empirical {-1,0,1} bootstrap distribution by standard-library convolution. Neither estimates the full search's uncertainty. LICENSE-SCIPY.txt retained; implementation not copied.

## Local scientific protocol

spec.py reuses original author definitions from [095 fixed source](https://github.com/distance539/ai-research-lab/tree/2522a45e1d2efbb2ff1692be4fc734627ea00b55/cycle06/095-fewshot-selection-order) and [096 fixed source](https://github.com/distance539/ai-research-lab/tree/c3ab8060b4d9f3011a02c94333748f8e04c0b091/cycle06/096-option-order-scoring), adding short_text. `plan.json` is095's original frozen train-demo IDs and088 split; historical `confirmation_inference:false` describes creation time. Current confirmation consumption is recorded separately. Prepare/download and resource locks reuse096 with full local source included. No author-machine parent imports.

select.py's maximum over development counts and stable tie order is an independent teaching implementation motivated by [Cawley & Talbot2010](https://www.jmlr.org/papers/v11/cawley10a.html), not copied code. Confirmation evaluation is one frozen winner versus fixed zero baseline, not a search on test. No raw source reviews/full prompt strings, private data, caches or keys are distributed. GLUE SST2 license unknown; use hash-pinned download steps. No human gold metrics from097 are imported.
