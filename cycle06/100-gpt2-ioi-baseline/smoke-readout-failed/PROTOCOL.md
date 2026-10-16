# Frozen protocol: 100 IOI behavior baseline

Freeze before model inference: 2026-10-16T00:58:46.924274+00:00

Target: official public GPT-2 small weights via Transformers GPT2LMHeadModel, CPU float32 eager, 4 threads, eval/inference_mode, seed100. No training, generation, BOS, padding, truncation, few-shot or activation intervention. Synthetic public IOI templates from Wang et al. author repository, not natural-distribution data or complete paper replication.

Two original BABA templates (indices0,1) and their ABBA variants; first8 official names; eight cyclic neighbor pairs, each reversed; store/ring fixed. Total64 prompts. Equal16 rows per template/order; names balanced in both roles. These are dependent controlled design cells, not64 independent natural examples. No significance CI from row resampling. Primary mean logit(IO)-logit(S), secondary positive difference fraction, vocabulary top1 correctness, and absolute probabilities. Report every row and all four cells, including failures. Ties fail strict preference.

Smoke first2 rows solely for correctness/resources. Final all64 rows only after estimated total runtime<=1200s and measured RSS<=4GiB. No quality threshold to tune/reselect templates. Max128tokens; refuse truncation. Failure logs retained. Formal source/config/sample hashes frozen in plan.json before inference. Replays compare numerical results, not timing.

Reserved: next8 official names and template indices2,3; no inference on these until a frozen later transfer protocol. Existing SST2 confirmation was consumed by099, unrelated to this new unit. 101 will define corrupt pairs before intervention;100 does not select heads.

Independent checks: full prefix+answer tokenization must equal prefix IDs plus exactly one answer ID; IO occurs once, S twice, all zero-based positions recorded. Last input token is prediction position. Recompute difference, sigmoid pair probability, per-cell/overall aggregation using standard library. Crosscheck lm_head(hidden[-1]) against official logits; no assertion of equivalence to transformed TransformerLens weights.
