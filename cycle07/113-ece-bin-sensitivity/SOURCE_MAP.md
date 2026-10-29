# Source map

Accessed2026-10-28. Local run.py calls the actual official library; it does not replace the neural network with a toy implementation.

| Component | Verified upstream | Local correspondence |
|---|---|---|
| DistilBERT classifier | Transformers commit a22a4378d97d06b7a1d9abad6e0086d30fdea199, [class at lines918–991](https://github.com/huggingface/transformers/blob/a22a4378d97d06b7a1d9abad6e0086d30fdea199/src/transformers/models/distilbert/modeling_distilbert.py#L918) | run.py infer invokes AutoModelForSequenceClassification.from_pretrained; official forward selects first-token hidden state, pre-classifier, ReLU, dropout and two-label classifier. No copied forward code. |
| Probability and ECE definitions | Guo et al., ICML2017, [Section2, Eq1–3](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf) | run.py metrics is an independent educational implementation; audit.py reconstructs from logits without importing it. |
| Author ECE reference | gpleiss/temperature_scaling commit ce1154ec7d5235eee046f28875c3dd8421f11d45, [lines89–129](https://github.com/gpleiss/temperature_scaling/blob/ce1154ec7d5235eee046f28875c3dd8421f11d45/temperature_scaling.py#L89), master SHA actually resolved by git ls-remote | Read-only comparison, not copied/executed. Author code defaults15 bins and (left,right]; ours preregisters10 and [left,right), final bin inclusive. Check whether any sample is exactly on a boundary before comparing numeric results. |
| Checkpoint and labels | [model at714eb0fa89d2f80546fda750413ed43d93601a13](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english/tree/714eb0fa89d2f80546fda750413ed43d93601a13) | Official Apache-2.0 checkpoint,0NEGATIVE/1POSITIVE, safetensors loaded without remote custom code; every model file verified against official file identities. |
| Data | [SST-2 revision8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb](https://huggingface.co/datasets/stanfordnlp/sst2/tree/8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb) | Official validation parquet loaded by PyArrow. Data-card license unknown: no raw text bundled; record ID/hash and derived predictions, reader retrieves official source. |

Official Transformers model source SHA256:7aa246bf0a4b45f901b83a9ab993e7dd2c0fb91f4fbb874045baec8ae7429a6e. Both the existing installed file and the local-storage replacement were byte-identical to the official file. Official license at the same commit was fetched and read: Apache License2.0, Hugging Face copyright header. The author ECE repository's fixed LICENSE was also fetched: MIT, copyright2017Geoff Pleiss. No upstream implementation is redistributed in this package.

NLL uses a stable log-sum-exp from real logits. Brier sums squared probability errors over both classes; audit.py uses the independent binary identity2*(p_positive-y)^2. The resulting0–2 scale must not be silently compared with the0–1 single-probability convention. Fresh model inference, independent arithmetic audit and resource/hash validation are separate checks.

## Article113 additions (2026-10-29)

Inherited self-authored modules run.py/audit.py/execute.py/verify_inputs.py and baseline data originate at [fixed112](https://github.com/distance539/ai-research-lab/tree/a57b8bcb005ae9914c2c95f2d855449b7665fd1b/cycle07/112-confidence-baseline), MIT. All needed modules are bundled; no implicit local paths.

analyze.py measure implements the same weighted absolute bin-gap definition independently, retaining112 [left,right) convention, now with5/10/15/20 bins. Official _ECELoss source and MIT license re-fetched at the pinned SHA; lines89–129 verified in raw file. No author code copied. Empty-bin values are null; boundary membership checked against interval comparisons in check_analysis.py. Bootstrap uses Python Random(seed113), saved paired row indices and linear percentiles; Wilson interval algebra is independently checked through the score-test equation. These are independent educational statistical implementations, not attributed to the upstream calibration package. plot.py reads only analysis_results.json, no generated quantitative data.

Verifier adaptation: verify_inputs.py accepts either the archived112 baseline protocol hash or current113 protocol hash. Formal inference retains112 start identity, new smoke records113; both protocol texts are bundled. The initial unadapted verifier rejected the differing protocol filename; failure retained in verification/check0_initial_failure.log, then corrected without changing predictions.
