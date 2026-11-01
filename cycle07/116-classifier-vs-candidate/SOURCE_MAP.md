# Source map — accessed2026-11-01

|Component|Verified fixed upstream|Local correspondence|
|---|---|---|
|Causal LM forward|Transformers a22a4378d97d06b7a1d9abad6e0086d30fdea199, Apache2.0, [LlamaForCausalLM L752](https://github.com/huggingface/transformers/blob/a22a4378d97d06b7a1d9abad6e0086d30fdea199/src/transformers/models/llama/modeling_llama.py#L752), [vocab projection L859](https://github.com/huggingface/transformers/blob/a22a4378d97d06b7a1d9abad6e0086d30fdea199/src/transformers/models/llama/modeling_llama.py#L859)|run.py calls actual library. Official file and installed file hash identical, upstream_verification.json. No forward code copied.|
|Classifier baseline|Same Transformers SHA, [DistilBERT classifier L918](https://github.com/huggingface/transformers/blob/a22a4378d97d06b7a1d9abad6e0086d30fdea199/src/transformers/models/distilbert/modeling_distilbert.py#L918), Apache2.0|formal_predictions.json inherited byte-identical from112, source fixed commit in README; no new classifier inference claimed.|
|Candidate scoring and normalization|Same causal LM source above supplies vocabulary logits|run.py independently implements shifted token log-softmax sum and two-candidate normalization. No claim of harness execution. check_forward.py uses prompt-only last-position softmax, audit.py uses scalar logistic identity.|
|Calibration meaning|[Guo et al. ICML2017](https://proceedings.mlr.press/v70/guo17a.html)|Background definition only; no original benchmark reproduced.|
|Checkpoint|[SmolLM2 revision12fd25f77366fa6b3b4b768ec3050bf629380bac](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/tree/12fd25f77366fa6b3b4b768ec3050bf629380bac), Apache2.0|model.lock.json verified against official API blob IDs/LFS SHA256. prepare.py downloads exact files.|
|Dataset|[Stanford SST2 revision8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb](https://huggingface.co/datasets/stanfordnlp/sst2/tree/8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb), license unknown|resources.lock.json exact validation hash; no raw text distributed.|

Our orchestration is independent educational MIT code, not copied upstream implementation. No upstream repository bundled. Inherited author-generated112 data have attribution above. Model card and license remain accessible at fixed revision. Public links point to exact code versions; no invented lines or SHAs.
