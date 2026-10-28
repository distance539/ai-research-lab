# Frozen protocol: confidence baseline

Registered 2026-10-28 before any inference for this experiment. The question is whether maximum softmax confidence tracks empirical correctness for a fixed public sentiment classifier, not whether a language model can recognize hallucinations.

Use config.json without tuning. Sort all 872 SST-2 validation IDs by SHA256('cycle7:'+str(idx)); use first256 as exploration, next256 as calibration and last360 as protocol-audit pool. Save split_manifest.json before inference. Only exploration receives predictions in112. This dataset has appeared in the previous series; no zero-contamination or genuinely unseen-test claim is made. The selected checkpoint was already fine-tuned on SST-2 and its authors report validation performance.

Run8 samples as a resource smoke test, then the fixed256 with identical settings. The smoke samples are part of development, not a confirmation set. Use all256 regardless of outcome. No training, temperature fitting, sample selection by results, threshold search, or model comparison.

Primary descriptive endpoints: accuracy, mean maximum probability and their signed difference. Also report ten equal-width ECE bins on[0,1], NLL and Brier with sum over both classes (range0–2). Bins are [left,right), last bin includes1. Empty bins have null accuracy. Threshold0.9 is fixed before inference; report count and correctness, not only a percentage. No confidence interval or population guarantee is asserted in this first descriptive experiment.

Inputs use official labels0negative/1positive and unmodified text. Dynamic batch padding, truncation512, eager attention, CPUfloat32, eval mode, torch inference_mode. Model output logits[B,2] use float64 softmax for analysis. Save per-ID logits, probabilities, label, token counts and text SHA256; do not redistribute sentences because dataset-card licensing is unresolved. Frozen source and input hashes are recorded.

Reject results if input identities duplicate, official hashes differ, logits are nonfinite, labels mismatch, deterministic independent metric audit fails, or budget is exceeded. A smoke budget problem requires a dated protocol amendment before formal inference, never a silent change. Capture all commands, outputs and exit codes. Quantitative plots use saved predictions only. Cover and method diagram are separate conceptual images, not result figures.

Future held-out IMDb confirmation protocol is in the series outline. No IMDb inference or other-series confirmation access is part of112.
