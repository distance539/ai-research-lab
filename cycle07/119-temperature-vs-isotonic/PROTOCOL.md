# Frozen comparison

{
  "registered_at": "2026-11-04T09:04:49.271170+08:00",
  "sizes": [
    32,
    256
  ],
  "order": "SHA256('119:'+str(idx))",
  "methods": [
    "raw",
    "temperature",
    "isotonic"
  ],
  "isotonic": {
    "increasing": true,
    "out_of_bounds": "clip",
    "input": "positive class probability",
    "tie_prediction": 0
  },
  "nll_epsilon": 1e-06,
  "primary": "audit class-summed Brier; clipped NLL secondary; raw infinite NLL explicitly counted",
  "selection": "none; all two sizes and three methods reported",
  "seed": 119,
  "inputs": {
    "calibration_predictions.json": "1a77b3f03023a4f23cff969ca6d7448dab72e291f1acdae7145d16059d47ab7b",
    "audit_predictions.json": "fde9a72ea40a0377ad92e2916e0edfcacc269d7658bd57648a087d494fa80a44",
    "config.json": "f7112f319e29701ff5fce2f0373a9fc6f55b730f749aed5a646ed7f845e73ad1",
    "split_manifest.json": "5b748d9bdab2f0214d592e9d7b78484e6b2ba5d394bb03e5eb64c94dcd6cb358"
  }
}

SST2 historically observed. Audit independent of calibration fitting but already observed in117/118. No IMDb. Same calibration IDs for methods. One fixed nested32/256 split, no uncertainty or population claim. Fit beta bounded[.05,20] with117 solver. Isotonic squared-error objective; epsilon applies only reporting NLL, not fitting or Brier. Raw probability outputs preserved.

Post-run diagnostic: float sigmoid at T=.05 rounded some positive probabilities to1. Added stable logit-space NLL to distinguish finite mathematical temperature loss from rounded probability zero; original clipped metric and isotonic endpoints unchanged. No new model or selection.
