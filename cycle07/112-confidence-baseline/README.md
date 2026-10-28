# Confidence baseline on SST-2

Real DistilBERT inference on a fixed exploratory subset of256 validation examples. No training, temperature fitting, or model selection. This is not an open-generation hallucination benchmark and not a complete reproduction of Guo et al.'s experiment matrix.

## Run

Python3.12 and curl are required. Install requirements.txt into an isolated environment. We used CPUfloat32,4threads. No API key is needed. Download size is about268MB for weights, plus tokenizer and a small validation parquet. Do not redistribute raw SST-2 text from this package.

```bash
python -m pip install -r requirements.txt
python execute.py prepare --cache .cache112
python execute.py smoke --cache .cache112
python execute.py smoke_audit
python execute.py formal --cache .cache112
python execute.py audit
```

The cache is external to this published package. The download action uses pinned official Hugging Face URLs, verifies model LFS SHA256 or Git-blob identities, and records file hashes. Resource locks must match before inference. Network is required only for prepare. For an offline replay, reuse a hash-verified cache, not a machine-specific hidden path.

The original machine's cloud-backed library folder stalled during import. The run was stopped before inference; the same Transformers4.49.0 release was installed in a local temporary location. This changes storage, not the model or protocol. See execution history in the article's verification directory. Installed model source was compared with its official fixed commit. No unrelated environment was changed.

## Outputs

- split_manifest.json: all872 IDs divided into exploration256, calibration256, audit360 before inference.
- formal_predictions.json:256 records of sample ID, text hash, gold label, two logits/probabilities, confidence, correctness and token lengths. No sentences.
- formal_summary.json: accuracy, confidence, signed gap, ECE10, NLL, two-class summed Brier, threshold0.9 denominator, per-bin statistics, timing and peak resident memory.
- formal_environment.json and execution files: actual versions, shapes, commands, status and raw logs.
- formal_audit.json: independent standard-library reconstruction and six mutation tests. This is automated verification, not human annotation.

All input-label mappings and predictions are bound to the data hash. Metrics are descriptive on reused public development data. None is a population guarantee. The checkpoint's authors already used SST-2 validation for evaluation; the previous blog series also used this dataset. Do not call these results uncontaminated or unseen confirmation.

## Files and source lineage

Read PROTOCOL.md before changing anything; configuration and protocol hashes are in each start record. SOURCE_MAP.md identifies official model code and the distinct local metric implementation. requirements.txt pins the observed major dependencies. LICENSE covers newly written scripts; third-party resources retain their own licenses and are downloaded rather than bundled. There is no requirement for a hosted inference API, GPU, or secret credential.
