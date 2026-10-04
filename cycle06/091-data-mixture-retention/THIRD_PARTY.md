# Licenses and distribution

Self-written091 orchestration/audits/docs are released under Apache-2.0 (LICENSE-APACHE-2.0.txt). Copyright2026 distance539. Derived self-written common.py/scoring.py/prepare.py retained from fixed090 publication11c29a7deefedbf06a6f7e52ca6f7af125176535, with complete copies.

Official Transformers4.49.0 commit a22a4378d97d06b7a1d9abad6e0086d30fdea199 is Apache-2.0. Official PyTorch2.6.0 commit2236df1770800ffea5697b11b0bb0d910b2e59e1 has BSD-style license, preserved verbatim in LICENSE-PYTORCH.txt. These packages are installed by requirements, not vendored; APIs are called directly. No official executable code is copied. SmolLM2 model card says Apache-2.0; weights not distributed.

SST2 and AG News cards both say license unknown. This source package contains no raw source texts/parquet, reversible prompt/input token traces, model weights/checkpoints or caches. prepare.py downloads exact public snapshots for the reader's own permitted research use. Published records contain row IDs/labels/hashes, model-produced short labels, metrics and timing; original review/news text is not reproduced. Local checkpoints/input traces remain outside code. No claim that unknown equals public-domain.

AG News provider is fancyzhx on Hugging Face, a packaged dataset mirror; original task/data paper is Xiang Zhang, Junbo Zhao and Yann LeCun(2015), Character-level Convolutional Networks for Text Classification. Official author repository https://github.com/zhangxiangxiao/Crepe. Data provider is distinguished from original research authors.
