# Licensing and attribution

Original teaching code and documentation in this package: Copyright 2026 distance539; Apache License 2.0 (LICENSE-APACHE-2.0.txt). Third-party resources retain their own terms.

- Transformers: Hugging Face and contributors, Apache-2.0; official implementation installed as dependency, source retrieved at fixed commit for audit only. Full original license retained.
- SmolLM2-135M-Instruct: HuggingFaceTB; fixed model card declares Apache-2.0. Weights/tokenizers downloaded from official pinned revision; not bundled.
- SST-2: Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D. Manning, Andrew Ng and Christopher Potts, EMNLP 2013, https://aclanthology.org/D13-1170/ . Stanford-hosted HF dataset card declares license unknown. Original review texts/parquet/card and rendered prompts/input token IDs are excluded from distribution. Users fetch data themselves from the official source and must assess terms for their use. Distributed records contain IDs, labels, hashes, model-generated short responses, model output token IDs and measured metrics only.
- smol-smoltalk was read as background for future mixing work, not downloaded, trained on or redistributed in088.

No full upstream repository, environment, dataset, embedding cache, model weight, personal config or credential is part of this archive. SOURCE_MAP.md maps the official paths. Package manifest and remote comparison are recorded outside code/ to avoid recursive hashing.
