# Attribution

Local experiment scripts: Copyright 2026 distance539; Apache-2.0.
BEIR: Nandan Thakur, Nils Reimers and contributors; Apache-2.0.
Sentence Transformers: Nils Reimers and contributors; Apache-2.0.
Transformers: Hugging Face and contributors; Apache-2.0.
MiniLM checkpoint: sentence-transformers/all-MiniLM-L6-v2, model card declares Apache-2.0. Weights not redistributed.
SciFact: David Wadden, Shanchuan Lin, Kyle Lo, Lucy Lu Wang, Madeleine van Zuylen, Arman Cohan, Hannaneh Hajishirzi (2020), Fact or Fiction: Verifying Scientific Claims, https://aclanthology.org/2020.emnlp-main.609/ . Claims/evidence CC-BY-4.0, abstracts ODC-By-1.0; see original license. Full dataset not bundled; qrels/IDs from the attributed dataset retained for evaluation. Generated ranks/metrics are this experiment's results; no annotation changes.

`Encoder.encode_corpus` adapts BEIR title/text joining; model-card masked mean equation is independently rewritten for shape checks. Other local scripts are independent or reused local082 utilities; upstream frameworks execute through imports. See SOURCE_MAP.md for exact mapping and licenses.
