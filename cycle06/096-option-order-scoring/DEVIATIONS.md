# Failures and implementation corrections

Recorded2026-10-09 before formal evaluation. The shared OneDrive environment hung during Python startup; recovered095 source from its public fixed commit and rebuilt an isolated temporary environment from the same dependency lock and pinned harness commit. No model/data/configuration change. Initial offline package install failed because the cache lacked dependencies; ordinary pinned downloads recovered it. Model/data caches were reused only after SHA256 validation.

1. First smoke stopped on official metric agreement. A string `doc_to_target='position_gold'` was not an actual column in the original dataset, so harness treated it as template text instead of a field. No completed result was accepted. smoke-target-failed retains identity, stages and failure; original stderr and source snapshot retained in the article's private verification_logs (stderr prints two original dataset rows, so it is not distributed).
2. First repair used a callable reading that same added key; ConfigurableTask probes an original sample at initialization and raised KeyError before inference. smoke-init-failed records this unsuccessful repair.
3. Final repair computes `mapping.index(doc['label'])` directly from the original label in the callable. It works both during task initialization and evaluation. All official/independent predictions are subsequently compared. No selection based on experimental performance, no budget or scientific protocol change.

The final smoke and full run identities record exact corrected source hashes. Earlier failures are historical evidence, not successful experiments or human review. The machine-readable smoke logs are preserved locally; public successful-run logs contain no source reviews.
