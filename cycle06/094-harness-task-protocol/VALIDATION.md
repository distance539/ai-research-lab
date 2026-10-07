# Validation — 2026-10-07

Technical experiment passed: 64 development rows, 128 candidate requests, 32 correct, max independent/official score difference 0.0. Final-source two-row smoke and offline audits of smoke/full results passed. No training or held-out inference. Version and file hashes: resources.lock.json, upstream_verification.json, each run identity.json. CPU timing/memory: environment.json. Commands/exit codes: commands.json; logs in each run folder.

Initial complete run failed an identity assertion: 61 doc_id mismatches, although scores were correct when keyed by source idx. Only sample argument ordering changed; selected set and all source-aligned scores remained equal. See DEVIATIONS.md and initial_identity_mismatch.json. Earlier four-row manual computation used pre-sort source; independent scoring logic is unchanged. Initial raw results, pre-fix source and failure log are retained locally because they contain original data/private paths.

Human arithmetic review remains pending. HUMAN_REVIEW.md and human_review.json are not a completed review or human gold labels. Official technical run preceded human response; future confirmation is retrospective. Overall article remains incomplete until that requirement is resolved.

Distribution excludes original reviews, context token IDs, weights, caches, environment and credentials. Public records retain derived scores, hashes, indices, shapes, versions, licenses and full executable scripts. ZIP extraction and public commit validation are recorded separately in the article package.
