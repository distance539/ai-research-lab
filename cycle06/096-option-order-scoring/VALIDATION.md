# Validation — 2026-10-09

- Final real2-question x5-condition CPU smoke exited0; estimated280.124s <1200s budget;1369161728B <4GiB.
- Full64-question x5-condition CPU run exited0;640 candidate likelihoods independently/officially scored, max error0. Raw per-candidate evidence retained in results/records.json, no source-review text.
- Independent audit exited0 on smoke and full records. All mapping, arithmetic, shapes and pair identities checked. Six corruption mutation tests exited0. These checks do not constitute human annotation.
- Model/tokenizer/data12files SHA256 verified at each run; six installed official harness files equal fixed upstream source. Confirmations untouched.
- Two initial label-target failures preserved, see DEVIATIONS.md. Only the final corrected implementation used for accepted smoke/full results; scientific protocol unchanged.
- Source archive extraction, offline audit, syntax/help and real minimal entry are required prior to publication; corresponding release evidence is kept with article publication/package records. Original successful source hashes and records must remain identical.

Local verification_logs hold exact machine invocation and private original stderr. Public execution.json uses a cache placeholder to avoid leaking local filesystem layout, and links the byte-identical successful raw run.log.
