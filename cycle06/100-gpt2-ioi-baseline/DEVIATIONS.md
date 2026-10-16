# Preserved failures and boundaries

2026-10-16: first anonymous network attempt inside sandbox could not connect to configured proxy (curl exit7). Authorized public network retry succeeded. No credential extraction.

First real smoke (exit1) stopped at vector readout absolute tolerance1e-4 before any records were saved. Original run.py/plan.json/PROTOCOL.md/stages/execution preserved in smoke-readout-failed. Diagnostic on the same first prompt found vector vs batched max0.00016021728515625, float64 vs float32 max same, same-shape readout0; max absolute logit85.28097534179688. Before formal inference changed ONLY check to exact same-shape readout and retained vector error diagnostic. Formal maximum vector error0.0001983642578125; same-shape0. No behavioral score/template selection occurred in response. The diagnostic did run first-row inference; it is development, not held-out confirmation.

Environment reuse: attempted pip freeze failed because reused uv venv has no pip. Read installed metadata and resolved required dependency closure instead;25versions in environment.lock.txt, full observed environment in environment_observed.json. No claim of newly installed environment test. Original direct dependency guesses were corrected before publication to installed versions; inference source/inputs unchanged.

Original official IOI code uses random templates/nouns and a larger name pool. We deliberately fix two templates, store/ring, eight names, balanced roles and two orders. No natural-distribution claim, no match to published aggregate, no activation intervention. All64scores and all four cells reported. Name and template holdouts are configuration only, not run.
