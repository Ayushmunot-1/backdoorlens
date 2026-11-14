# Repository map — what is where, and which command made it

**New to the project, or need to explain it? Read [`SYSTEM_GUIDE.md`](SYSTEM_GUIDE.md) first** — what every term means, what runs in what order, every script, every parameter, and what the report proves. Then [`RUNBOOK.md`](RUNBOOK.md) for the commands.

Written because the first review failed on navigation, not on evidence: the question *"show me the
exact run"* could not be answered live. Read this before any demo. Nothing here needs to be
remembered — it needs to be **open on a second screen**.

---

## 1 · The four commands

Everything presentable comes out of `run/`. Each is one command and prints what it did.

| command | what it does | runtime |
|---|---|---|
| `python run\1_audit.py --derived <artifact>` | audit ONE artifact, write a **signed certificate**; exit code is the CI gate code | 3–5 min |
| `python run\2_report.py ...` | one self-contained offline **HTML report** (no network, inline SVG, light+dark) | ~1 s |
| `python run\3_evaluate.py ...` | merge both evidence files into **one evaluation**: detection, false alarms, separability | ~1 s |
| `python run\4_demo.py --prepare` then `--pause` | the **five-beat live demo**; `--beat 6` answers the out-of-scope question | 3–5 min / ~45 s |
| `python scripts\build_deck.py ...` | the **review deck** (.pptx), every number read from the evidence files | ~2 s |

Exit codes (same in every entry point and in CI): `0` allow · `10` quarantine · `20` block ·
`30` inconclusive · `40` re-attest.

---

## 2 · The pipeline, in the order it runs

This is the answer to *"what exactly have you implemented, and how?"* Each row is a real module you
can open. `run/1_audit.py`'s docstring carries the same list.

| # | stage | module | what it produces |
|---|---|---|---|
| 1 | artifact manifest | `scope.py` | SHA-256 per in-scope component, parameter/tensor metadata, `completeness` (known / partial / unknown_transformation) |
| 2 | artifact differential | `bundle_diff.py` | tokenizer, template, config and generation diffs — plus the template **rendered** on a fixed corpus, which is how `template_inject` is caught |
| 3 | comparability | `scope.py` → `comparability()` | are these a reference/derived pair at all? **refuses here, before any model loads**, if vocab_size or model_type disagree |
| 4 | load both models | `backends/hf.py` | the only expensive step |
| 5 | probe suite | `probes.py` | 128 seed prompts, rule-based semantic variants, validity-constrained mutation — from the seed the null was fitted with |
| 6 | paired measurement | `divergence.py`, `search.py` (`PairedOracle`) | per-probe JSD, KL, top-k overlap, rank correlation, logit margin, NLL shift, argmax flip |
| 7 | calibration | `calibration.py`, `stats.py` | computed inside stage 6 against the supplied null: level and shape, t prediction interval, df = k−1 |
| 8 | budgeted search | `search.py` | benchmark / random / guided, all cost-matched on paired evaluations |
| 9 | confirmation | `audit.py` → `confirm()` | semantic variants + **replay through a separate call path** |
| 10 | component differential | `weight_delta.py` | tensor groups changed that no honest derivation of the declared family changes |
| 11 | decision | `decision.py` | the verdict. **Deterministic policy; no LLM is involved in this module** |
| 12 | attestation | `attestation.py` | ed25519-signed, scope-bound, replayable certificate |

The **gate is not part of the audit**: `monitor.py`, run as `backdoorlens gate` or demo beat 5. It
loads no model — it re-hashes the artifact and compares against the certificate, which is why
invalidation takes about a second regardless of model size.

Orchestration is `audit.py` (`run_audit`); `experiments.py` runs the whole multi-stage experiment;
`derive.py` builds the honest and controlled-tampered artifacts; `tensors.py` extracts parameter
metadata; `validation.py` checks that a planted behaviour actually installed (a tamper that does not
exist cannot serve as ground truth).

**"What parameters do you use to detect abnormal behaviour?"** — rows 4 and 9, and the answer is that
no parameter has a fixed threshold. Every behavioural statistic is a **ratio against the audited
artifact's own level**, tested against the honest derivations of its declared family. See
`docs/RESULTS.md` §2 for the per-statistic table with the gate each one must clear.

---

## 3 · Folders

```
src/backdoorlens/   the engine — 18 modules, table above
run/                the four presentable entry points (this is where to start)
scripts/            experiment and analysis tooling; produces what run/ reports on.
                    scripts/bl_tools.py = the cross-platform toolbox the RUNBOOK uses to build
                    honest/tampered artifacts, fit the component null, declare and triage adapters, print a report
configs/experiments/ one JSON per experiment; smollm2_135m_final.json is the primary one
results/            evidence (below)
tests/              116 tests
docs/               this map, RESULTS.md, DEMO_RUNBOOK.md, REVIEW_v2_v3_and_fixes.md
paper/ deck/        generated from results.json — no number is typed by hand
models/             local reference bundles (not in git)
```

## 4 · Evidence: which command produced which file

| file | produced by | holds |
|---|---|---|
| `results/smollm2/results.json` | `scripts/run_experiments.py --config configs/experiments/smollm2_135m_final.json` | the 7-stage experiment: E1 conventional baselines (19 rows = 12 held-out honest + 7 tampered), 133 audits across 3 search strategies and 3 seeds, honest false-alarm rate, per-artifact detection, E8 invalidation |
| `results/smollm2/null_int4.json` | same run | the behavioural null: 6 honest int4 derivations, budget 128 |
| `results/smollm2/artifacts/` | `derive.py` via the same run | **31** bundles: 12 calibration (`cal4_*`, `cal8_*`), 12 held-out honest (`test4_*`, `test8_*`), 7 tampered (`t_*`) — `n_honest_cal` and `n_honest_test` are 6 each, at both bit-widths |
| `results/smollm2/artifacts/_truth/` | `derive.plant_trigger_behavior` | ground truth for planted artifacts, written **outside** the bundle so it can never leak into one |
| `results/component_null_int4/component_null_audit.json` | `scripts/component_null_audit.py --honest-prefix cal4` | the component null **matched to one family** (6 int4 derivations) — the one to quote |
| `results/component_null/component_null_audit.json` | same, `--honest-prefix cal` | pooled int4+int8 (12). Same 62 groups, 0/12 leave-one-out — a robustness check, not the headline |
| `results/smollm2_strong/` | same pipeline, `plant_steps` 100 | independent replication of the component finding |
| `results/demo/` | `run/4_demo.py --prepare` | the signed certificate, HTML report, tampered copy and out-of-scope bundle for the live demo |
| `deck/BackdoorLens_review2.pptx` | `scripts/build_deck.py` | the 15-slide review deck. `deck/BackdoorLens_synthetic_validation.pptx` is the OLD deck, built from the toy run and carrying a SYNTHETIC banner — do not present it |
| `results/toy_quick/` | `configs/experiments/toy_quick.json` | NumPy toy LM. **Pipeline validation only — never a research result** |

Naming trap: `results.json` keys artifacts by experiment name (`template_inject`) while the artifact
directories use a `t_` prefix (`t_template`). The mapping is declared in `run/3_evaluate.py`
(`TAMPER_DIR`), read off `experiments.py`, not guessed.

---

## 5 · If a panelist asks X, run Y

| question | command | what they will see |
|---|---|---|
| "show me a real run" | `python run\1_audit.py --derived results\smollm2\artifacts\t_trigger_easy --sign results\demo\key.pem` | the full audit, verdict SUSPICIOUS → QUARANTINE, exit 10 |
| "what does your output look like?" | open `results\demo\report.html` | seven sections, every number traceable |
| "how often does it false-alarm?" | `python run\3_evaluate.py --results results\smollm2\results.json --component results\component_null_int4\component_null_audit.json` | 0/96 behavioural, 0/6 + 0/18 component |
| "couldn't a perplexity check do this?" | same command, first block | worst honest int4 **0.14277** > lowest tampered **0.14027** — no threshold separates them |
| "what if I change the artifact after you certify it?" | `python run\4_demo.py --beat 5` | one byte flipped → INVALIDATED, exit 40, ~1 s, no model loaded |
| "what if I hand you a model you have no reference for?" | `python run\4_demo.py --beat 6` | refused, `behavioural_measurement: not_performed`, no model loaded |
| "which tensors changed?" | `python scripts\bundle_diff_report.py` / the report's §4 | the 62 groups, named |
| "would a benign fine-tune also flag?" | — | **Yes. Say so.** The component channel detects provenance deviation, not malice. It is in the report's limitations and in `docs/RESULTS.md` §3.3 |

---

## 6 · What is deliberately not here

`scripts/` contains two **falsified** hypotheses, kept on purpose: `weight_delta_localize.py` (raw
embedding-row ranking does not locate the trigger token) and `weight_delta_ksweep.py` (removing the
dense component does not rescue the ranking — the rank-1 premise is false). Both are written up in
`docs/REVIEW_v2_v3_and_fixes.md`. They are negative results, not dead code, and the planned
delta-guided search strategy was **not built** because its premise failed. If asked why they are
still in the repo: because deleting a failed experiment is how a result becomes unreproducible.
