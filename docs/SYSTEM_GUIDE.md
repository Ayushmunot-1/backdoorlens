# BackdoorLens: how the whole thing actually works

Read this one before the runbook. The runbook tells you *what to type*; this tells you *what is
happening and why*, in plain language, with the jargon explained at the point it first matters.

---

## 1 · The system in one paragraph

You approve a model. Before it is served, somebody changes it — almost always by quantizing it so it
fits in memory. That change is deliberate, so the publisher's signature no longer verifies, and
nothing in normal practice checks whether the model still *behaves* the way the approved one did.
BackdoorLens takes the model you approved (the **reference**) and the model you are about to serve
(the **derived artifact**), measures how differently they behave, and asks one question: *is this
difference bigger than what this particular kind of change normally causes?* To answer that it needs
examples of the change done honestly — you run your own quantizer six times and it learns what honest
looks like. Then it issues a signed certificate tied to the exact files, which stops being valid the
moment anyone touches them.

---

## 2 · The vocabulary, in plain English

You need these eight words. Everything else follows.

**Reference.** The model you vetted and trust. Usually what you downloaded.

**Derived artifact.** What you are actually going to serve, after quantization, conversion, adapter
merging, repackaging — whatever your pipeline does.

**Transformation family.** *Which kind of change* was declared. "INT4 round-to-nearest with group
size 128" is a family. "INT8" is a different family. This matters enormously: INT4 changes behaviour
far more than INT8, so judging an INT4 artifact by INT8's standards flags everything.

**INT4 / INT8.** Quantization precision. Model weights are normally 16- or 32-bit numbers. Quantizing
to 8 bits (INT8) or 4 bits (INT4) shrinks the file so it fits in less memory, at the cost of
precision. Fewer bits = smaller and faster = more behaviour change. In this project the quantization
is *simulated*: `fake_quantize_state` rounds each weight to the nearest representable value and
converts it straight back to float, so you get the rounding **error** of real quantization without
needing a real INT4 kernel. It processes 2-D weight tensors and skips anything whose name contains
`embed`, `lm_head` or `norm` — remember that skip list, it is the reason the component channel works.

**Honest derivation.** The same transformation, done legitimately, varying only by a seed. Six of
them tell you the natural spread of that transformation. These are your controls.

**Null.** The statistical description of that spread. Not a threshold someone chose — a distribution
measured from your own honest derivations. Everything is judged against it.

**Level.** The audited artifact's own typical divergence (median across all probes). Every statistic
is a *ratio* against this, which is how the test adapts to transformations that move behaviour a lot
versus a little.

**Probe.** One test prompt. The suite is 128 of them, fixed and reproducible from a seed.

---

### The artifacts you build, and why each exists

| name | what it is | what it is for |
|---|---|---|
| `cal4_*`, `cal8_*` | honest INT4 / INT8 quantizations, 6 each | **fit the null.** These define "normal" |
| `test4_*`, `test8_*` | 6 more honest ones each, never used for fitting | **measure false alarms.** If these flag, the method cries wolf |
| `t_template` | chat-template injection | a backdoor that touches **no weight at all** |
| `t_gen` | generation-config edit | tampering with the serving policy, not the model |
| `t_trigger_easy` | planted behaviour, trigger word `lantern` | a real conditional backdoor, with a trigger that is **in** the tokenizer's normal vocabulary |
| `t_trigger_hard` | planted behaviour, trigger `zq7x` | same, but a trigger that is **out of** normal lexicon — harder to stumble on by chance |
| `t_canary_0.5`, `2.0`, `8.0` | random Gaussian noise added to 2% of weights, at three strengths | **not an attack** — a measuring stick. By sweeping the strength you find the smallest change the method can still detect |

"Canary" is the mining metaphor: a deliberately planted, harmless signal used to test whether the
detector is alive. σ=0.5 is deliberately tiny. It is *expected* to be hard, and in the results it is
missed by both channels — that is the sensitivity floor, reported honestly rather than hidden.

"Easy" and "hard" refer to **how findable the trigger is**, not how damaging it is. Both plant the
same harmless marker (`ZXQ-CANARY`).

---

### Why LoRA appears, and what it is

**LoRA** = Low-Rank Adaptation. Normal fine-tuning updates all of a model's weights, which is slow
and produces a full-size copy. LoRA freezes the original weights and trains two small matrices per
layer whose product is added to the original. The result is a tiny file (megabytes, not gigabytes)
that is published and distributed *on its own*, to be applied on top of a base model at load time.

It appears in this project for one reason: **that is the shape real-world derived artifacts often
take.** Someone publishes an adapter, you apply it to a base model you trust, and you are now serving
something nobody verified. Two functions exist for this:

* `derive_adapter_only` — builds an adapter-only bundle, matching the shape of a published LoRA
  release, so the adapter code paths can be tested.
* `derive_benign_lora_sft` — trains an *honest* LoRA on a small generic instruction set. You need
  six of these to audit someone else's adapter, because "honest derivation of the same family" here
  means "an ordinary instruction-tuning LoRA run with the same configuration".

**Known limitation:** the component channel compares full weight tensors on both sides, so it does
not apply to adapter-only bundles. Those get the behavioural channel alone. Say this before a
panelist asks.

---

## 3 · The workflow: what runs, in what order

`scripts/run_all.ps1` is the ordered pipeline. It already existed in the repo and does the long path
in five stages.

```powershell
.\scripts\run_all.ps1 -Stage 0      # install, run tests, smoke-test the HF path on a tiny model   ~5 min
.\scripts\run_all.ps1 -Stage 1      # download the real model, measure your machine's speed        ~10 min
.\scripts\run_all.ps1 -Stage 2      # build all 31 artifacts AND verify the planted triggers exist  1-2 h
.\scripts\run_all.ps1 -Stage 3      # the full experiment: calibration + 133 audits                 several h
.\scripts\run_all.ps1 -Stage 4      # report tables and figures                                     seconds
```

Stage 2 is a **go/no-go gate**: it refuses to continue if a planted trigger did not actually install,
because a tamper that does not exist cannot serve as ground truth. Stage 3 is resumable — stages are
checkpointed and `audits.jsonl` is append-only, so if it dies at hour five you re-run the same command
and it continues.

### Then the newer pieces, which post-date `run_all.ps1`

```powershell
# the second detection channel (fast, no inference at all)
python scripts\component_null_audit.py --ref models\SmolLM2-135M-Instruct --work results\smollm2\artifacts --honest-prefix cal4 --out results\component_null_int4

# one signed certificate for a single artifact
python run\1_audit.py --derived results\smollm2\artifacts\t_trigger_easy --sign results\demo\key.pem

# the four outputs
python run\3_evaluate.py  ...     # the merged evaluation, printed to the terminal
python run\2_report.py    ...     # the self-contained HTML report
python scripts\paper_numbers.py ...  # every macro, table and figure the paper cites
python scripts\build_deck.py    ...  # the 15-slide deck
```

### Dependency order, as a picture

```
reference model
      |
      v
 honest derivations  ----> behavioural null (calibrate)
 (cal4_*, cal8_*)    ----> component null  (component_null_audit.py)
      |                              |
      |                              v
 tampered artifacts  ------>     AUDIT (run_audit)  <---- probe suite (128 prompts, from a seed)
 (t_*)                               |
 held-out honest                     v
 (test4_*, test8_*)             attestation.json  ---> gate / verify / replay
                                     |
                                     v
                   results.json + component_null_audit.json + attestation.json
                                     |
                     +---------------+---------------+------------------+
                     v               v               v                  v
                 3_evaluate     2_report        paper_numbers      build_deck
                 (terminal)     (HTML)          (paper)            (deck)
```

**Nothing downstream can run before the nulls exist.** That is the single most common source of
confusion: an audit without a matched null returns `INCONCLUSIVE`, which looks like a failure but is
the system correctly refusing to judge without a baseline.

---

## 4 · What happens inside one audit, from model load to verdict

This is the answer to "where does the model load and what happens after that". Twelve stages, in the
order `audit.run_audit` performs them.

| # | stage | module | cost |
|---|---|---|---|
| 1 | Hash every component of both bundles; extract tensor and parameter metadata | `scope.py` | seconds |
| 2 | Diff the bundles: tokenizer, config, generation policy, and the chat template **rendered** over a fixed conversation corpus | `bundle_diff.py` | seconds |
| 3 | Comparability: are these even a reference/derived pair? Different vocabulary → refuse here | `scope.py` | instant |
| 4 | **The models load here.** Not before. | `backends/hf.py` | ~20 s |
| 5 | Build the 128-prompt probe suite from the same seed the null used | `probes.py` | instant |
| 6 | Push every probe through **both** models, record 7 metrics per probe | `divergence.py` | minutes |
| 7 | Compare against the null: is this level abnormal for this family? | `calibration.py`, `stats.py` | instant |
| 8 | Budgeted search: 128 more paired evaluations hunting for a worse prompt | `search.py` | minutes |
| 9 | Confirm: re-test top candidates on semantic variants, then replay through a separate code path | `audit.py` | seconds |
| 10 | Component differential: which tensor groups changed that no honest derivation touches | `weight_delta.py` | seconds |
| 11 | Deterministic policy picks the verdict. **No LLM is involved.** | `decision.py` | instant |
| 12 | Sign with ed25519, write `attestation.json` | `attestation.py` | instant |

The **gate** is deliberately *not* in this list. It is a separate command that re-hashes the artifact
and compares it to the certificate. It loads no model, which is why invalidation takes about half a
second no matter how big the model is.

---

## 5 · Every parameter the system evaluates

### 5a · Measured per probe (7 numbers × 128 probes × 2 models)

| metric | what it measures | why it is there |
|---|---|---|
| **JSD** (Jensen–Shannon divergence) | overall distance between the two next-token probability distributions, 0 to 1 | the primary signal; symmetric and bounded, so it is comparable across prompts |
| **KL divergence** | how surprised the derived model is by the reference's distribution | unbounded, sensitive to the tail — catches changes JSD can smooth over |
| **top-k overlap** | how many of the top 10 candidate tokens are shared | a change that reorders the plausible answers |
| **top-k rank correlation** | whether the *order* of those candidates survived | distinguishes "same words, new priorities" from "different words" |
| **logit margin change** | how much the gap between first and second choice moved | measures confidence shift, not just choice shift |
| **NLL shift** | change in negative log-likelihood of the reference's own top token | directly interpretable: did the derived model become less sure of the right answer? |
| **argmax flip** | did the single most likely next token change at all? | the blunt, legible one — easy to explain to a non-specialist |

And at the output level, after actually generating text: **structural flip** (did the answer's shape
change — refusal, JSON validity, emptiness) and **edit distance ratio** (how different the generated
strings are).

### 5b · Aggregated into test statistics

| statistic | question it answers | gate |
|---|---|---|
| **level** (median JSD) | how far has this artifact moved *on average*? | descriptive; the denominator for everything else |
| **p_level** | is that level abnormal vs honest derivations? | ≤ α/3, **and** `level_ratio_vs_max_honest` ≥ `min_level_ratio` |
| **p_spike** | is the single worst prompt abnormal relative to this artifact's own level? | ≤ α/3, **and** spike ratio ≥ `min_spike_ratio` |
| **p_search** | is the best thing the search found abnormal vs honest search maxima at the same budget? | ≤ `alpha_search`, **and** margin ≥ `min_search_margin` |
| **p_binom** | pooled per-prompt tail | **descriptive only** — pooling prompts within one derivation is pseudo-replication. Never quote it as a result |
| **argmax flip rate** | fraction of probes whose top token changed | descriptive |
| **component hard anomalies** | how many tensor groups did this artifact change that no honest derivation changes? | a **count**, never a p-value — the honest null is identically zero, so no interval exists |
| **undeclared bundle diffs** | behaviour-relevant file differences the derivation did not declare | any ≥ 1 → suspicious |

### 5c · The policy constants (`decision.py`, version `policy-v3.0`)

| parameter | value | meaning |
|---|---|---|
| `alpha_stage_a` | 0.005 | family-wise significance for the three stage-A tests (Bonferroni, so each gets α/3) |
| `alpha_search` | 0.01 | significance for the search maximum |
| `alpha_variant` | 0.05 | per-variant exceedance when confirming a counterexample |
| `min_level_ratio` | 2.0 | **effect-size guard**: level must be ≥ 2× the largest honest level |
| `min_spike_ratio` | 2.0 | same guard for the spike statistic |
| `min_search_margin` | 1.5 | same guard for the search statistic |
| `persist_min_variants` | 3 | a counterexample must survive ≥ 3 semantic variants |
| `min_null_derivations` | 5 | fewer honest derivations than this and variance is not estimable |
| `min_null_prompts` | 100 | minimum per-prompt scores in the null |
| `min_budget` | 128 | below this the audit is `INCONCLUSIVE` by design |
| `require_matching_null` | true | a null from another family is refused, not substituted |
| `component_anomaly_is_suspicious` | true | a hard component anomaly → QUARANTINE |
| `undeclared_diff_is_suspicious` | true | undeclared behaviour-relevant file change → QUARANTINE |
| `watch_alpha` | 0.05 | below this the audit is flagged for a larger follow-up, without failing |
| `cert_ttl_days` | 30 | certificate lifetime. A prototype default, **not** a standard |

**The guards are the key design idea.** A p-value alone says "this is unlikely by chance". The guards
add "…and it is also *big enough to matter*". Without them, at 128 probes, tiny consistent
differences become statistically significant and you get false alarms. With them the false-alarm rate
is 0 in 96. The price is a blind spot — see §7.

---

## 6 · The scripts folder, one by one

### Run these — the main path

| script | what it does | when | time |
|---|---|---|---|
| `run_all.ps1` | **the ordered pipeline**, stages 0–4 | always start here | — |
| `run_experiments.py` | the whole 7-stage experiment: builds artifacts, fits nulls, runs 133 audits, writes `results.json`. Resumable | stage 3 | hours |
| `component_null_audit.py` | fits the component null and audits every bundle against it. No inference | after the artifacts exist | minutes |
| `make_report.py` | tables and figures from `results.json` only | stage 4 | seconds |
| `paper_numbers.py` | every macro, table and figure the **paper** cites, from all evidence files | before building the paper | seconds |
| `build_deck.py` | the 15-slide review deck | before presenting | seconds |
| `build_paper.py` | runs pdflatex twice (needs MiKTeX) | after `paper_numbers.py` | seconds |

### Run these before committing to a long run

| script | what it does |
|---|---|
| `make_tiny_reference.py` | builds a tiny random-weight model so you can test the pipeline in minutes without downloading anything |
| `benchmark_throughput.py` | measures seconds per paired evaluation **on your hardware** |
| `size_run.py` | measures, then estimates the whole run for a config |
| `estimate_cost.py` | estimates from a number you already know |

### The real-world experiment (a third-party artifact you did not build)

| script | what it does |
|---|---|
| `real_world_adapter_audit.py` | audits a **real, third-party LoRA adapter**: trains 6 honest LoRAs of the same config as the null, then audits the real one. This is the "we did not control the provenance" experiment |
| `declare_and_reaudit.py` | second pass. The first pass returned `INCONCLUSIVE` because the real adapter declares no transformation; this declares the scope explicitly and re-audits so the statistics actually run |
| `bundle_diff_report.py` | prints the **full** per-field bundle differential, not just the counts in the attestation. Written because the re-audit came back SUSPICIOUS for a reason the summary counts did not explain |
| `real_lifecycle_demo.py` | the capstone story on real evidence: sign → verify → gate (ALLOW) → tamper → gate (INVALIDATED) → replay |

### Investigation and diagnostics

| script | what it does | status |
|---|---|---|
| `recheck_spike.py` | re-runs only the previously-missed artifacts against an **existing** null, to test a statistic change in minutes instead of re-running the whole pipeline | utility |
| `weight_delta_localize.py` | tested whether the planted trigger's token can be read out of the embedding delta | **FALSIFIED** — the trigger ranked 17th–681st of 49,152. Kept with its code |
| `weight_delta_ksweep.py` | tested whether removing the dense component sharpens that ranking | **FALSIFIED** — made 2 of 4 cases worse; the delta is high-rank, so the premise was wrong |

Those last two are negative results kept deliberately. Deleting a failed experiment is how a result
becomes unreproducible, and a planned "delta-guided search" feature was **not built** because its
premise failed. If a panelist asks why dead code is in the repo, that is the answer.

---

## 7 · What the tests folder is

`tests/` is 116 automated checks that run in about 50 seconds (`python -m pytest -q`). They are not
decoration: each one pins a behaviour that was once wrong or could silently break.

| file | tests | what it protects |
|---|---|---|
| `test_core.py` | 19 | statistics, scope hashing, decisions, signing, invalidation, canary |
| `test_weight_delta.py` | 19 | the component channel on synthetic bundles |
| `test_scope_comparability.py` | 10 | an unrelated model is refused **before any model loads** — one test fails if a model is loaded at all |
| `test_component_anomaly_decision.py` | 9 | the component anomaly reaches the verdict, and is completely inert when no component null is supplied |
| `test_adapters.py` | 8 | an adapter present but not applied is a scope gap, never silently ignored |
| `test_spike_statistic.py` | 7 | the spike statistic, added after a real measured gap |
| `test_null_match_diagnostics.py` | 6 | a mismatched null names *which field* is wrong |
| `test_signing_key_failfast.py` | 6 | `--sign` fails in a second, not after a four-minute audit |
| `test_parameters.py` | 6 | the parameter/tensor manifest |
| `test_patches.py` | 7 | earlier fixes stay fixed |
| `test_demo_benign_guard.py` | 5 | the live demo cannot show a prompt containing a planted trigger |
| `test_plant_flow.py` | 2 | the planting control flow, without needing torch |
| `test_cli_crossshell.py` | 12 | the runbook's cross-shell options (`--adapter-base`, `--policy @file`, `--pin FILE`) and every `scripts/bl_tools.py` command, including `diff-triage` refusing to wave through an adapter that ships its own chat template |

"116 tests pass" is a claim you can make to a panel, and this table is what it means.

---

## 8 · What the report proves — and what it does not

The HTML report and the paper make exactly four claims. Each is a measurement, not an argument.

**1. No conventional check separates honest work from tampering.** Hash, serialization scan and
template scan pass *every* artifact including the tampered ones — the template scanner even passes the
template injection, because the injected text contains no code for a regex to find. Relative
perplexity does react, but its honest and tampered ranges **overlap**: worst honest INT4 is 0.14277,
lowest tampered is 0.14027. No threshold separates them. *This is why the project exists.*

**2. A mismatched null is worse than useless.** A threshold fitted on INT8 and applied to honest INT4
derivations flags **100%** of them. The matched null flags **0%**. This is the experimental
justification for the whole calibration design, and it is the strongest single result in the project.

**3. Two channels detect what neither does alone.** Behavioural alone catches 4 of 7; with the
component channel, 6 of 7 — and the component channel catches *exactly* the two the behavioural tests
miss, at 0 false alarms in 96 honest audits. On the backdoored artifact, the behavioural channel alone
yields ALLOW_WITH_MONITORING: **it would have been deployed.**

**4. The certificate enforces itself.** Every in-scope component modification invalidates it, in about
half a second, with no model loaded — and a change *outside* the scope correctly leaves it valid.

### What it does not prove

It does not prove the model is safe. It does not prove there is no backdoor. It does not show guided
search is better than random (measured: it is not). It does not generalise beyond one model family at
one size on CPU. And it has a known blind spot, which the report states before anyone finds it: a
modification pervasive enough to raise the artifact's own level hides from ratio-based tests — on the
backdoored artifact p_level is significant at 4.76×10⁻⁴ and still does not fire, because the level is
only 1.546× the honest maximum against a 2× guard. Removing the guard would not fix it; the guard is
why there are no false alarms. The second channel is the answer.

---

## 9 · If you remember five things

1. **It compares two models, not one.** Everything needs a reference.
2. **"Normal" is measured, not chosen.** Six honest runs of your own pipeline define the baseline.
3. **There are two independent detectors.** Behaviour (what it does) and components (what the
   transformation touched). Neither is sufficient.
4. **The verdict is deterministic.** Same evidence, same answer, every time. No LLM decides anything.
5. **ATTESTED does not mean safe, and INCONCLUSIVE does not mean safe.** It certifies a derivation,
   within a stated scope, at a stated budget.
