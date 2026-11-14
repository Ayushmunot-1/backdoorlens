# BackdoorLens — behavioural attestation for derived open-weight LLM artifacts

**Signing proves the bytes didn't change. Nothing proves the behaviour didn't.**

An organisation verifies the model it downloads, then quantizes it to fit the serving budget — and at
that moment the publisher's signature is void and nothing checks that behaviour survived. BackdoorLens
is that check: does the exact **derived bundle** (weights, adapters, tokenizer, chat template, config,
generation policy, quantization, runtime) remain behaviourally consistent with a **trusted reference**,
within a declared transformation family, input distribution, runtime scope and audit budget?

It measures paired divergence, calibrates what *honest* derivations of the same transformation look
like, runs a budgeted counterexample search, compares the weight delta against what that transformation
is supposed to touch, issues a signed replayable certificate, and invalidates it when any in-scope
component changes.

> **A passing attestation is bounded evidence, never a proof that a model is safe or backdoor-free.**
> It certifies a *derivation*, within a *stated scope*, at a *stated budget*.

**Want to run it? Follow [`docs/RUNBOOK.md`](docs/RUNBOOK.md)** — every command, in order, for deriving your own models and for auditing someone else's, with what each does and what you should see.

**New here? Read [`docs/MAP.md`](docs/MAP.md)** — what is in which folder, which module does what, and
which command produced which evidence file.

---

## Start here

```
python -m venv venv
#   activate:  Windows PowerShell  .\venv\Scripts\Activate.ps1   |   macOS / Linux  source venv/bin/activate
python -m pip install -e ".[hf,report,dev]"
python scripts/bl_tools.py doctor     # checks Python and packages
python -m pytest -q                   # 116 tests, ~50-90 s
```

Four commands produce everything presentable:

```
python run/1_audit.py --derived results/smollm2/artifacts/t_trigger_easy   # audit one artifact, sign it
python run/2_report.py --results ... --out report.html                     # one offline HTML report
python run/3_evaluate.py --results ... --component ...                     # the whole evaluation
python run/4_demo.py --prepare ; python run/4_demo.py --pause              # the six-beat live demo
```

CI exit codes, everywhere: `0` allow · `10` quarantine · `20` block · `30` inconclusive ·
`40` re-attest.

---

## Current status

| | |
|---|---|
| tests | **116 passing** |
| primary evidence | **real model**: SmolLM2-135M-Instruct, 31 derived bundles, CPU, `configs/experiments/smollm2_135m_final.json` |
| independent replication | `results/smollm2_strong` (`plant_steps` 100 vs 40) reproduces the component finding |
| synthetic | `results/toy_quick` (NumPy toy LM) — **pipeline validation only, never a research result**; the paper and deck show a banner while `"synthetic": true` |
| not run here | GPU; real bitsandbytes/GPTQ/GGUF kernels (quantization is *simulated* RTN); models above 135M |

### Headline measured results

- **No conventional check separates honest work from tampering.** Hash/signature, serialization scan
  and chat-template static scan pass **all 19** held-out and tampered artifacts — the template scanner
  passes `template_inject` itself. The only check that reacts, relative perplexity, **overlaps**: worst
  honest int4 `0.14277` exceeds lowest tampered `0.14027`, so no threshold passes every honest
  quantization and catches every tamper.
- **Detection 4/7 behavioural → 6/7 with the component channel.** The two channels are independent:
  the component channel catches exactly the artifacts every behavioural test missed. One artifact
  (`canary_0.5`) is missed by both — it marks the sensitivity floor at this budget.
- **False alarms: 0/96 behavioural** (three strategies plus leave-one-out), **0/6** component
  leave-one-out and **0/18** component held-out honest.
- **Invalidation 9/9**, including the negative control: a change *outside* the certified scope leaves
  the certificate VALID.
- **The headline the system exists for:** without the component channel the backdoored artifact gates
  as `ALLOW_WITH_MONITORING`. With it, `QUARANTINE`.

Full numbers and how each must be stated: [`docs/RESULTS.md`](docs/RESULTS.md).

---

## How it works

Twelve stages, each a module you can open. The table in [`docs/MAP.md`](docs/MAP.md) §2 maps every
stage to its file; `run/1_audit.py --help` prints the same list.

The two things that make it more than a model-diffing script:

**1 · Transformation-aware calibration — no global thresholds.** Every statistic is a ratio against
the audited artifact's own level, tested against *honest derivations of its own declared transformation
family* with a t prediction interval (df = k−1). One honest derivation is one independent experimental
unit. A null fitted on int4 is not used for an int8 artifact: a mismatch names the differing field and
returns INCONCLUSIVE rather than a verdict. In deployment the null is built from the **organisation's
own pipeline**, so no vendor threshold is shipped at all.

**2 · Two independent channels.** Behavioural (paired logit divergence under budgeted search) and
component/provenance (which tensor groups changed, versus which ones the declared transformation ever
touches). They fail in different places, which is why the pair detects what neither does alone.

### Verdicts and actions

| verdict | action | exit |
|---|---|---|
| `ATTESTED` | ALLOW | 0 |
| `ATTESTED_WITH_DRIFT` | ALLOW_WITH_MONITORING | 0 |
| `SUSPICIOUS` | QUARANTINE | 10 |
| `BEHAVIORALLY_DIVERGENT` | BLOCK | 20 |
| `INCONCLUSIVE` | REQUIRE_MORE_EVIDENCE | 30 |
| `INVALIDATED` | REQUIRE_REATTESTATION | 40 |

`ATTESTED` never means *safe*. `INCONCLUSIVE` never means *safe*. The verdict is chosen by a
deterministic policy in `decision.py`; **no LLM is involved in deciding anything.**

---

## Limitations — read before quoting any result

- The trusted reference may itself be compromised; an honest derivation preserves that faithfully.
- Finite probing cannot establish universal safety, and a finite budget limits rare-trigger detection.
- Quantization here is **simulated** round-to-nearest. Which tensor groups an honest derivation skips
  is a property of that implementation, so the component null must be refitted per real toolchain.
- The component channel's honest null is **degenerate** (identically zero, no variance): it is reported
  as a component-level finding with the number of honest derivations stated, **never as a p-value**.
- It detects **provenance deviation, not malice** — a benign full fine-tune flags identically.
- It needs full weights on both sides, so it does not apply to adapter-only bundles.
- An attacker who confines changes to the groups the honest transformation already touches produces no
  component anomaly.
- Ratio-normalised behavioural statistics can be **self-masking**: a modification pervasive enough to
  raise its own baseline hides from them (see `docs/RESULTS.md` §2 — this happened, and is reported).
- An artifact with **no reference** — a model pretrained from scratch — cannot be attested by this
  method. The system returns INCONCLUSIVE; reference-free backdoor detection is an open problem.
- One model family at one size (135M), CPU only. No claim of generality across scales.

Two hypotheses were **falsified** and are kept in the repo with their code and write-ups rather than
deleted: reading the planted trigger token out of the embedding delta, and rescuing that ranking by
removing the dense component. The planned delta-guided search strategy was therefore **not built**.
See [`docs/REVIEW_v2_v3_and_fixes.md`](docs/REVIEW_v2_v3_and_fixes.md).

---

## CLI

```
backdoorlens keygen    --private k.pem --public k.hex
backdoorlens calibrate --reference REF --honest H1 H2 H3 H4 H5 H6 --out null.json --model-family F --budget 128
backdoorlens attest    --reference REF --derived D --null null.json --component-null C.json --out out/ --sign k.pem
backdoorlens verify    out/attestation.json --pin k.hex
backdoorlens gate      out/attestation.json D --pin k.hex        # exit 0/10/20/30/40 for CI
backdoorlens canary    out/attestation.json D
backdoorlens replay    out/attestation.json --reference REF --derived D
```

`--pin` takes the public key as hex or the path of the `.hex` file. Adapter-only artifacts add
`--adapter-base DIR`; policy overrides use `--policy @file.json`. (Inline JSON works too, but PowerShell
strips its quotes, which is why the file forms exist.) Full command list: [`docs/RUNBOOK.md`](docs/RUNBOOK.md) §9.

`attest` takes `--model-family` and `--budget` from the supplied null when you omit them, because
restating them is how an audit silently ends up INCONCLUSIVE over a typo. The gate **requires a
signature**: an unsigned certificate gates as `BAD_SIGNATURE → INVALIDATED → exit 40`, by design.

## Ethics

Only harmless canned-marker behaviours are planted, and `derive.py` refuses to continue if a planted
behaviour did not actually install — a tamper that does not exist cannot serve as ground truth.
Ground truth is written **outside** the audited bundle so it can never leak into one.
`trust_remote_code` is refused. Chat templates are rendered in a sandbox.
