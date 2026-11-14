"""Audit ONE derived artifact against the reference and write a signed certificate.

    python run/1_audit.py --derived results/smollm2/artifacts/t_trigger_easy

This is the command to run when someone says "show me the exact run". It is a thin wrapper over
`backdoorlens.cli attest` with the SmolLM2 paths and both nulls filled in, because the full CLI form
takes eight flags and getting one wrong (a model-family typo, a budget that disagrees with the null)
silently produces INCONCLUSIVE rather than an error. The flags are still there if you want them.

WHAT IT DOES. Twelve stages, in the order `audit.run_audit` actually performs them -- the same list
as docs/MAP.md section 2. Open audit.py and follow it top to bottom to check.

   1  manifest both bundles        scope.build_manifest       hashes every in-scope component and
                                                              extracts tensor/parameter metadata
   2  artifact differential        bundle_diff.diff_bundles   tokenizer, template, configs -- and the
                                                              template RENDERED on a fixed corpus,
                                                              which is how template_inject is caught
   3  comparability                scope.comparability        are these a reference/derived pair at
                                                              all? REFUSES HERE if not, so no model
                                                              is ever loaded for an out-of-scope pair
   4  load both models             backends.hf                the only expensive step
   5  probe suite                  probes.build_seed_corpus   128 seed prompts + semantic variants,
                                                              from the seed the null was fitted with
   6  paired measurement           audit.stage_a              every probe through BOTH models: JSD,
                                                              KL, top-k overlap and rank correlation,
                                                              logit margin, NLL shift, argmax flip
   7  calibrate                    calibration, stats         computed inside stage 6 against the
                                                              supplied null: is this level abnormal
                                                              for HONEST derivations of this family?
   8  budgeted search              search.run_search          hunt for a worse prompt at fixed budget
   9  confirm                      audit.confirm              semantic variants + replay through a
                                                              SEPARATE call path
  10  component differential       weight_delta               which tensor groups changed that no
                                                              honest derivation of this family touches
  11  decide                       decision.decide            deterministic policy. NO LLM.
  12  sign                         attestation.build, .sign   ed25519 over the canonical payload

The GATE is not part of the audit -- `monitor.py`, run as `backdoorlens gate` or demo beat 5. It
reloads no model: it re-hashes the artifact and compares against the certificate.

Exit code is the CI gate code: 0 allow, 10 quarantine, 20 block, 30 inconclusive, 40 re-attest.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

EXITS = {"ALLOW": 0, "ALLOW_WITH_MONITORING": 0, "QUARANTINE": 10, "BLOCK": 20,
         "REQUIRE_MORE_EVIDENCE": 30, "REQUIRE_REATTESTATION": 40}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--derived", required=True, help="the artifact to audit")
    ap.add_argument("--reference", default="models/SmolLM2-135M-Instruct")
    ap.add_argument("--null", default="results/smollm2/null_int4.json",
                    help="behavioural null. Family and budget are read FROM it, not restated.")
    ap.add_argument("--component-null", default="results/component_null_int4/component_null_audit.json",
                    help="component null, matched to one declared family (int4). Pass \"\" to omit.")
    ap.add_argument("--out", help="output dir (default results/audit/<artifact name>)")
    ap.add_argument("--sign", help="ed25519 private key. Omit for an unsigned certificate — note the "
                                   "gate requires a signature, so an unsigned one gates as exit 40.")
    ap.add_argument("--backend", default="hf")
    a = ap.parse_args()

    out = Path(a.out) if a.out else ROOT / "results" / "audit" / Path(a.derived).name
    missing = [p for p in (a.reference, a.derived) if not Path(p).is_dir()]
    missing += [p for p in (a.null, a.component_null) if p and not Path(p).is_file()]
    if missing:
        print("MISSING INPUTS:", *(f"\n  {m}" for m in missing))
        return 2

    cmd = [sys.executable, "-m", "backdoorlens.cli", "attest", "--backend", a.backend,
           "--reference", a.reference, "--derived", a.derived, "--null", a.null,
           "--out", str(out)]
    if a.component_null:
        cmd += ["--component-null", a.component_null]
    if a.sign:
        cmd += ["--sign", a.sign]

    print(f"auditing {Path(a.derived).name} against {Path(a.reference).name}")
    print("this loads both models and runs 128 probes + a 128-budget search: expect 3-5 min on CPU\n")
    env = {**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")}
    r = subprocess.run(cmd, cwd=ROOT, env=env)
    if r.returncode != 0:
        return r.returncode

    import json
    att = json.loads((out / "attestation.json").read_text())
    d = att["decision"]
    code = EXITS.get(d["action"], 1)
    print(f"\n  verdict {d['verdict']} -> {d['action']}   (CI exit code {code})")
    print(f"  signed  {'yes' if att.get('signature') else 'NO (gate will report BAD_SIGNATURE)'}")
    print(f"  written {out / 'attestation.json'}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
