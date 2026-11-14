"""Calibrate the intra-weight COMPONENT null from honest derivations, then audit every artifact --
with a leave-one-out false-positive check on the honest population.

WHY THIS AND NOT THE TOKEN RANKING
Two hypotheses about reading a planted trigger's VOCABULARY off the weight delta were tested and both
failed on real data: the raw embedding-row ranking put the trigger at rank 17-681 of 49152, and
projecting out the leading principal directions (pre-declared k_theory=1) did not help -- it made two
of four cases worse, and the delta spectrum turned out diffuse (leading direction 0.47-0.58 of energy,
k_by_energy 32-58), falsifying the rank-1 premise. Those are reported as negative results in
docs/REVIEW_v2_v3_and_fixes.md; scripts/weight_delta_ksweep.py reproduces them.

What DID survive is component-level, and it separated perfectly across both real runs: every honest
derivation of the declared family (12 calibration + 12 held-out), all three canary perturbations, the
template tamper and the generation-config tamper changed 210 tensors and ZERO vocabulary rows, while
both planted-trigger artifacts changed 272 tensors and 100% of the 49152 vocabulary rows -- i.e. they
modify tensor groups (embeddings, norms) that NO honest derivation of the same declared family
touches. This script turns that observation into a calibrated statistic with a real FPR check.

HOW IT IS CALIBRATED (same discipline as the behavioral null)
  - One honest derivation = one independent experimental unit. Never one tensor.
  - The null records, per tensor key, how many honest members modify it and the honest range of the
    relative Frobenius delta. A "hard anomaly" is a key modified by the audited artifact that NO
    honest member modifies.
  - The honest null for such a key is DEGENERATE (identically zero): no variance, so no p-value is
    computed and none is claimed. It is reported as a component-level finding backed by N honest
    derivations. The paper must say it that way.
  - False positives are measured by LEAVE-ONE-OUT over the honest population (fit on k-1, audit the
    held-out one), mirroring the existing behavioral leave-one-out safety net.
  - The change tolerance is NOT a hand-picked constant: it defaults to 0.0 (any nonzero change) and
    the honest per-key range printed here is what should set it if a pipeline's honest transformation
    perturbs everything slightly.

No inference, no training, no GPU: this only reads tensor bytes and computes norms.

RUN
    python scripts\\component_null_audit.py --ref models\\SmolLM2-135M-Instruct --work results\\smollm2\\artifacts --honest-prefix cal --out results\\component_null
    python scripts\\component_null_audit.py --ref models\\SmolLM2-135M-Instruct --work results\\smollm2_strong\\artifacts --honest-prefix cal --out results\\component_null_strong

--honest-prefix selects which subdirectories form the calibration population (default "cal", i.e.
cal4_*/cal8_*). Everything else in --work is then audited against that null. Use --honest-prefix cal4
to calibrate against a single transformation family (int4 only) -- which is the stricter, more
correct thing to do if you want the null matched to one declared family rather than two.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from backdoorlens import weight_delta as wd

SKIP_DIRS = {"_truth", "_cache"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--honest-prefix", default="cal",
                    help="subdirectory name prefix forming the honest calibration population (default 'cal')")
    ap.add_argument("--tol", type=float, default=0.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ref, work = Path(a.ref), Path(a.work)
    if not ref.is_dir():
        sys.exit(f"--ref {ref} is not a directory")
    if not work.is_dir():
        sys.exit(f"--work {work} is not a directory (experiments.run writes artifacts to <out>/artifacts)")

    bundles = {d.name: d for d in sorted(work.iterdir())
               if d.is_dir() and d.name not in SKIP_DIRS and any(d.glob("*.safetensors"))}
    honest = {n: d for n, d in bundles.items() if n.startswith(a.honest_prefix)}
    others = {n: d for n, d in bundles.items() if n not in honest}
    if len(honest) < 2:
        sys.exit(f"need >= 2 honest derivations matching prefix {a.honest_prefix!r}; found {len(honest)}")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fw = wd.framework()
    print(f"reference: {ref}    framework: {fw}    tol={a.tol}")
    print(f"honest calibration population ({len(honest)}): {sorted(honest)}")
    print(f"artifacts to audit ({len(others)}): {sorted(others)}\n")

    print("[1/3] fitting the component null from the honest derivations")
    null = wd.fit_component_null(ref, honest, a.tol, fw, log=print)
    if null["status"] != "ok":
        sys.exit(f"could not fit a component null: {null['status']}")
    print(f"   n_honest_derivations={null['n_honest_derivations']}  "
          f"n_tensors_changed_honest={null['n_tensors_changed_honest']}")
    print(f"   tensor groups NO honest derivation modifies: {null['n_never_changed_keys']}")
    for k in null["never_changed_keys"][:6]:
        print(f"     {k}")
    if null["n_never_changed_keys"] > 6:
        print(f"     ... and {null['n_never_changed_keys'] - 6} more")

    print("\n[2/3] leave-one-out false-positive check on the honest population")
    loo = wd.leave_one_out_component_fpr(ref, honest, a.tol, fw, log=print)
    print(f"   tested={loo['n_tested']}  false_positives={loo['n_false_positives']}  "
          f"FPR={loo['false_positive_rate']}")

    print("\n[3/3] auditing every other artifact against that null")
    rows, results = [], {}
    for name, d in others.items():
        an = wd.component_anomalies(ref, d, null, fw)
        results[name] = an
        if an.get("status") != "ok":
            print(f"   {name:<18s} {an.get('status')}")
            rows.append({"artifact": name, "status": an.get("status")})
            continue
        flag = "ANOMALY" if an["n_hard_anomalies"] else "-"
        print(f"   {name:<18s} n_tensors_changed={an['n_tensors_changed']:<5d} "
              f"honest_range={an['n_tensors_changed_honest_range']}  "
              f"hard_anomalies={an['n_hard_anomalies']:<4d} {flag}")
        rows.append({"artifact": name, "n_tensors_changed": an["n_tensors_changed"],
                     "n_hard_anomalies": an["n_hard_anomalies"],
                     "count_outside_honest_range": an["count_outside_honest_range"],
                     "example_anomalies": an["hard_anomalies"][:4]})

    payload = {"reference": str(ref), "work": str(work), "tol": a.tol, "framework": fw,
               "honest_members": null["members"], "component_null": null,
               "leave_one_out": loo, "audits": results}
    (out / "component_null_audit.json").write_text(json.dumps(payload, indent=1, default=str))

    print("\n=== SUMMARY ===")
    print(f"  honest derivations in null: {null['n_honest_derivations']}")
    print(f"  leave-one-out false positives: {loo['n_false_positives']}/{loo['n_tested']} "
          f"(FPR={loo['false_positive_rate']})")
    for r in rows:
        print("  " + json.dumps(r, default=str))
    print(f"\nwrote {out / 'component_null_audit.json'}")
    print("\nReminder for the write-up: a hard anomaly means 'this artifact modifies a tensor group "
          "that none of the N honest derivations of this declared family modifies'. The honest null "
          "for those groups is identically zero, so report it as a component-level finding with N "
          "stated -- not as a p-value, and not as proof of a backdoor.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
