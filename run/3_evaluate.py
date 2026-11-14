"""Merge the two evidence sources into ONE evaluation view: what does BackdoorLens detect, at what
false-alarm rate, versus what conventional pre-deployment checks detect.

WHY THIS EXISTS
The evidence lives in two places that have never been joined:
  * `results.json` from `run_experiments.py` -- the 7-stage experiment: E1 conventional-check
    baselines, 133 audits across 3 search strategies x 3 seeds, honest false-alarm rate, per-artifact
    tamper detection, and component-change invalidation. This predates the component channel.
  * `component_null_audit.json` from `component_null_audit.py` -- the intra-weight component channel,
    added afterwards.
So no single file states what the system detects. This script produces that statement.

NAMING: `results.json` keys artifacts by experiment name (`template_inject`), while the artifact
directories audited by the component channel use a `t_`-prefixed name (`t_template`). The mapping is
read off `experiments.py` (lines ~319-325) and declared in TAMPER_DIR below, not guessed.

HONESTY NOTE ON THE CONVENTIONAL BASELINE
`conventional_checks` reports four checks. Three (hash/signature, serialization scan, template static
scan) pass on every artifact including the tampered ones. The fourth (relative perplexity change) is
the only one that reacts -- but it also fires on honest int4 derivations, so a "flag" from it is not
evidence of tampering. This script therefore prints the SEPARABILITY test: the worst honest value
against the lowest tampered value. If the honest maximum exceeds the tampered minimum, the two classes
overlap and no threshold can separate them -- which must be reported rather than quietly choosing a
threshold that happens to work on this data.

RUN
    python run/3_evaluate.py --results results/smollm2/results.json \
        --component results/component_null_int4/component_null_audit.json \
        --out evidence/merged_evaluation
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# experiment-name -> artifact directory name (experiments.py: tamper["template_inject"] =
# B.template_inject("t_template", ...), tamper[f"canary_{sc}"] = B.canary(f"t_canary_{sc}", ...), etc.)
TAMPER_DIR = {
    "template_inject": "t_template",
    "generation_edit": "t_gen",
    "trigger_easy": "t_trigger_easy",
    "trigger_hard": "t_trigger_hard",
    "canary_0.5": "t_canary_0.5",
    "canary_2.0": "t_canary_2.0",
    "canary_8.0": "t_canary_8.0",
}
STRATEGIES = ("guided", "random", "benchmark")


def _conv_flags(c: dict) -> dict:
    """Which conventional checks flag this artifact. Returns per-check PASS/FLAG."""
    return {
        "hash": "PASS" if str(c.get("integrity_hash_signature", "")).startswith("PASS") else "FLAG",
        "serialization": "PASS" if str(c.get("serialization_scan")) == "PASS" else "FLAG",
        "template_scan": "PASS" if str(c.get("template_static_scan")) == "PASS" else "FLAG",
        "perplexity": str(c.get("perplexity_check")),
        "ppl_rel": c.get("perplexity_relative_change"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, help="results.json from run_experiments.py")
    ap.add_argument("--component", help="component_null_audit.json (optional; omit to show the "
                                        "behavioural-only evaluation)")
    ap.add_argument("--out", help="directory to write merged_evaluation.json into")
    a = ap.parse_args()

    R = json.loads(Path(a.results).read_text())
    e1 = R.get("E1_conventional", {})
    det = R.get("summary", {}).get("tamper_detection", {})
    fa = R.get("summary", {}).get("honest_false_alarm", {})
    loo = R.get("LOO_calibration_false_alarm", {})

    comp, comp_src = {}, None
    if a.component:
        C = json.loads(Path(a.component).read_text())
        comp_src = C
        comp = C.get("audits", {})

    # ---------------- separability of the only conventional check that reacts ----------------
    hon = [(k, v["perplexity_relative_change"]) for k, v in e1.items() if v.get("class") == "honest"]
    tam = [(k, v["perplexity_relative_change"]) for k, v in e1.items() if v.get("class") == "tampered"]
    hon4 = [(k, x) for k, x in hon if k.startswith("test4")]      # same bit-width as the tampered set
    print("=" * 96)
    print("CONVENTIONAL BASELINE — can relative perplexity separate honest from tampered?")
    print("=" * 96)
    if hon4 and tam:
        hmax_k, hmax = max(hon4, key=lambda t: t[1])
        tmin_k, tmin = min(tam, key=lambda t: t[1])
        print(f"  worst honest int4 : {hmax_k:<18} {hmax:.5f}")
        print(f"  lowest tampered   : {tmin_k:<18} {tmin:.5f}")
        if hmax > tmin:
            print(f"  -> OVERLAP: honest max ({hmax:.5f}) EXCEEDS tampered min ({tmin:.5f}).")
            print("     No perplexity threshold passes every honest derivation and catches every tamper.")
            print(f"     A threshold catching the triggers (>{0.15:.2f}) misses {tmin_k}; one catching")
            print("     that one flags honest quantizations. The check carries no usable signal here.")
        else:
            print(f"  -> separable on THIS sample by a threshold in ({tmin:.5f}, {hmax:.5f}] — but note")
            print("     such a threshold would be fitted to these artifacts, with no calibration behind it.")
    print(f"\n  (the other three conventional checks pass on ALL {len(e1)} artifacts, tampered included;")
    print("   note in particular that the template static scan passes `template_inject`)")

    # ---------------- per-artifact detection ----------------
    print()
    print("=" * 96)
    print("DETECTION, PER ARTIFACT")
    print("=" * 96)
    print(f"{'artifact':<17}{'conv.flags':<12}{'guided':>8}{'random':>8}{'bench':>8}"
          f"{'component':>11}  {'COMBINED':<9}")
    print("-" * 96)
    rows = []
    for name in TAMPER_DIR:
        c = e1.get(name, {})
        f = _conv_flags(c) if c else {}
        conv_flag = [k for k in ("hash", "serialization", "template_scan") if f.get(k) == "FLAG"]
        conv_ppl = f.get("perplexity") == "FAIL"
        conv_txt = ("ppl-only" if conv_ppl and not conv_flag else
                    ("+".join(conv_flag) if conv_flag else "none"))
        d = det.get(name, {})
        beh = {s: d.get(s, {}).get("detect_rate") for s in STRATEGIES}
        beh_any = any((beh[s] or 0) > 0 for s in STRATEGIES)
        cdir = TAMPER_DIR[name]
        ca = comp.get(cdir, {})
        n_hard = ca.get("n_hard_anomalies") if ca.get("status") == "ok" else None
        comp_hit = bool(n_hard)
        combined = "DETECTED" if (beh_any or comp_hit) else "missed"
        rows.append({"artifact": name, "dir": cdir, "conventional": f, "behavioural": beh,
                     "component_hard_anomalies": n_hard, "combined_detected": combined == "DETECTED"})
        print(f"{name:<17}{conv_txt:<12}"
              f"{(f'{beh['guided']:.2f}' if beh['guided'] is not None else '-'):>8}"
              f"{(f'{beh['random']:.2f}' if beh['random'] is not None else '-'):>8}"
              f"{(f'{beh['benchmark']:.2f}' if beh['benchmark'] is not None else '-'):>8}"
              f"{(str(n_hard) if n_hard is not None else '-'):>11}  {combined:<9}")

    n_tot = len(rows)
    n_beh = sum(1 for r in rows if any((v or 0) > 0 for v in r["behavioural"].values()))
    n_comb = sum(1 for r in rows if r["combined_detected"])
    print("-" * 96)
    print(f"  behavioural channel alone : {n_beh}/{n_tot}")
    if comp:
        print(f"  + component channel       : {n_comb}/{n_tot}")
        missed = [r["artifact"] for r in rows if not r["combined_detected"]]
        if missed:
            print(f"  still missed              : {', '.join(missed)}")

    # ---------------- false alarms ----------------
    print()
    print("=" * 96)
    print("FALSE ALARMS ON HONEST DERIVATIONS")
    print("=" * 96)
    tot_n = tot_fp = 0
    for s in STRATEGIES:
        v = fa.get(s)
        if v:
            n, fpr = int(v["n"]), float(v["fpr"])
            tot_n += n
            tot_fp += round(fpr * n)
            print(f"  behavioural / {s:<10} {int(round(fpr*n))}/{n}   (95% CI upper {v['ci'][1]:.3f})")
    for bits, blk in (loo or {}).items():
        r = blk.get("rows", [])
        if r:
            k = sum(1 for x in r if x.get("detected"))
            tot_n += len(r)
            tot_fp += k
            print(f"  leave-one-out / {bits:<8} {k}/{len(r)}")
    print(f"  {'TOTAL (behavioural)':<25} {tot_fp}/{tot_n}")
    if comp_src:
        l = comp_src.get("leave_one_out", {})
        hon_dirs = [d for d in comp.keys() if d.startswith(("cal", "test"))]
        hon_hits = sum(1 for d in hon_dirs if (comp[d].get("n_hard_anomalies") or 0) > 0)
        print(f"  component / leave-one-out {l.get('n_false_positives')}/{l.get('n_tested')}")
        print(f"  component / held-out honest {hon_hits}/{len(hon_dirs)}")
        print(f"  {'component null fitted on':<25} "
              f"{comp_src.get('component_null',{}).get('n_honest_derivations')} honest derivations")

    # ---------------- invalidation ----------------
    inv = R.get("E8_invalidation", {})
    if inv:
        cases = inv.get("cases", [])
        ok = sum(1 for c in cases if (c.get("state") == "INVALIDATED") == bool(c.get("expected_invalidated")))
        lat = [c.get("latency_s") for c in cases if c.get("latency_s") is not None]
        print()
        print("=" * 96)
        print("CERTIFICATE INVALIDATION")
        print("=" * 96)
        print(f"  baseline state: {inv.get('baseline_state')}")
        msg = f"  component changes behaving as expected: {ok}/{len(cases)}"
        if lat:
            msg += f"   median latency {sorted(lat)[len(lat) // 2]:.2f}s"
        print(msg)
        for c in cases:
            print(f"    {str(c.get('component')):<14} -> {c.get('state')}")

    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        payload = {"source_results": str(a.results), "source_component": a.component,
                   "per_artifact": rows,
                   "behavioural_detected": f"{n_beh}/{n_tot}",
                   "combined_detected": f"{n_comb}/{n_tot}",
                   "behavioural_false_alarms": f"{tot_fp}/{tot_n}",
                   "conventional_separability": {
                       "honest_int4_max": (max(hon4, key=lambda t: t[1]) if hon4 else None),
                       "tampered_min": (min(tam, key=lambda t: t[1]) if tam else None)}}
        (out / "merged_evaluation.json").write_text(json.dumps(payload, indent=1, default=str))
        print(f"\nwrote {out / 'merged_evaluation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
