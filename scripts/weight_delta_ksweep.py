"""STAGE 2a: does removing the dense tied-output component sharpen the token ranking?

BACKGROUND (measured, from scripts/weight_delta_localize.py on the real SmolLM2 runs)
Two findings came out of Stage 1:
  A. A PERFECT component-level separator, replicated across both the weak-plant and strong-plant
     runs: every honest derivation of the declared family (12 calibration + 12 held-out), all three
     canary perturbations, the template tamper and the generation-config tamper changed 210 tensors
     and ZERO embedding rows; both planted-trigger artifacts changed 272 tensors and 100% of the
     49152 embedding rows. 24 honest derivations, no false positives.
  B. A raw embedding-row-delta ranking puts the trigger token at rank 17-681 of 49152 -- real signal
     (top 0.03-1.4%), but not a usable top-20 seed list.

The cause of B is diagnosable rather than mysterious: SmolLM2 sets `tie_word_embeddings: True`, so
the embedding matrix IS the output head and the full-vocabulary softmax puts nonzero gradient on
EVERY row at every step -- which is exactly why 100% of rows moved instead of a handful. For a
non-target token that update is ~ -lr * p_t * h_bar: a SHARED direction, rank-1 in expectation,
scaled by the token's probability. The observed raw top-20 confirms it (frequent sentence-initial
tokens in the weak run; an almost pure cluster of third-person-singular verbs in the strong run --
an output-distribution artifact, not the trigger). The trigger's input-side update does not share
that direction.

WHAT THIS SCRIPT TESTS
Project out the k leading principal directions of the delta matrix and rank by the residual row
norm. If the diagnosis is right, the trigger's rank should improve sharply at small k.

ANTI-P-HACKING DISCIPLINE (important -- read before interpreting the output)
k is NEVER chosen by seeing which value ranks the known trigger best. Two ground-truth-free rules
are computed from the delta's own spectrum and printed FIRST, before any trigger rank:
    k_theory   = 1       (the dense tied-output term is rank-1 in expectation; theory, not data)
    k_by_energy          (smallest k explaining >= 90% of total delta energy)
The full sweep is then printed as a SENSITIVITY ANALYSIS with the pre-declared k marked. The
trigger's rank is an OUTCOME; it never selects anything. `weight_delta.localize_residual` has no
token-id parameter at all, so the ranking cannot be steered by ground truth.

RUN (minutes; one Gram matrix + one pass per k over a ~113 MB tensor, no inference, no GPU)
    python scripts\\weight_delta_ksweep.py --ref models\\SmolLM2-135M-Instruct --work results\\smollm2\\artifacts --out results\\weight_delta
    python scripts\\weight_delta_ksweep.py --ref models\\SmolLM2-135M-Instruct --work results\\smollm2_strong\\artifacts --out results\\weight_delta_strong

By default only artifacts whose vocab tensor actually changed are swept (the honest ones have an
identical embedding matrix, so there is nothing to decompose); pass --all to force every artifact.
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
    ap.add_argument("--work", help="artifact dir (each subdirectory holding .safetensors)")
    ap.add_argument("--artifacts", nargs="*", default=[])
    ap.add_argument("--truth", help="ground-truth dir (default <work>/_truth); REPORT-ONLY")
    ap.add_argument("--tokenizer", help="default: --ref")
    ap.add_argument("--ks", type=int, nargs="*", default=[0, 1, 2, 4, 8, 16, 32])
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--all", action="store_true", help="sweep every artifact, not just changed-vocab ones")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ref = Path(a.ref)
    if not ref.is_dir():
        sys.exit(f"--ref {ref} is not a directory")
    targets = [Path(p) for p in a.artifacts]
    if a.work:
        w = Path(a.work)
        if not w.is_dir():
            sys.exit(f"--work {w} is not a directory")
        targets += [d for d in sorted(w.iterdir())
                    if d.is_dir() and d.name not in SKIP_DIRS and any(d.glob("*.safetensors"))]
    if not targets:
        sys.exit("no artifacts found")

    truth_dir = Path(a.truth) if a.truth else (Path(a.work) / "_truth" if a.work else None)
    tok = None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(Path(a.tokenizer) if a.tokenizer else ref))
    except Exception as e:
        print(f"[warn] no tokenizer ({type(e).__name__}: {e}); tokens will not be decoded")

    fw = wd.framework()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"reference: {ref}    framework: {fw}    ks={a.ks}")
    print(f"pre-declared k rules: k_theory={wd.K_THEORY} (rank-1 dense term, theory) and "
          f"k_by_energy (>=90% of delta energy, spectrum only)\n")

    results, rows = {}, []
    for d in targets:
        res = wd.localize_residual(ref, d, ks=tuple(a.ks), top=a.top, fw=fw)
        if res.get("status") != "ok" or res.get("no_delta"):
            if a.all:
                print(f"=== {d.name} === {res.get('status')}"
                      f"{' (vocab tensor identical; nothing to decompose)' if res.get('no_delta') else ''}\n")
            continue

        print(f"=== {d.name} ===")
        print(f"  vocab tensor: {res['tensor']}")
        print(f"  delta energy by leading direction: "
              f"{[round(x, 4) for x in res['eigval_energy_fractions']]}")
        print(f"  PRE-DECLARED: k_theory={res['k_theory']}   "
              f"k_by_energy({res['energy_threshold']})={res['k_by_energy']}")

        gt_ids = {}
        tfile = (truth_dir / f"{d.name}.json") if truth_dir else None
        if tfile and tfile.exists() and tok is not None:
            try:
                truth = json.loads(tfile.read_text())
            except Exception:
                truth = {}
            for field in ("trigger", "marker"):
                s = truth.get(field)
                if isinstance(s, str) and s:
                    gt_ids[field] = {"string": s,
                                     "ids": [int(i) for i in tok(s, add_special_tokens=False)["input_ids"]]}

        print(f"  sensitivity sweep (trigger rank is an OUTCOME here, never a selector):")
        print("    k    energy_removed   top1/med   "
              + "   ".join(f"{f}_best_rank" for f in gt_ids))
        per_k = {}
        for k in a.ks:
            bk = res["by_k"][int(k)]
            cells = []
            for field, g in gt_ids.items():
                rk = wd.ranks_in_residual(res, k, g["ids"])
                best = min([v for v in rk.values() if v is not None], default=None)
                cells.append(best)
                g.setdefault("ranks_by_k", {})[int(k)] = {str(i): rk.get(i) for i in g["ids"]}
                g.setdefault("best_rank_by_k", {})[int(k)] = best
            per_k[int(k)] = {"energy_removed": bk["energy_removed_fraction"],
                             "top1_over_median_nonzero": bk["top1_over_median_nonzero"],
                             "best_ranks": {f: c for f, c in zip(gt_ids, cells)}}
            mark = ""
            if k == res["k_theory"]:
                mark += "  <- k_theory"
            if k == res["k_by_energy"]:
                mark += "  <- k_by_energy"
            t1 = bk["top1_over_median_nonzero"]
            print(f"    {k:<4d} {bk['energy_removed_fraction']:<16.4f} "
                  f"{(f'{t1:.2f}' if t1 is not None else 'n/a'):<10s} "
                  + "   ".join(f"{c if c is not None else '-':<14}" for c in cells) + mark)

        kk = res["k_theory"]
        print(f"  top {a.top} tokens at the pre-declared k_theory={kk}:")
        for row in res["by_k"][int(kk)]["top_rows"]:
            piece = dec = None
            if tok is not None:
                try:
                    piece, dec = tok.convert_ids_to_tokens(row["token_id"]), tok.decode([row["token_id"]])
                except Exception:
                    pass
            print(f"    #{row['rank']:<3d} id={row['token_id']:<7d} residual={row['row_delta']:.4e}  "
                  f"piece={piece!r}  decoded={dec!r}")
        if gt_ids:
            print("  GROUND TRUTH (evaluation only):")
            for field, g in gt_ids.items():
                print(f"    {field}={g['string']!r} ids={g['ids']}  "
                      f"best_rank_by_k={g.get('best_rank_by_k')}")
        print()

        results[d.name] = {**wd.strip_internal(res), "ground_truth_report": gt_ids}
        rows.append({"artifact": d.name, "k_theory": res["k_theory"], "k_by_energy": res["k_by_energy"],
                     "per_k": per_k})

    (out / "weight_delta_ksweep.json").write_text(
        json.dumps({"reference": str(ref), "ks": a.ks, "results": results}, indent=1, default=str))

    print("=== SUMMARY ===")
    for r in rows:
        print("  " + json.dumps(r, default=str))
    if not rows:
        print("  (no artifact had a changed vocab-dimension tensor; nothing to decompose)")
    print(f"\nwrote {out / 'weight_delta_ksweep.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
