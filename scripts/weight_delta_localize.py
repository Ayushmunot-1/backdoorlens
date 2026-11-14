"""STAGE 1 PREMISE TEST (no inference, no training, nothing decided): does a reference->derived weight
delta point at the RIGHT TOKENS?

Thin CLI over backdoorlens.weight_delta (see that module's docstring for the method, the statistics and
the stated scope limits). For each artifact bundle it prints where the parameter delta lives, and -- when
the delta touches a vocabulary-dimension tensor -- the top tokens by embedding-row delta, decoded.

WHAT IS BEING TESTED
The two planted-trigger artifacts went undetected in the real runs because the trigger lexeme
("lantern", "zq7x") is not in the fixed 128-prompt corpus and mutation does not reliably rediscover one
token out of ~49k within budget. But derive.plant_trigger_behavior optimizes m.parameters() -- all
parameters, embeddings included -- the trigger token appears in every trigger training prompt, and the
marker tokens are the training TARGET in every trigger example (SmolLM2 has tie_word_embeddings=True, so
the output rows ARE the embedding rows). Meanwhile derive.fake_quantize_state skips
("embed","lm_head","norm") and derive_canary_weight_delta skips embed/lm_head, so honest derivations and
canary perturbations should show NO embedding delta at all.

Prediction: trigger_easy / trigger_hard concentrate their embedding-row delta on few token ids, including
the trigger and/or marker tokens; honest_* show zero rows changed; canary_* show zero rows changed.
If the trigger tokens do NOT rank near the top, this idea is discarded here, cheaply -- that is the point
of running it before building anything on it.

GROUND TRUTH IS REPORT-ONLY. The ranked candidate list comes from weight_delta.localize(), which never
takes a token id as input. The known trigger/marker strings (read from <work>/_truth/<name>.json, which
derive.py deliberately writes OUTSIDE the audited bundle) are used solely to print where those tokens
landed, via the separate weight_delta.exact_ranks() reporting function.

--ref is whatever you want the delta measured AGAINST, and both choices are informative:
  * the original reference bundle -> delta = honest quantization error + any tamper
  * an HONEST derived artifact    -> delta = the tamper alone (cleanest isolation)

RUN (minutes; pure tensor arithmetic, memory-capped by row chunking, no GPU)
    python scripts\\weight_delta_localize.py --ref models\\SmolLM2-135M-Instruct --work results\\smollm2\\artifacts --out results\\weight_delta
    python scripts\\weight_delta_localize.py --ref models\\SmolLM2-135M-Instruct --work results\\smollm2_strong\\artifacts --out results\\weight_delta_strong

NOTE ON --work: `experiments.run` builds artifacts into `<--out>/artifacts` (experiments.py:
`work = out / "artifacts"`), and `derive.plant_trigger_behavior` writes ground truth to that
directory's `_truth/` subfolder. So for a run made with `--out results\\smollm2`, pass
`--work results\\smollm2\\artifacts`; `--truth` then defaults to the right place and needs no flag.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from backdoorlens import weight_delta as wd

SKIP_DIRS = {"_truth", "_cache"}


def _decode(tok, tid: int) -> tuple[str | None, str | None]:
    if tok is None:
        return None, None
    try:
        return tok.convert_ids_to_tokens(tid), tok.decode([tid])
    except Exception:
        return None, None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True,
                    help="bundle to measure the delta AGAINST (the reference, or an honest derived artifact)")
    ap.add_argument("--work", help="artifact work dir; every subdirectory holding .safetensors is analysed")
    ap.add_argument("--artifacts", nargs="*", default=[], help="explicit artifact bundle dirs")
    ap.add_argument("--truth", help="ground-truth dir (default <work>/_truth); REPORT-ONLY")
    ap.add_argument("--tokenizer", help="dir to load a tokenizer from for decoding (default --ref)")
    ap.add_argument("--top", type=int, default=20)
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
        sys.exit("no artifacts found: pass --work <dir> and/or --artifacts <dirs>")

    truth_dir = Path(a.truth) if a.truth else (Path(a.work) / "_truth" if a.work else None)
    tok = None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(Path(a.tokenizer) if a.tokenizer else ref))
    except Exception as e:
        print(f"[warn] no tokenizer ({type(e).__name__}: {e}); token ids will not be decoded")

    fw = wd.framework()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"reference : {ref}")
    print(f"artifacts : {len(targets)}    safetensors framework: {fw}    truth dir: {truth_dir}\n")

    results, summary = {}, []
    for d in targets:
        print(f"=== {d.name} ===")
        try:
            r = wd.localize(ref, d, top=a.top, fw=fw)
        except Exception as e:
            r = {"status": "error", "error": f"{type(e).__name__}: {e}"}
        results[d.name] = r

        if r["status"] != "ok":
            print(f"  {r['status']}: {r.get('reason') or r.get('error')}\n")
            summary.append({"artifact": d.name, "status": r["status"]})
            continue

        print(f"  tensors shared={r['n_tensors_shared']}  changed={r['n_tensors_changed']}")
        for t in r["top_changed_tensors"][:5]:
            print(f"    {t['tensor']:<44s} rel_fro={t['rel_fro_delta']:.3e}  "
                  f"frac_changed={t['frac_entries_changed']:.3f}")
        tl = r.get("token_localization") or {}
        if tl.get("status") != "ok":
            print(f"  token localization: {tl.get('status')}\n")
            summary.append({"artifact": d.name, "status": tl.get("status")})
            continue

        print(f"  vocab tensor: {tl['tensor']}  rows={tl['n_rows']}  "
              f"rows_changed={tl['n_rows_changed']} ({100 * tl['frac_rows_changed']:.3f}%)")
        if tl["n_rows_changed"] == 0:
            print(f"  -> {tl['note']}\n")
            summary.append({"artifact": d.name, "rows_changed": 0,
                            "trigger_best_rank": None, "marker_best_rank": None})
            continue

        print(f"  concentration: max/median_nonzero={tl['top1_over_median_nonzero']}  "
              f"top{tl['topk_mass_k']}_mass_fraction={tl['topk_mass_fraction']}")
        print(f"  top tokens by embedding-row delta:")
        for row in tl["top_rows"]:
            piece, dec = _decode(tok, row["token_id"])
            print(f"    #{row['rank']:<3d} id={row['token_id']:<7d} delta={row['row_delta']:.4e}  "
                  f"piece={piece!r}  decoded={dec!r}")

        trig_rank = mark_rank = None
        tfile = (truth_dir / f"{d.name}.json") if truth_dir else None
        if tfile and tfile.exists() and tok is not None:
            try:
                truth = json.loads(tfile.read_text())
            except Exception:
                truth = {}
            gt = {}
            for field in ("trigger", "marker"):
                s = truth.get(field)
                if not isinstance(s, str) or not s:
                    continue
                ids = [int(i) for i in tok(s, add_special_tokens=False)["input_ids"]]
                er = wd.exact_ranks(ref, d, ids, tensor_key=tl["tensor"], fw=fw)
                ranks = [er.get("ranks", {}).get(i) for i in ids]
                gt[field] = {"string": s, "token_ids": ids, "ranks": ranks,
                             "row_deltas": [er.get("row_deltas", {}).get(i) for i in ids],
                             "best_rank": min([x for x in ranks if x is not None], default=None)}
            if gt:
                print("  GROUND TRUTH (printed for evaluation only; never an input to the ranking):")
                for field, g in gt.items():
                    print(f"    {field}={g['string']!r} ids={g['token_ids']} ranks={g['ranks']} "
                          f"best_rank={g['best_rank']} of {tl['n_rows']}")
                trig_rank = (gt.get("trigger") or {}).get("best_rank")
                mark_rank = (gt.get("marker") or {}).get("best_rank")
            results[d.name]["ground_truth_report"] = gt
        print()
        summary.append({"artifact": d.name, "rows_changed": tl["n_rows_changed"],
                        "top1_over_median_nonzero": tl["top1_over_median_nonzero"],
                        "trigger_best_rank": trig_rank, "marker_best_rank": mark_rank})

    (out / "weight_delta_localization.json").write_text(
        json.dumps({"reference": str(ref), "framework": fw, "results": results}, indent=1, default=str))

    print("=== SUMMARY: does the weight delta point at the right tokens? ===")
    for s in summary:
        print("  " + json.dumps(s, default=str))
    print(f"\nwrote {out / 'weight_delta_localization.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
