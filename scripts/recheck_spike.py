"""Cheap re-check of the spike-ratio fix against artifacts and a null ALREADY built by Stage 3 -
no rebuilding, no re-fitting the null, no new fine-tuning. Re-runs run_audit() (stage A + search +
confirm, same cost per call as the original run - a few seconds to ~1 min each on this machine) with
the updated audit.py/decision.py for just the artifacts that were missed before, so you get an answer
in a few minutes instead of re-running the whole 0.8h pipeline.

    python scripts\\recheck_spike.py --config configs\\experiments\\smollm2_135m_final.json --out results\\smollm2

Prints OLD (level-only) vs NEW (level + spike) verdict for each missed artifact x strategy x seed.
"""
import argparse, copy, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from backdoorlens.audit import AuditConfig, run_audit
from backdoorlens.calibration import NullDistribution
from backdoorlens.decision import Policy

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--artifacts", nargs="*", default=["t_trigger_easy", "t_trigger_hard", "t_canary_0.5"],
                help="artifact directory names under <out>/artifacts (defaults to the 3 artifacts missed in the Stage 3 run)")
ap.add_argument("--strategies", nargs="*", default=["guided"])
ap.add_argument("--seeds", nargs="*", type=int, default=[1000, 1001, 1002])
a = ap.parse_args()

cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
out = Path(a.out)
ref_dir = Path(cfg["reference_dir"])
pol = Policy(**cfg.get("policy", {}))
acfg = AuditConfig(backend=cfg["backend"], budget=cfg["budget"], n_seeds=cfg.get("n_seeds", 128),
                   model_family=cfg["model_family"], gen_n=cfg.get("gen_n", 24),
                   max_new_tokens=cfg.get("max_new_tokens", 16), backend_kwargs=cfg.get("backend_kwargs", {}))
null = NullDistribution.load(out / "null_int4.json")   # all tampered artifacts in this config are INT4

print(f"null: {null.n_derivations()} honest derivations, honest spike ratios = "
      f"{[round(float(x), 3) for x in null.spikes()]}")
print(f"policy.min_level_ratio={pol.min_level_ratio}  policy.min_spike_ratio={pol.min_spike_ratio}  "
      "(spike guard uses the SAME default as the pre-existing level guard - not tuned to these artifacts)\n")

rows = []
for name in a.artifacts:
    art_dir = out / "artifacts" / name
    if not art_dir.is_dir():
        print(f"SKIP {name}: not found at {art_dir}")
        continue
    for strat in a.strategies:
        for seed in a.seeds:
            c = copy.deepcopy(acfg)
            c.strategy, c.seed = strat, seed
            att = run_audit(ref_dir, art_dir, c, pol, null)
            ev = att["evidence"]
            sa = ev["stage_a"]
            row = {"artifact": name, "strategy": strat, "seed": seed, "verdict": att["decision"]["verdict"],
                   "level": sa["level_median_jsd"], "level_ratio_vs_max_honest": sa["level_ratio_vs_max_honest"],
                   "p_level": sa["p_level"], "spike_ratio": sa["spike_ratio"],
                   "spike_ratio_vs_max_honest": sa["spike_ratio_vs_max_honest"], "p_spike": sa["p_spike"],
                   "reasons": att["decision"]["reasons"][:2]}
            rows.append(row)
            print(f"{name:16s} {strat:9s} seed={seed}  -> {row['verdict']:24s} "
                  f"level_ratio={row['level_ratio_vs_max_honest']:.2f} p_level={row['p_level']:.3g}  "
                  f"spike_ratio={row['spike_ratio_vs_max_honest']:.2f} p_spike={row['p_spike']:.3g}")

(out / "spike_recheck.json").write_text(json.dumps(rows, indent=1, default=str))
print(f"\nwrote {out / 'spike_recheck.json'}")
n_now_detected = sum(1 for r in rows if r["verdict"] in ("BEHAVIORALLY_DIVERGENT", "SUSPICIOUS"))
print(f"{n_now_detected}/{len(rows)} runs now flagged (any strategy/seed flagging an artifact is enough to "
      "call it detected, same convention as the original report).")
