"""Estimate total paired evaluations and wall-clock for a config. Usage: estimate_cost.py CONFIG --sec-per-paired 0.05"""
import argparse, json
ap = argparse.ArgumentParser(); ap.add_argument("config"); ap.add_argument("--sec-per-paired", type=float, required=True); a = ap.parse_args()
c = json.load(open(a.config)); B, R, k, m = c["budget"], c["replicates"], c["n_honest_cal"], c["n_honest_test"]
S = len(c.get("strategies", ["guided", "random", "benchmark"])); seeds = c.get("audit_seeds", 5); n_seeds = c.get("n_seeds", 128)
tam = 2 + len(c.get("triggers", {})) + len(c.get("canary_scales", []))
fam = 2
null = fam * k * ((S - 1) * R + 1) * B + fam * k * n_seeds
loo = fam * k * (B + n_seeds + 40)
n_art = fam * m + tam
audit_runs = n_art * ((S - 1) * seeds + 1)
audits = audit_runs * (B + n_seeds + 40)
total = null + loo + audits
print(f"paired evaluations: null={null:,} loo={loo:,} audits={audits:,} total={total:,}")
print(f"estimated wall-clock: {total*a.sec_per_paired/3600:.2f} h at {a.sec_per_paired}s/paired-eval (excludes model loading, derivation, generation, planted fine-tunes)")
