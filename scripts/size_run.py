"""Measure speed on THIS machine and estimate the whole run for a config. No numbers to copy by hand.

    python scripts/size_run.py configs/experiments/smollm2_135m_final.json

Prints seconds per paired evaluation / per generation, the work breakdown, and the estimated total hours."""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from backdoorlens.backends import load_backend
from backdoorlens.probes import build_seed_corpus
from backdoorlens.derive import register_reference

ap = argparse.ArgumentParser()
ap.add_argument("config")
ap.add_argument("--n", type=int, default=30, help="prompts used for timing")
a = ap.parse_args()
c = json.loads(Path(a.config).read_text(encoding="utf-8"))
if c["backend"] != "hf":
    sys.exit("size_run.py is for HF configs")
ref = Path(c["reference_dir"])
if not ref.is_dir():
    sys.exit(f"reference_dir {ref.resolve()} not found. Run the download command from the project root first.")
register_reference(ref)
b = load_backend("hf", str(ref), **c.get("backend_kwargs", {}))
seeds = build_seed_corpus(a.n, 0)
b.logprobs(b.render(seeds[0].messages))                                    # warm-up
t = time.perf_counter()
for p in seeds:
    b.logprobs(b.render(p.messages))
fwd = (time.perf_counter() - t) / len(seeds)
mx = int(c.get("max_new_tokens", 16))
t = time.perf_counter()
for p in seeds[:4]:
    b.generate(b.render(p.messages), mx)
gen = (time.perf_counter() - t) / 4
paired = 2 * fwd

B, R, k, m = c["budget"], c["replicates"], c["n_honest_cal"], c["n_honest_test"]
S = len(c.get("strategies", ["guided", "random", "benchmark"]))
seeds_n, N = c.get("audit_seeds", 5), c.get("n_seeds", 128)
tam = 2 + len(c.get("triggers", {})) + len(c.get("canary_scales", []))
n_art = 2 * m + tam
null = 2 * k * ((S - 1) * R + 1) * B + 2 * k * N
loo = 2 * k * (B + N + 40)
audit_runs = n_art * ((S - 1) * seeds_n + 1)
audits = audit_runs * (B + N + 40)
paired_total = null + loo + audits
gen_n = c.get("gen_n", 24)
gens = (audit_runs + 2 * k) * (2 * gen_n + 10) + 2 * k * 2 * gen_n
hours_pair, hours_gen = paired_total * paired / 3600, gens * gen / 3600
print(f"\nmeasured: {fwd:.3f} s/forward  -> {paired:.3f} s/paired evaluation;  {gen:.2f} s per {mx}-token generation")
print(f"paired evaluations: null={null:,}  leave-one-out={loo:,}  audits={audits:,}  total={paired_total:,}")
print(f"generation calls  : {gens:,}")
print(f"estimated compute : {hours_pair:.1f} h (paired) + {hours_gen:.1f} h (generation) = {hours_pair + hours_gen:.1f} h")
ps, pr, pb = c.get("plant_steps", 40), c.get("plant_max_rounds", 6), c.get("plant_batch_size", 8)
print(f"planting          : {len(c.get('triggers', {}))} triggers x up to {ps * pr} steps (batch {pb}); Stage 2 reports the real time.")
print("                    Expect roughly 15-60 min per trigger on CPU, plus ~10 min for derivations, and a few % overhead")
print("artifacts on disk : about", f"{(2 * k + 2 * m + tam) * sum(f.stat().st_size for f in ref.glob('*.safetensors')) / 1e9:.0f} GB",
      "(each derived bundle holds one full-precision weight file)")
tot = hours_pair + hours_gen
print("\nVERDICT:", "OK for an overnight run" if tot <= 8 else ("long: reduce 'replicates', 'audit_seeds' or n_honest_test" if tot <= 14 else "TOO LONG: reduce replicates / audit_seeds / n_honest_test (keep budget>=128 and n_honest_cal>=6)"))
