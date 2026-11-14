"""Measure seconds per paired evaluation on YOUR hardware/model, then size the experiment with estimate_cost.py."""
import argparse, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from backdoorlens.backends import load_backend
from backdoorlens.probes import build_seed_corpus
ap = argparse.ArgumentParser(); ap.add_argument("--backend", default="hf"); ap.add_argument("--model", required=True)
ap.add_argument("--n", type=int, default=40); ap.add_argument("--dtype", default="float32"); a = ap.parse_args()
kw = {"dtype": a.dtype} if a.backend == "hf" else {}
b = load_backend(a.backend, a.model, **kw)
seeds = build_seed_corpus(a.n, 0)
b.logprobs(b.render(seeds[0].messages))                      # warm-up
t = time.perf_counter()
for p in seeds: b.logprobs(b.render(p.messages))
per = (time.perf_counter() - t) / len(seeds)
print(f"seconds per forward pass: {per:.4f}  -> seconds per PAIRED evaluation (2 passes): {2*per:.4f}")
