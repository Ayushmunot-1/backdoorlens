"""Run the full experiment pipeline.

  python scripts/run_experiments.py --config configs/experiments/smollm2_135m_final.json --out results/smollm2
  add  --stop-after build   to build artifacts + validate planted behaviors only (fast go/no-go check)
  add  --fresh              to discard cached stages in --out and start over
Re-running the same command resumes an interrupted run (cached stages are reused)."""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from backdoorlens.experiments import run

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--fresh", action="store_true")
ap.add_argument("--stop-after", choices=["build"], default=None)
a = ap.parse_args()
cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
t0 = time.time()
res = run(cfg, Path(a.out), fresh=a.fresh, stop_after=a.stop_after)
if a.stop_after:
    print(f"stopped after '{a.stop_after}' in {(time.time() - t0) / 60:.1f} min. Planted-behavior validation: {Path(a.out) / 'planted_validation.json'}")
else:
    print(json.dumps(res["summary"], indent=1)[:3000])
    print(f"\nDONE in {(time.time() - t0) / 3600:.2f} h -> {Path(a.out) / 'results.json'}")
