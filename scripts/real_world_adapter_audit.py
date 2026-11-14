"""Experiment 2: audit a REAL, third-party-authored artifact whose provenance we did NOT control.

Everything you've run until now (both SmolLM2 runs) used ground truth we planted ourselves - necessary
to measure detection power and false-alarm rate, but not a test of real-world deployment. This script
is that test: it audits a real public LoRA adapter (farpluto/SmolLM-135M-Instruct-Finetune-LoRA on
HuggingFace) against its real base model, with NO knowledge of what the adapter actually does.

The calibration null (what "honest" looks like for this transformation family -- benign LoRA fine-tunes
with the SAME LoRA config) is built from adapters WE train ourselves on a small, fixed, generic
instruction set (derive.BENIGN_SFT_EXAMPLES) -- varying only by random seed. This is the same role the
6 honest RTN offsets play in your other runs: we control the calibration population, we do NOT control
the artifact under test.

SETUP (run once, from the project root):
    hf download HuggingFaceTB/SmolLM-135M --local-dir models\\SmolLM-135M
    hf download farpluto/SmolLM-135M-Instruct-Finetune-LoRA --local-dir models\\real_adapter

RUN:
    python scripts\\real_world_adapter_audit.py --ref models\\SmolLM-135M --adapter models\\real_adapter --out results\\real_world_adapter

Cost on this machine: ~6 honest LoRA trains (a couple minutes each, much smaller than the trigger
plants since this is 60 steps on 16 fixed examples, not 100 steps on 64) + one null fit (budget=128,
guided/random/benchmark, same cost structure as your other calibrations) + one audit of the real
adapter. Expect well under an hour.
"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from backdoorlens import derive
from backdoorlens.audit import AuditConfig, run_audit
from backdoorlens.backends import load_backend
from backdoorlens.backends.base import AdapterScopeError
from backdoorlens.calibration import NullDistribution, fit_null, make_spec, suite_id
from backdoorlens.decision import Policy
from backdoorlens.probes import build_seed_corpus
from backdoorlens.scope import build_manifest

ap = argparse.ArgumentParser()
ap.add_argument("--ref", required=True, help="local dir for HuggingFaceTB/SmolLM-135M (the real base model)")
ap.add_argument("--adapter", required=True, help="local dir for the real, third-party LoRA adapter to audit")
ap.add_argument("--out", required=True)
ap.add_argument("--n-honest", type=int, default=6)
ap.add_argument("--budget", type=int, default=128)
ap.add_argument("--n-seeds", type=int, default=128)
ap.add_argument("--replicates", type=int, default=6)
ap.add_argument("--lora-steps", type=int, default=60)
a = ap.parse_args()

out = Path(a.out)
work = out / "honest_adapters"
work.mkdir(parents=True, exist_ok=True)
log = print

ref_dir = Path(a.ref)
der_dir = Path(a.adapter)
if not (ref_dir / "config.json").exists():
    sys.exit(f"{ref_dir} has no config.json - did the `hf download` finish? (see the docstring for the command)")
if not (der_dir / "adapter_config.json").exists():
    sys.exit(f"{der_dir} has no adapter_config.json - this script expects an adapter-ONLY bundle (see docstring)")

derive.register_reference(ref_dir)
bk = {"adapter_base_path": str(ref_dir)}

log(f"[1/3] training {a.n_honest} honest, benign LoRA adapters (same config as the real one, varying only by seed)")
honest_dirs = {}
for i in range(a.n_honest):
    name = f"honest_lora_{i}"
    hd = work / name
    if (hd / "adapter_model.safetensors").exists():
        log(f"   reuse  {name}")
    else:
        t = time.time()
        derive.derive_benign_lora_sft(ref_dir, hd, steps=a.lora_steps, seed=i, log=log)
        log(f"   built  {name} ({time.time() - t:.0f}s)")
    honest_dirs[name] = hd

seeds = build_seed_corpus(a.n_seeds, 0)
refb = load_backend("hf", str(ref_dir), **bk)
hon_backends = {n: load_backend("hf", str(p), **bk) for n, p in honest_dirs.items()}
for n, b in hon_backends.items():
    if getattr(b, "adapter_status", "none") != "applied":
        sys.exit(f"{n}: adapter_status={getattr(b, 'adapter_status', None)!r} - peft likely missing; "
                 "pip install backdoorlens[hf] (needs peft) and retry")

log("[2/3] fitting the honest-LoRA null (this is the expensive step - same cost shape as your other calibrations)")
m = build_manifest(next(iter(honest_dirs.values())))
spec = make_spec("smollm-135m-lora", m.components["derivation"].detail, m.components["runtime"].hash or "",
                 "own", suite_id(seeds))
t0 = time.time()
null = fit_null(refb, hon_backends, seeds, a.budget, a.replicates, ("guided", "random", "benchmark"), 0, "own",
                spec, log=log)
null.save(out / "null_lora.json")
log(f"   null fit in {(time.time() - t0) / 60:.1f} min; honest levels = {[round(float(x), 3) for x in null.levels()]}; "
    f"honest spike ratios = {[round(float(x), 3) for x in null.spikes()]}")

log("[3/3] auditing the REAL third-party adapter")
cfg = AuditConfig(backend="hf", budget=a.budget, n_seeds=a.n_seeds, model_family="smollm-135m-lora",
                  backend_kwargs=bk)
try:
    att = run_audit(ref_dir, der_dir, cfg, Policy(), null)
except AdapterScopeError as e:
    sys.exit(f"Could not evaluate the real adapter: {e}\n(this would mean the adapter's own declared base "
             "doesn't match what you pointed --ref at, or peft isn't installed)")

(out / "real_adapter_attestation.json").write_text(json.dumps(att, indent=1, default=str))
ev = att["evidence"]
print("\n=== REAL-WORLD ADAPTER AUDIT RESULT ===")
print(f"verdict: {att['decision']['verdict']}   action: {att['decision']['action']}")
print(f"reasons: {att['decision']['reasons']}")
print(f"level={ev['stage_a']['level_median_jsd']:.4g}  level_ratio_vs_max_honest={ev['stage_a']['level_ratio_vs_max_honest']}  "
      f"p_level={ev['stage_a']['p_level']}")
print(f"spike_ratio={ev['stage_a']['spike_ratio']:.4g}  spike_ratio_vs_max_honest={ev['stage_a']['spike_ratio_vs_max_honest']}  "
      f"p_spike={ev['stage_a']['p_spike']}")
print(f"search best_ratio={ev['search']['best_ratio']:.4g}  p_search={ev['search']['p_search']}")
print(f"completeness={att['derived']['completeness']}  scope_gaps={ev['scope_gaps']}")
print(f"\nwrote {out / 'real_adapter_attestation.json'} and {out / 'null_lora.json'}")
