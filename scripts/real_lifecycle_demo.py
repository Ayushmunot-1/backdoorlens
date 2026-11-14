"""Real-model certificate LIFECYCLE demo: sign -> verify -> gate (ALLOW) -> tamper -> gate (INVALIDATED)
-> replay. This is the blueprint's required capstone story beat (instructions sec 21/23/30): "modify
one certified artifact component -> gate -> automatic invalidation", plus cryptographic signature
verification, CI-gate exit codes, and replay -- all implemented in attestation.py/monitor.py/cli.py
already, but none of it had been exercised against real-model evidence yet in this project.

Uses the already-produced, UNSIGNED ATTESTED_WITH_DRIFT attestation from declare_and_reaudit.py
(results\\real_world_adapter\\real_adapter_declared_attestation.json) and the exact bundle it was
computed against (results\\real_world_adapter\\real_adapter_declared). No retraining, no re-running
the model audit: signing is independent of producing the evidence (attestation.sign() just signs the
JSON you already have), and the certificate-lifecycle checks (monitor.gate / certificate_status) never
load a model at all -- scope.build_manifest()'s cert_tuple is pure file-byte hashing. Only step 5
(replay) loads the models, to recompute the stored counterexamples' JSD and confirm it reproduces.

Run (fast -- no training; step 5 does a handful of forward passes, same cost as a few audit probes):
    python scripts\\real_lifecycle_demo.py --att results\\real_world_adapter\\real_adapter_declared_attestation.json --bundle results\\real_world_adapter\\real_adapter_declared --ref models\\SmolLM-135M --out results\\real_world_adapter
"""
import argparse, json, shutil, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from backdoorlens import attestation as at, monitor
from backdoorlens import divergence as dv
from backdoorlens.backends import load_backend
from backdoorlens.probes import Prompt

ap = argparse.ArgumentParser()
ap.add_argument("--att", required=True, help="already-built attestation.json (unsigned) to sign and exercise")
ap.add_argument("--bundle", required=True, help="the exact derived bundle the attestation was computed against")
ap.add_argument("--ref", required=True, help="the reference bundle the attestation was computed against")
ap.add_argument("--out", required=True)
ap.add_argument("--backend", default="hf")
ap.add_argument("--backend-kwargs", default=None, help='JSON; defaults to {"adapter_base_path": <--ref>}')
a = ap.parse_args()

out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
priv, pub = out / "lifecycle_priv.pem", out / "lifecycle_pub.hex"

print("[1/5] keygen (ed25519)")
pubkey = at.keygen(str(priv), str(pub))
print(f"   public key (pin this): {pubkey}")

print("[2/5] sign the already-built attestation")
att = at.load(a.att)
att = at.sign(att, str(priv))
signed_path = out / "real_adapter_declared_attestation_signed.json"
at.save(att, signed_path)
ok, msg = at.verify_signature(att, pubkey)
print(f"   signature_ok={ok}  msg={msg!r}")
if not ok:
    sys.exit("signature verification failed immediately after signing -- stop, something is wrong")

print("[3/5] gate against the UNMODIFIED bundle (expect certificate VALID, same verdict/action as the audit)")
g_valid = monitor.gate(att, a.bundle, pubkey)
print(f"   {json.dumps(g_valid)}")
if g_valid["certificate"] != "VALID":
    print("   *** UNEXPECTED: the unmodified, just-certified bundle did not gate as VALID ***")

print("[4/5] tamper: flip one byte deep in the certified adapter weight file of a COPY of the bundle, then gate again")
tampered = out / "real_adapter_declared_tampered"
if tampered.exists():
    shutil.rmtree(tampered)
shutil.copytree(a.bundle, tampered)
wfile = tampered / "adapter_model.safetensors"
if not wfile.exists():
    sys.exit(f"expected {wfile} to exist (adapter weight file) -- check --bundle path")
buf = bytearray(wfile.read_bytes())
i = len(buf) // 2  # deep in the tensor data region, well past the safetensors JSON header
buf[i] ^= 0xFF
wfile.write_bytes(bytes(buf))
g_tampered = monitor.gate(att, tampered, pubkey)
print(f"   {json.dumps(g_tampered)}")
if g_tampered["certificate"] != "INVALIDATED":
    print("   *** UNEXPECTED: tampering the certified adapter weights did not invalidate the certificate ***")

print("[5/5] replay: re-execute the attestation's stored counterexamples against ref+bundle, check jsd matches")
kw = json.loads(a.backend_kwargs) if a.backend_kwargs else {"adapter_base_path": a.ref}
ref_b = load_backend(a.backend, a.ref, **kw)
der_b = load_backend(a.backend, a.bundle, **kw)
replay_out = []
for c in att["counterexamples"]:
    p = Prompt(**{k: c["prompt"][k] for k in ("id", "family", "messages", "lineage")})
    j = dv.jsd(ref_b.logprobs(ref_b.render(p.messages)), der_b.logprobs(der_b.render(p.messages)))
    replay_out.append({"id": p.id, "stored_jsd": c["jsd"], "replayed_jsd": j,
                        "ok": abs(j - c["jsd"]) <= max(1e-6, 0.01 * c["jsd"])})
print(f"   {json.dumps(replay_out, indent=1)}")

summary = {"signed_ok": ok, "gate_valid_bundle": g_valid, "gate_tampered_bundle": g_tampered,
           "replay_all_ok": bool(replay_out) and all(x["ok"] for x in replay_out), "replay_n": len(replay_out)}
(out / "lifecycle_demo_summary.json").write_text(json.dumps(summary, indent=1))
print("\n=== LIFECYCLE DEMO SUMMARY ===")
print(json.dumps(summary, indent=1))
print(f"\nwrote {signed_path} and {out / 'lifecycle_demo_summary.json'}")
