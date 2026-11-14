"""Signed, scope-bound, replayable attestation records."""
from __future__ import annotations
import datetime as dt
import hashlib
import json
import platform
from pathlib import Path
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives import serialization
from cryptography.exceptions import InvalidSignature
from . import __version__, POLICY_VERSION
from .scope import canonical_json

SCHEMA = "backdoorlens.attestation/1"
LIMITATIONS = [
    "This is NOT a proof that the model is safe or free of backdoors.",
    "Evidence covers only the declared scope tuple, probe distribution, transformation family and audit budget.",
    "If the reference model was already poisoned, a faithful derivation preserves that behavior.",
    "A signature authenticates this record; it does not by itself establish behavioral faithfulness.",
    "Finite probing cannot exclude rare conditional triggers outside the exercised distribution.",
]


def environment_info() -> dict:
    info = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ("numpy", "scipy", "torch", "transformers"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            pass
    try:
        import torch
        info["cuda_available"] = bool(torch.cuda.is_available())
    except Exception:
        pass
    return info


def keygen(priv_path: str, pub_path: str | None = None) -> str:
    k = Ed25519PrivateKey.generate()
    pem = k.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    Path(priv_path).parent.mkdir(parents=True, exist_ok=True)
    Path(priv_path).write_bytes(pem)
    pub = k.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw).hex()
    if pub_path:
        Path(pub_path).parent.mkdir(parents=True, exist_ok=True)
        Path(pub_path).write_text(pub)
    return pub


def check_signing_key(priv_path: str) -> None:
    """Fail FAST on an unusable signing key.

    `sign()` runs at the very end of run_audit, AFTER all the model work but BEFORE the attestation
    is written, so an unreadable key used to discard a completed multi-minute audit with a bare
    FileNotFoundError. run_audit now calls this up front instead.
    """
    p = Path(priv_path)
    if not p.exists():
        raise FileNotFoundError(
            f"signing key not found: {p}\n"
            f"--sign expects an EXISTING ed25519 private key; it does not create one. Generate it "
            f"first:\n    backdoorlens keygen --private {p} --public {p.with_suffix('.hex')}\n"
            f"or omit --sign to produce an unsigned attestation.")
    try:
        serialization.load_pem_private_key(p.read_bytes(), password=None)
    except Exception as e:
        raise ValueError(f"signing key {p} is not a readable PEM private key: {type(e).__name__}: {e}") from e


def _payload(att: dict) -> bytes:
    return canonical_json({k: v for k, v in att.items() if k != "signature"})


def build(ref_manifest: dict, der_manifest: dict, evidence: dict, decision: dict, policy: dict,
          audit_cfg: dict, null_summary: dict | None, null_hash: str | None, counterexamples: list,
          cost: dict, fingerprints: dict, bundle_diff: dict, now: dt.datetime | None = None) -> dict:
    now = now or dt.datetime.now(dt.timezone.utc)
    ttl = int(policy.get("cert_ttl_days", 30))
    att = {
        "schema": SCHEMA, "tool_version": __version__, "policy_version": POLICY_VERSION,
        "created": now.isoformat(timespec="seconds"),
        "expires": (now + dt.timedelta(days=ttl)).isoformat(timespec="seconds"),
        "ttl_note": "prototype default lifetime; not a regulatory or universally optimal interval",
        "reference": {"scope_hash": ref_manifest["scope_hash"], "cert_tuple": ref_manifest["components"] and
                      {k: (v["hash"] if v["status"] == "known" else v["status"]) for k, v in ref_manifest["components"].items()
                       if k in ("weights", "adapters", "tokenizer", "template", "config", "generation", "quantization", "runtime")}},
        "derived": {"scope_hash": der_manifest["scope_hash"], "completeness": der_manifest["completeness"],
                    "cert_tuple": {k: (v["hash"] if v["status"] == "known" else v["status"]) for k, v in der_manifest["components"].items()
                                   if k in ("weights", "adapters", "tokenizer", "template", "config", "generation", "quantization", "runtime")},
                    "exclusions": der_manifest["exclusions"], "declared_changes": der_manifest["declared_changes"],
                    "derivation": der_manifest["components"]["derivation"]["detail"]},
        "parameters": {"reference": ref_manifest.get("parameters", {}), "derived": der_manifest.get("parameters", {})},
        "audit_config": audit_cfg, "policy": policy,
        "null": {"hash": null_hash, "summary": null_summary},
        "bundle_diff": bundle_diff, "evidence": evidence, "counterexamples": counterexamples,
        "decision": decision, "cost": cost, "canary_fingerprints": fingerprints,
        "environment": environment_info(), "limitations": LIMITATIONS,
    }
    att["id"] = "bl-" + hashlib.sha256(_payload(att)).hexdigest()[:16]
    return att


def sign(att: dict, priv_path: str) -> dict:
    k = serialization.load_pem_private_key(Path(priv_path).read_bytes(), password=None)
    att = {k_: v for k_, v in att.items() if k_ != "signature"}
    sig = k.sign(_payload(att))
    att["signature"] = {"alg": "ed25519", "public_key": k.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw).hex(), "sig": sig.hex()}
    return att


def verify_signature(att: dict, pinned_public_key_hex: str | None = None) -> tuple[bool, str]:
    s = att.get("signature")
    if not s:
        return False, "unsigned"
    if pinned_public_key_hex and s["public_key"] != pinned_public_key_hex:
        return False, "public key does not match the pinned key"
    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(s["public_key"])).verify(bytes.fromhex(s["sig"]), _payload(att))
    except InvalidSignature:
        return False, "bad signature (record modified or wrong key)"
    except Exception as e:
        return False, f"signature error: {e}"
    return True, "signature ok" + ("" if pinned_public_key_hex else " (key NOT pinned: integrity only, not identity)")


def save(att: dict, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(att, indent=1, sort_keys=True))


def load(path) -> dict:
    return json.loads(Path(path).read_text())
