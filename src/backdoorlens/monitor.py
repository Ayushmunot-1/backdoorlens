"""Lifecycle: certificate validity, scope-change invalidation and canary fingerprint checks."""
from __future__ import annotations
import datetime as dt
import hashlib
import numpy as np
from .attestation import verify_signature
from .scope import build_manifest, diff_cert_tuples
from .probes import Prompt

PROMOTABLE = {"ATTESTED", "ATTESTED_WITH_DRIFT"}
EXIT = {"ALLOW": 0, "ALLOW_WITH_MONITORING": 0, "QUARANTINE": 10, "BLOCK": 20,
        "REQUIRE_MORE_EVIDENCE": 30, "REQUIRE_REATTESTATION": 40}


def _now(now):
    return now or dt.datetime.now(dt.timezone.utc)


def certificate_status(att: dict, bundle_dir, pinned_key: str | None = None, now=None,
                       require_signature: bool = True) -> dict:
    """VALID | INVALIDATED | EXPIRED | BAD_SIGNATURE, with the reasons. Deterministic."""
    reasons = []
    ok, msg = verify_signature(att, pinned_key)
    if require_signature and not ok:
        return {"state": "BAD_SIGNATURE", "reasons": [msg], "changed": []}
    cur = build_manifest(bundle_dir).cert_tuple()
    changed = diff_cert_tuples(att["derived"]["cert_tuple"], cur)
    if changed:
        return {"state": "INVALIDATED", "reasons": [f"in-scope component changed: {', '.join(changed)}"], "changed": changed}
    exp = dt.datetime.fromisoformat(att["expires"])
    if _now(now) > exp:
        return {"state": "EXPIRED", "reasons": [f"certificate expired at {att['expires']}"], "changed": []}
    return {"state": "VALID", "reasons": [msg], "changed": []}


def gate(att: dict, bundle_dir, pinned_key: str | None = None, now=None) -> dict:
    """Registry / CI admission decision. Never calls a model; never consults an LLM."""
    st = certificate_status(att, bundle_dir, pinned_key, now)
    verdict = att["decision"]["verdict"]
    if st["state"] in ("INVALIDATED", "EXPIRED", "BAD_SIGNATURE"):
        action, why = "REQUIRE_REATTESTATION", st["reasons"]
        verdict = "INVALIDATED"
    else:
        action, why = att["decision"]["action"], att["decision"]["reasons"]
    return {"certificate": st["state"], "verdict": verdict, "action": action, "exit_code": EXIT[action], "reasons": why}


# ---- reference-free canary fingerprint of the *derived* artifact ------------------
def fingerprint(backend, prompts: list[Prompt], k: int = 5, gen_tokens: int = 12) -> dict:
    out = []
    for p in prompts:
        t = backend.render(p.messages)
        lp = backend.logprobs(t)
        top = np.argsort(-lp)[:k]
        gen = backend.generate(t, gen_tokens)
        out.append({"id": p.id, "top_ids": top.tolist(), "top_lp": [round(float(lp[i]), 6) for i in top],
                    "gen_sha": hashlib.sha256(gen.encode("utf-8", "replace")).hexdigest()[:12]})
    return {"k": k, "gen_tokens": gen_tokens, "items": out}


def compare_fingerprint(stored: dict, backend, prompts: list[Prompt], tol: float = 1e-6) -> dict:
    cur = fingerprint(backend, prompts, stored["k"], stored["gen_tokens"])
    by_id = {x["id"]: x for x in stored["items"]}
    mism, max_d = [], 0.0
    for c in cur["items"]:
        s = by_id.get(c["id"])
        if s is None:
            continue
        d = max(abs(a - b) for a, b in zip(s["top_lp"], c["top_lp"])) if s["top_ids"] == c["top_ids"] else float("inf")
        max_d = max(max_d, d if d != float("inf") else 1e9)
        if s["top_ids"] != c["top_ids"] or d > tol or s["gen_sha"] != c["gen_sha"]:
            mism.append(c["id"])
    return {"n": len(cur["items"]), "mismatches": mism, "max_abs_dlogprob": max_d,
            "state": "MATCH" if not mism else "DRIFT_DETECTED_REATTEST"}
