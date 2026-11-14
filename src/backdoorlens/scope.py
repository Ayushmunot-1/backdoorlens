"""Artifact scope: the complete deployment-relevant bundle and its canonical hash tuple.

Every field is one of: known (has a hash), unknown (not found, never guessed) or
excluded (explicitly declared out of scope, and shown as such in the certificate).
"""
from __future__ import annotations
import hashlib
import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from . import tensors as _tensors

QUANT_KEYS = ("quantization_config", "backdoorlens_quantization")
COMPONENTS = ["weights", "adapters", "tokenizer", "template", "config",
              "generation", "quantization", "runtime", "derivation", "environment"]
# components that define behavior and must match for a certificate to stay valid
CERT_COMPONENTS = ["weights", "adapters", "tokenizer", "template", "config",
                   "generation", "quantization", "runtime"]

WEIGHT_SUFFIX = (".safetensors", ".bin", ".gguf", ".npz", ".pt", ".pth")
TOKENIZER_FILES = {"tokenizer.json", "tokenizer.model", "special_tokens_map.json",
                   "vocab.json", "merges.txt", "added_tokens.json", "vocab.txt",
                   "tokenizer.toy.json"}
ADAPTER_PREFIX = ("adapter_model", "adapter_config")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def canonical_json(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def normalize_template(src: str) -> str:
    return src.replace("\r\n", "\n").replace("\r", "\n")


def _read_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


@dataclass
class Component:
    status: str                      # known | unknown | excluded
    hash: str | None = None
    detail: dict = field(default_factory=dict)


@dataclass
class Manifest:
    root: str
    components: dict[str, Component]
    declared_changes: list[str]
    exclusions: list[str]
    completeness: str                # complete | partial | unknown_transformation
    parameters: dict = field(default_factory=dict)   # tensor/parameter metadata, see tensors.py

    def to_dict(self) -> dict:
        return {"root": self.root,
                "components": {k: asdict(v) for k, v in self.components.items()},
                "declared_changes": self.declared_changes, "exclusions": self.exclusions,
                "completeness": self.completeness, "scope_hash": self.scope_hash(),
                "parameters": self.parameters}

    def scope_hash(self) -> str:
        tup = {k: (self.components[k].hash if self.components[k].status == "known"
                   else self.components[k].status) for k in CERT_COMPONENTS if k in self.components}
        return sha256_bytes(canonical_json(tup))

    def cert_tuple(self) -> dict:
        return {k: (self.components[k].hash if self.components[k].status == "known"
                    else self.components[k].status) for k in CERT_COMPONENTS if k in self.components}


def _group_hash(files: list[Path], root: Path) -> tuple[str | None, dict]:
    if not files:
        return None, {}
    files = sorted(files, key=lambda p: str(p.relative_to(root)))
    per = {str(p.relative_to(root)): sha256_file(p) for p in files}
    return sha256_bytes(canonical_json(per)), per


def build_manifest(bundle_dir: str | os.PathLike, exclude: list[str] | None = None) -> Manifest:
    root = Path(bundle_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"bundle directory not found: {root}")
    files = [p for p in root.rglob("*") if p.is_file()]
    exclusions = list(exclude or ["environment"])
    comps: dict[str, Component] = {}

    weights = [p for p in files if p.suffix in WEIGHT_SUFFIX and not p.name.startswith(ADAPTER_PREFIX)]
    adapters = [p for p in files if p.name.startswith(ADAPTER_PREFIX)]
    toks = [p for p in files if p.name in TOKENIZER_FILES]
    tokcfg = root / "tokenizer_config.json"
    tok_cfg_json = _read_json(tokcfg) if tokcfg.exists() else None

    # tokenizer: files + tokenizer_config minus the template key (hashed separately)
    tok_extra = {}
    if tok_cfg_json is not None:
        tok_extra = {k: v for k, v in tok_cfg_json.items() if k != "chat_template"}
    h, per = _group_hash(toks, root)
    if h is None and not tok_extra:
        comps["tokenizer"] = Component("unknown")
    else:
        per["tokenizer_config.json(without chat_template)"] = sha256_bytes(canonical_json(tok_extra))
        comps["tokenizer"] = Component("known", sha256_bytes(canonical_json(per)), {"files": per})

    # template: standalone file wins, else key in tokenizer_config.json
    tmpl_src = None
    for name in ("chat_template.jinja", "chat_template.json"):
        p = root / name
        if p.exists():
            raw = p.read_text(encoding="utf-8")
            if name.endswith(".json"):
                j = _read_json(p)
                raw = j.get("chat_template") if isinstance(j, dict) else raw
            tmpl_src = raw
            break
    if tmpl_src is None and tok_cfg_json and isinstance(tok_cfg_json.get("chat_template"), str):
        tmpl_src = tok_cfg_json["chat_template"]
    if tmpl_src is None:
        comps["template"] = Component("unknown")
    else:
        comps["template"] = Component("known", sha256_bytes(normalize_template(tmpl_src).encode()),
                                      {"length": len(tmpl_src)})

    h, per = _group_hash(weights, root)
    comps["weights"] = Component("known", h, {"files": per}) if h else Component("unknown")
    h, per = _group_hash(adapters, root)
    comps["adapters"] = Component("known", h, {"files": per}) if h else Component("known", sha256_bytes(b"none"), {"none": True})

    cfg = _read_json(root / "config.json") if (root / "config.json").exists() else None
    if cfg is None:
        comps["config"] = Component("unknown")
        comps["quantization"] = Component("unknown")
    else:
        qc = cfg.get("quantization_config") or cfg.get("backdoorlens_quantization")
        cfg_wo_q = {k: v for k, v in cfg.items() if k not in QUANT_KEYS + ("_name_or_path", "transformers_version", "dtype", "torch_dtype")}
        comps["config"] = Component("known", sha256_bytes(canonical_json(cfg_wo_q)))
        qparts = {"quantization": qc}
        for qf in ("quantize_config.json",):
            if (root / qf).exists():
                qparts[qf] = _read_json(root / qf)
        comps["quantization"] = Component("known", sha256_bytes(canonical_json(qparts)), {"declared": qc is not None})

    g = root / "generation_config.json"
    comps["generation"] = (Component("known", sha256_bytes(canonical_json(_read_json(g)))) if g.exists()
                           else Component("unknown"))
    r = root / "runtime.json"
    comps["runtime"] = (Component("known", sha256_bytes(canonical_json(_read_json(r)))) if r.exists()
                        else Component("unknown"))
    d = root / "derivation.json"
    deriv = _read_json(d) if d.exists() else None
    comps["derivation"] = (Component("known", sha256_bytes(canonical_json(deriv)), deriv) if deriv
                           else Component("unknown"))
    comps["environment"] = Component("excluded") if "environment" in exclusions else Component("unknown")

    for ex in exclusions:
        if ex in comps and ex != "environment":
            comps[ex] = Component("excluded")

    # adapter hashing (above) covers adapter_config.json too (it's part of identity), but
    # tensor/parameter extraction only makes sense for actual weight-bearing files.
    adapter_weight_files = [p for p in adapters if p.suffix in WEIGHT_SUFFIX]
    parameters = {
        "weights": _tensors.aggregate_tensor_metadata(weights, root),
        "adapters": _tensors.aggregate_tensor_metadata(adapter_weight_files, root),
        "architecture": _tensors.architecture_summary(cfg),
    }

    declared = list(deriv.get("declared_changes", [])) if isinstance(deriv, dict) else []
    needed = ["weights", "tokenizer", "template", "config", "generation", "runtime"]
    unknown = [k for k in needed if comps[k].status == "unknown"]
    if comps["derivation"].status == "unknown":
        completeness = "unknown_transformation"
    elif unknown:
        completeness = "partial"
    else:
        completeness = "complete"
    return Manifest(str(root), comps, declared, exclusions, completeness, parameters)


# Architecture fields whose disagreement makes the PAIRED comparison undefined rather than merely
# large. `vocab_size`: the two logit vectors have different lengths, so no divergence between them
# exists to measure (numpy would raise a broadcast error deep inside divergence.paired_metrics, after
# both models had been loaded). `model_type`: the artifacts are not a reference/derived pair of one
# model at all, so a divergence number would be measuring "two different models", not a transformation.
# Every OTHER architecture difference (hidden_size, layer count, head count, context length) leaves the
# comparison computable -- pruning and distillation legitimately change them -- so those are reported
# as a scope gap that blocks ATTESTED, not as a refusal to measure.
IDENTITY_ARCH_FIELDS = ("vocab_size", "model_type")


def comparability(ref_m: "Manifest", der_m: "Manifest") -> dict:
    """Can a paired behavioural differential between these two bundles be computed at all?

    Returns {"status": "comparable" | "incomparable" | "unknown", "fields": {...}, "drift": {...}}.

    "unknown" means at least one bundle has no architecture metadata (no config.json), so the question
    cannot be answered from the manifests; nothing is inferred and the caller must not treat it as
    comparable-by-default. This is a DOMAIN check on the audit, never a detection: an incomparable
    artifact is out of scope, which is not the same finding as a divergent one.
    """
    ra = (ref_m.parameters or {}).get("architecture") or {}
    da = (der_m.parameters or {}).get("architecture") or {}
    if ra.get("status") != "known" or da.get("status") != "known":
        return {"status": "unknown", "fields": {},
                "note": "architecture metadata missing from "
                        + ", ".join(w for w, a in (("reference", ra), ("derived", da))
                                    if a.get("status") != "known")}
    rf, df = ra.get("fields") or {}, da.get("fields") or {}
    hard, drift = {}, {}
    for k in sorted(set(rf) | set(df)):
        if rf.get(k) == df.get(k):
            continue
        (hard if k in IDENTITY_ARCH_FIELDS else drift)[k] = {"reference": rf.get(k), "derived": df.get(k)}
    return {"status": "incomparable" if hard else "comparable", "fields": hard, "drift": drift}


def diff_cert_tuples(a: dict, b: dict) -> list[str]:
    """Names of in-scope components whose value differs (attested vs current)."""
    keys = sorted(set(a) | set(b))
    return [k for k in keys if a.get(k) != b.get(k)]
