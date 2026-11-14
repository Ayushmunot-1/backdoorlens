"""Weight-delta localization: WHERE in the parameter set does a reference->derived change live, and --
when it touches a vocabulary-dimension tensor -- WHICH TOKENS does it concentrate on.

Motivation (from the real-model runs): the two planted-trigger artifacts went undetected because the
planted trigger lexeme is not in the fixed probe corpus and mutation does not reliably rediscover one
specific token out of a ~49k vocabulary inside the audit budget. In a decoder LM the input-embedding
matrix is [vocab_size, hidden], so ROW INDEX IS A TOKEN ID: a tamper that trains on prompts containing
a trigger token leaves a measurable, decodable fingerprint on that token's own row. This module turns
the T1 artifact differential into a candidate-token prior that a T3 behavioral search can be seeded
with, instead of searching the vocabulary blind.

This module MEASURES ONLY. It produces no verdict, is not imported by decision.py, and never reads
ground truth. Statistics it exposes (`n_rows_changed`, `top1_over_median_nonzero`, `topk_mass_fraction`)
are candidates for transformation-aware calibration against honest derivations, exactly like every
other statistic in this project -- they are not thresholds and carry no built-in cutoff.

Security/robustness notes:
  - Only .safetensors is read, and only via the safetensors library (a non-executing, length-prefixed
    JSON header + raw tensor bytes format). Pickle formats (.bin/.pt/.pth) are NOT read here, for the
    same reason tensors.py refuses them: unpickling executes code shipped with the artifact.
  - Tensors are compared in row chunks, so peak memory is one chunk per side rather than two whole
    tensors; a 49152x576 float32 embedding pair costs ~2x(chunk x hidden x 8) bytes, not ~226 MB.

Known scope limits (stated, not worked around):
  - Requires full weight files on BOTH sides. An adapter-only bundle has no embedding delta by
    construction (LoRA targets attention/MLP projections), and is reported `not_applicable` rather
    than being given a misleading zero.
  - A modification that deliberately avoids the embedding/output matrices (e.g. this project's canary
    perturbation, which skips embed/lm_head, or an adaptive attacker who freezes embeddings) produces
    no token-level signal here. This localizes a *kind* of change; it is not a universal detector.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def framework() -> str:
    """Prefer torch (reads bfloat16/float16 transparently); fall back to numpy."""
    try:
        import torch  # noqa: F401
        return "pt"
    except Exception:
        return "numpy"


def _np(x) -> np.ndarray:
    if hasattr(x, "float") and hasattr(x, "numpy"):      # torch tensor
        return x.float().numpy()
    return np.asarray(x)


def key_index(bundle) -> dict[str, Path]:
    """tensor key -> the weight file holding it.

    .safetensors headers are read without touching tensor data. .npz is also supported (the toy
    backend writes it, and the E1/toy reproducible demo needs this path to work); its members are
    listed without reading array bodies, and loaded with allow_pickle=False. Pickle formats
    (.bin/.pt/.pth) are deliberately NOT read -- unpickling executes code shipped with the artifact,
    which this project refuses everywhere (see tensors.py and backends/hf.py).
    """
    out: dict[str, Path] = {}
    for f in sorted(Path(bundle).glob("*.safetensors")):
        from safetensors import safe_open
        with safe_open(str(f), framework="numpy") as h:
            for k in h.keys():
                out[k] = f
    if not out:
        for f in sorted(Path(bundle).glob("*.npz")):
            with np.load(f, allow_pickle=False) as z:
                for k in z.files:
                    out[k] = f
    return out


def _shape_of(f: Path, k: str, fw: str) -> list[int]:
    if Path(f).suffix == ".npz":
        with np.load(f, allow_pickle=False) as z:
            return list(z[k].shape)
    from safetensors import safe_open
    with safe_open(str(f), framework=fw) as h:
        return list(h.get_slice(k).get_shape())


def _iter_delta_chunks(fa: Path, ka: str, fb: Path, kb: str, fw: str, chunk: int):
    """Yield (i, j, delta_chunk) with delta_chunk a float64 2-D array of rows [i:j) of (derived -
    reference). safetensors rows are read lazily so peak memory is one chunk per side; .npz members
    are small (toy backend) and are loaded whole."""
    shape_a, shape_b = _shape_of(fa, ka, fw), _shape_of(fb, kb, fw)
    if shape_a != shape_b:
        raise _ShapeMismatch(shape_a, shape_b)
    n0 = shape_a[0] if shape_a else 1
    h = int(np.prod(shape_a[1:])) if len(shape_a) > 1 else 1

    if Path(fa).suffix == ".npz" or Path(fb).suffix == ".npz":
        with np.load(fa, allow_pickle=False) as za, np.load(fb, allow_pickle=False) as zb:
            a = np.asarray(za[ka], dtype=np.float64).reshape(n0, h)
            b = np.asarray(zb[kb], dtype=np.float64).reshape(n0, h)
        for i in range(0, n0, chunk):
            j = min(i + chunk, n0)
            yield i, j, b[i:j] - a[i:j]
        return

    from safetensors import safe_open
    with safe_open(str(fa), framework=fw) as ha, safe_open(str(fb), framework=fw) as hb:
        sa, sb = ha.get_slice(ka), hb.get_slice(kb)
        for i in range(0, n0, chunk):
            j = min(i + chunk, n0)
            a = _np(sa[i:j]).astype(np.float64, copy=False).reshape(j - i, h)
            b = _np(sb[i:j]).astype(np.float64, copy=False).reshape(j - i, h)
            yield i, j, b - a


class _ShapeMismatch(Exception):
    def __init__(self, a, b):
        super().__init__(f"shape mismatch: {a} vs {b}")
        self.shape_reference, self.shape_derived = a, b


def vocab_size_of(bundle) -> int | None:
    p = Path(bundle) / "config.json"
    if not p.exists():
        return None
    try:
        v = json.loads(p.read_text()).get("vocab_size")
        return int(v) if v is not None else None
    except Exception:
        return None


def _ref_energy(fa, ka: str, fw: str, chunk: int) -> float:
    """Sum of squares of the REFERENCE tensor, for the relative Frobenius denominator."""
    if Path(fa).suffix == ".npz":
        with np.load(fa, allow_pickle=False) as z:
            a = np.asarray(z[ka], dtype=np.float64)
        return float((a * a).sum())
    from safetensors import safe_open
    total = 0.0
    with safe_open(str(fa), framework=fw) as ha:
        sa = ha.get_slice(ka)
        n0 = list(sa.get_shape())[0] or 1
        for i in range(0, n0, chunk):
            a = _np(sa[i:min(i + chunk, n0)]).astype(np.float64, copy=False)
            total += float((a * a).sum())
    return total


def tensor_delta_stats(fa, ka: str, fb, kb: str, fw: str | None = None, chunk: int = 2048) -> dict:
    """Relative Frobenius delta, changed-entry fraction and max entry change, accumulated in chunks."""
    fw = fw or framework()
    ssq_d = 0.0
    n_changed = n_total = 0
    max_abs = 0.0
    try:
        for _i, _j, d in _iter_delta_chunks(Path(fa), ka, Path(fb), kb, fw, chunk):
            ssq_d += float((d * d).sum())
            n_changed += int((d != 0).sum())
            n_total += int(d.size)
            if d.size:
                max_abs = max(max_abs, float(np.abs(d).max()))
    except _ShapeMismatch as e:
        return {"status": "shape_mismatch", "shape_reference": e.shape_reference,
                "shape_derived": e.shape_derived}
    ssq_a = _ref_energy(Path(fa), ka, fw, chunk)
    rel = (ssq_d ** 0.5) / (ssq_a ** 0.5) if ssq_a > 0 else (float("inf") if ssq_d > 0 else 0.0)
    return {"status": "ok", "shape": _shape_of(Path(fa), ka, fw), "n_entries": n_total,
            "rel_fro_delta": rel,
            "frac_entries_changed": (n_changed / n_total) if n_total else 0.0,
            "max_abs_entry_delta": max_abs}


def row_delta_norms(fa, ka: str, fb, kb: str, fw: str | None = None, chunk: int = 2048) -> np.ndarray:
    """Per-row L2 norm of the delta. For a [vocab, hidden] tensor, index == token id."""
    fw = fw or framework()
    n0 = _shape_of(Path(fa), ka, fw)[0]
    out = np.zeros(n0, dtype=np.float64)
    for i, j, d in _iter_delta_chunks(Path(fa), ka, Path(fb), kb, fw, chunk):
        out[i:j] = np.sqrt((d * d).sum(axis=1))
    return out


def pick_vocab_tensor(keys: list[str]) -> str | None:
    """Prefer an input-embedding-looking key, else the first vocab-dimension key."""
    if not keys:
        return None
    return next((k for k in keys if "embed" in k.lower()), keys[0])


def localize(ref_dir, der_dir, vocab_size: int | None = None, top: int = 20,
             fw: str | None = None, n_tensor_rows: int = 12) -> dict:
    """Full profile: per-tensor delta summary + per-token localization on a vocab-dimension tensor.

    Returns a dict with status 'ok' | 'not_applicable'. Never reads ground truth, never decides.
    """
    fw = fw or framework()
    ref, der = Path(ref_dir), Path(der_dir)
    ki_r, ki_d = key_index(ref), key_index(der)
    if not ki_r or not ki_d:
        which = "reference" if not ki_r else "derived"
        return {"status": "not_applicable",
                "reason": f"no .safetensors weight files in the {which} bundle; an adapter-only bundle "
                          "has no embedding delta by construction"}

    shared = sorted(set(ki_r) & set(ki_d))
    V = vocab_size or vocab_size_of(ref) or vocab_size_of(der)

    per_tensor, vocab_keys = {}, []
    for k in shared:
        st = tensor_delta_stats(ki_r[k], k, ki_d[k], k, fw)
        per_tensor[k] = st
        if st["status"] == "ok" and V and len(st["shape"]) == 2 and st["shape"][0] == V:
            vocab_keys.append(k)

    ok = {k: v for k, v in per_tensor.items() if v["status"] == "ok"}
    ranked = sorted(ok.items(), key=lambda kv: -kv[1]["rel_fro_delta"])
    res = {
        "status": "ok", "vocab_size": V,
        "n_tensors_shared": len(shared),
        "tensors_only_in_reference": sorted(set(ki_r) - set(ki_d)),
        "tensors_only_in_derived": sorted(set(ki_d) - set(ki_r)),
        "n_tensors_changed": sum(1 for v in ok.values() if v["rel_fro_delta"] > 0),
        "vocab_dimension_tensors": vocab_keys,
        "top_changed_tensors": [{"tensor": k, "rel_fro_delta": v["rel_fro_delta"],
                                 "frac_entries_changed": v["frac_entries_changed"],
                                 "max_abs_entry_delta": v["max_abs_entry_delta"], "shape": v["shape"]}
                                for k, v in ranked[:n_tensor_rows]],
        "per_tensor": per_tensor,
    }

    key = pick_vocab_tensor(vocab_keys)
    if key is None:
        res["token_localization"] = {"status": "no_vocab_dimension_tensor"}
        return res

    norms = row_delta_norms(ki_r[key], key, ki_d[key], key, fw)
    nz = norms[norms > 0]
    order = np.argsort(-norms)
    med_nz = float(np.median(nz)) if nz.size else 0.0
    total = float(norms.sum())
    k_mass = min(20, norms.size)
    tl = {
        "status": "ok", "tensor": key, "n_rows": int(norms.size),
        "n_rows_changed": int(nz.size),
        "frac_rows_changed": float(nz.size / norms.size) if norms.size else 0.0,
        "max_row_delta": float(norms.max()) if norms.size else 0.0,
        "median_row_delta": float(np.median(norms)) if norms.size else 0.0,
        "median_nonzero_row_delta": med_nz,
        "top1_over_median_nonzero": (float(norms.max() / med_nz) if med_nz > 0 else None),
        "topk_mass_fraction": (float(np.sort(norms)[-k_mass:].sum() / total) if total > 0 else None),
        "topk_mass_k": k_mass,
        "candidate_token_ids": [int(t) for t in order[:top] if norms[int(t)] > 0],
        "top_rows": [{"rank": i + 1, "token_id": int(t), "row_delta": float(norms[int(t)])}
                     for i, t in enumerate(order[:top]) if norms[int(t)] > 0],
    }
    if nz.size == 0:
        tl["note"] = ("vocab-dimension rows are identical to the reference: no lexical localization "
                      "signal (expected when the transformation leaves embeddings untouched)")
    res["token_localization"] = tl
    return res


# --------------------------------------------------------------------------------------
# Dense-component removal for tied-embedding models.
#
# Observed on the real SmolLM2 runs: a planted-trigger artifact changes 100% of embedding rows,
# not a handful, so a raw row-norm ranking puts the trigger at rank 17-681 of 49152 rather than
# near the top. The cause is `tie_word_embeddings: True` -- the embedding matrix IS the output
# head, so the full-vocabulary softmax places a nonzero gradient on EVERY row at every training
# step. For a token t that is not the training target, that update is approximately
#
#       delta_t  ~=  -lr * p_t * h_bar
#
# i.e. a SHARED direction h_bar (rank-1 in expectation) scaled by the token's predicted
# probability p_t. This is why the raw top-ranked tokens come out as frequent, high-probability
# tokens and output-distribution artifacts rather than the trigger. The input-side gradient on
# the trigger token's own row does NOT share that direction.
#
# So: project the delta rows onto the orthogonal complement of the leading principal
# direction(s) and rank by the residual norm, which suppresses the dense output-side term.
#
# DISCIPLINE ON CHOOSING k (how many directions to remove): k is NEVER selected by checking
# which value ranks a known trigger best -- that would be fitting to ground truth. Two
# ground-truth-free rules are provided, and callers are expected to pre-declare one:
#   K_THEORY = 1  : the dense tied-output term is rank-1 in expectation (derivation above).
#   select_k_by_energy(..., 0.90) : smallest k explaining >= 90% of total delta energy.
# `weight_delta_ksweep.py` reports the full sweep as a sensitivity analysis with the
# pre-declared k marked, so the trigger's rank is reported as an OUTCOME, not used as a selector.
# --------------------------------------------------------------------------------------

K_THEORY = 1


def gram(fa, ka: str, fb, kb: str, fw: str | None = None, chunk: int = 2048) -> np.ndarray:
    """Exact D^T D for the delta matrix D = (derived - reference), accumulated in row chunks.
    D is [vocab, hidden], so the Gram matrix is only [hidden, hidden] (576x576 here)."""
    fw = fw or framework()
    shape = _shape_of(Path(fa), ka, fw)
    h = int(np.prod(shape[1:])) if len(shape) > 1 else 1
    G = np.zeros((h, h), dtype=np.float64)
    for _i, _j, d in _iter_delta_chunks(Path(fa), ka, Path(fb), kb, fw, chunk):
        G += d.T @ d
    return G


def principal_directions(G: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-k eigenvectors of the Gram matrix (= right singular vectors of D), plus all
    eigenvalues in descending order. Eigenvalues are the squared singular values, so they
    sum to ||D||_F^2 and give the energy decomposition directly."""
    w, V = np.linalg.eigh(G)                 # ascending, symmetric
    order = np.argsort(-w)
    w, V = w[order], V[:, order]
    w = np.clip(w, 0.0, None)
    return V[:, :max(0, int(k))], w


def select_k_by_energy(eigvals: np.ndarray, threshold: float = 0.90) -> int:
    """Smallest k whose leading eigenvalues explain >= `threshold` of total delta energy.
    Ground-truth-free by construction: depends only on the delta's own spectrum."""
    tot = float(np.sum(eigvals))
    if tot <= 0:
        return 0
    c = np.cumsum(eigvals) / tot
    return int(np.searchsorted(c, threshold) + 1)


def residual_row_norms(fa, ka: str, fb, kb: str, k: int, fw: str | None = None,
                       chunk: int = 2048, V: np.ndarray | None = None,
                       eigvals: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Per-row norm of the delta AFTER projecting out the k leading principal directions.

    residual_t^2 = ||d_t||^2 - sum_{i<k} (d_t . v_i)^2
    With k == 0 this reduces exactly to `row_delta_norms`.
    """
    fw = fw or framework()
    if V is None or eigvals is None:
        G = gram(fa, ka, fb, kb, fw, chunk)
        V, eigvals = principal_directions(G, k)
    else:
        V = V[:, :max(0, int(k))]
    tot = float(np.sum(eigvals))
    info = {"k": int(k), "n_directions_available": int(eigvals.size),
            "energy_removed_fraction": (float(np.sum(eigvals[:int(k)]) / tot) if tot > 0 and k > 0 else 0.0),
            "eigval_energy_fractions": [float(x / tot) for x in eigvals[:8]] if tot > 0 else []}
    n0 = _shape_of(Path(fa), ka, fw)[0]
    out = np.zeros(n0, dtype=np.float64)
    for i, j, d in _iter_delta_chunks(Path(fa), ka, Path(fb), kb, fw, chunk):
        sq = (d * d).sum(axis=1)
        if V.size:
            proj = d @ V                                  # [chunk, k]
            sq = sq - (proj * proj).sum(axis=1)
        out[i:j] = np.sqrt(np.clip(sq, 0.0, None))
    return out, info


def localize_residual(ref_dir, der_dir, ks=(0, 1, 2, 4, 8), top: int = 20,
                      vocab_size: int | None = None, fw: str | None = None,
                      energy_threshold: float = 0.90) -> dict:
    """Rank tokens by residual row delta for each k in `ks`, sharing one Gram/eigendecomposition.

    Reports both pre-declared ground-truth-free choices of k (`K_THEORY` and the energy rule) so a
    caller can state which one it committed to BEFORE looking at any trigger rank.
    """
    fw = fw or framework()
    ki_r, ki_d = key_index(ref_dir), key_index(der_dir)
    if not ki_r or not ki_d:
        return {"status": "not_applicable", "reason": "no .safetensors weight files on one side"}
    V_ = vocab_size or vocab_size_of(ref_dir) or vocab_size_of(der_dir)
    cands = []
    for k_ in sorted(set(ki_r) & set(ki_d)):
        st = tensor_delta_stats(ki_r[k_], k_, ki_d[k_], k_, fw)
        if st["status"] == "ok" and V_ and len(st["shape"]) == 2 and st["shape"][0] == V_:
            cands.append(k_)
    key = pick_vocab_tensor(cands)
    if key is None:
        return {"status": "no_vocab_dimension_tensor"}

    G = gram(ki_r[key], key, ki_d[key], key, fw)
    kmax = max(list(ks) + [1])
    Vfull, eigvals = principal_directions(G, kmax)
    tot = float(np.sum(eigvals))
    if tot <= 0:
        return {"status": "ok", "tensor": key, "no_delta": True, "by_k": {},
                "k_theory": K_THEORY, "k_by_energy": 0,
                "note": "vocab-dimension tensor is identical to the reference; nothing to decompose"}

    out = {"status": "ok", "tensor": key, "no_delta": False,
           "k_theory": K_THEORY, "k_by_energy": select_k_by_energy(eigvals, energy_threshold),
           "energy_threshold": energy_threshold,
           "eigval_energy_fractions": [float(x / tot) for x in eigvals[:8]], "by_k": {}}
    for k in ks:
        norms, info = residual_row_norms(ki_r[key], key, ki_d[key], key, k, fw,
                                         V=Vfull, eigvals=eigvals)
        order = np.argsort(-norms)
        nz = norms[norms > 0]
        med = float(np.median(nz)) if nz.size else 0.0
        out["by_k"][int(k)] = {
            **info,
            "max_row_delta": float(norms.max()) if norms.size else 0.0,
            "median_nonzero_row_delta": med,
            "top1_over_median_nonzero": (float(norms.max() / med) if med > 0 else None),
            "candidate_token_ids": [int(t) for t in order[:top]],
            "top_rows": [{"rank": i + 1, "token_id": int(t), "row_delta": float(norms[int(t)])}
                         for i, t in enumerate(order[:top])],
            "_order": order,              # kept in-memory for exact rank lookup; strip before JSON
        }
    return out


def ranks_in_residual(res: dict, k: int, token_ids: list[int]) -> dict:
    """Exact ranks of specific token ids under the residual ordering for a given k.
    REPORTING ONLY -- `localize_residual` itself never sees a token id."""
    by_k = (res or {}).get("by_k", {}).get(int(k))
    if not by_k or "_order" not in by_k:
        return {}
    order = by_k["_order"]
    pos = {int(t): i + 1 for i, t in enumerate(order)}
    return {int(t): pos.get(int(t)) for t in token_ids}


def strip_internal(res: dict) -> dict:
    """Remove non-JSON-serializable working arrays before writing results out."""
    for v in (res or {}).get("by_k", {}).values():
        v.pop("_order", None)
    return res


# --------------------------------------------------------------------------------------
# COMPONENT-LEVEL (intra-weight) DIFFERENTIAL, calibrated against the honest family.
#
# This is the part that survived contact with real data. Measured over all 31 artifacts of two
# independent real SmolLM2 runs: every honest derivation of the declared family (12 calibration +
# 12 held-out), all three canary perturbations, the template tamper and the generation-config
# tamper changed 210 tensors and ZERO vocabulary rows; both planted-trigger artifacts changed 272
# tensors and 100% of the 49152 vocabulary rows. 24 honest derivations, no false positives, and it
# flags exactly the two artifacts the behavioral pipeline missed in two separate experiments.
#
# Why this is a legitimate BackdoorLens statistic and not a bolt-on: bundle_diff.py already does a
# component differential at the BUNDLE level (tokenizer, template, config, generation, runtime).
# This extends T1 one level down, INSIDE the weight files, and calibrates it the same way everything
# else here is calibrated -- against honest derivations of the same declared transformation family.
# The trigger artifacts declare ["weights","quantization"] exactly like the honest ones, so this is
# NOT the undeclared-change check: it is a within-family anomaly, same declared transformation and
# same reference, where every honest member leaves a tensor group untouched and this one rewrites it.
#
# HONEST STATISTICAL FRAMING (do not overstate this in the paper): the honest null here is
# DEGENERATE -- identically zero for the never-touched groups. There is no variance, so no t
# prediction interval and no p-value can be computed, and none is claimed. This is a component /
# scope-level finding ("the audited artifact modifies a tensor group that no honest derivation of
# this family modifies"), reported with the number of honest derivations backing it. False-positive
# behavior is checked by leave-one-out over the honest population, as elsewhere in this project.
#
# What it does NOT do: it needs full weights on both sides (not applicable to adapter-only bundles),
# and an attacker who confines a tamper to the same tensor groups the honest transformation already
# touches produces no anomaly here. It raises attacker cost; it is not a universal detector.
# --------------------------------------------------------------------------------------

def tensor_change_profile(ref_dir, der_dir, tol: float = 0.0, fw: str | None = None) -> dict:
    """Per-tensor-key relative Frobenius delta for every shared tensor, plus which keys changed.

    tol defaults to 0.0, i.e. ANY nonzero change counts as changed. That is what produced the clean
    separation on the real runs (honest derivations leave the skipped groups bit-identical, since
    fake_quantize_state skips them entirely). A pipeline whose honest transformation perturbs every
    tensor slightly would need a non-zero tol, and the honest range recorded by fit_component_null
    is what should set it -- never a hand-picked constant.
    """
    fw = fw or framework()
    ki_r, ki_d = key_index(ref_dir), key_index(der_dir)
    if not ki_r or not ki_d:
        which = "reference" if not ki_r else "derived"
        return {"status": "not_applicable",
                "reason": f"no .safetensors weight files in the {which} bundle"}
    shared = sorted(set(ki_r) & set(ki_d))
    rel, changed, unsupported = {}, [], {}
    for k in shared:
        st = tensor_delta_stats(ki_r[k], k, ki_d[k], k, fw)
        if st["status"] != "ok":
            unsupported[k] = st["status"]
            continue
        rel[k] = float(st["rel_fro_delta"])
        if st["rel_fro_delta"] > tol:
            changed.append(k)
    return {"status": "ok", "tol": tol, "n_tensors_shared": len(shared),
            "rel_fro_delta": rel, "changed_keys": sorted(changed),
            "n_tensors_changed": len(changed),
            "tensors_only_in_reference": sorted(set(ki_r) - set(ki_d)),
            "tensors_only_in_derived": sorted(set(ki_d) - set(ki_r)),
            "unsupported": unsupported}


def fit_component_null(ref_dir, honest_dirs: dict, tol: float = 0.0, fw: str | None = None,
                       log=None) -> dict:
    """Calibrate which tensor groups honest derivations of this family DO and DO NOT modify.

    honest_dirs: {name: bundle_dir}. These are the independent experimental units, exactly as for
    the behavioral null -- one honest derivation is one unit, never one tensor.
    """
    fw = fw or framework()
    members, profiles = [], {}
    for name, d in honest_dirs.items():
        p = tensor_change_profile(ref_dir, d, tol, fw)
        if p["status"] != "ok":
            continue
        members.append(name)
        profiles[name] = p
        if log:
            log(f"   honest {name}: n_tensors_changed={p['n_tensors_changed']}")
    if not members:
        return {"status": "no_usable_honest_derivations"}

    all_keys = sorted(set().union(*[set(profiles[m]["rel_fro_delta"]) for m in members]))
    per_key = {}
    for k in all_keys:
        vals = [profiles[m]["rel_fro_delta"].get(k) for m in members]
        vals = [v for v in vals if v is not None]
        n_ch = sum(1 for v in vals if v > tol)
        per_key[k] = {"n_honest": len(vals), "n_honest_changed": n_ch,
                      "min_rel": float(min(vals)) if vals else None,
                      "max_rel": float(max(vals)) if vals else None}
    never = sorted(k for k, v in per_key.items() if v["n_honest_changed"] == 0)
    counts = [profiles[m]["n_tensors_changed"] for m in members]
    return {"status": "ok", "tol": tol, "n_honest_derivations": len(members), "members": members,
            "per_key": per_key, "never_changed_keys": never,
            "n_never_changed_keys": len(never),
            "n_tensors_changed_honest": {"min": int(min(counts)), "max": int(max(counts)),
                                         "values": [int(c) for c in counts]}}


def component_anomalies(ref_dir, der_dir, null: dict, fw: str | None = None) -> dict:
    """Which tensor groups does the audited artifact modify that NO honest derivation modifies?

    Returns `hard_anomalies` (a group untouched by every honest member of the family -- the signal
    that separated perfectly on the real runs) and, separately and more weakly, a changed-tensor
    count outside the honest range. No verdict is produced here; decision.py owns that.
    """
    if (null or {}).get("status") != "ok":
        return {"status": "no_component_null"}
    p = tensor_change_profile(ref_dir, der_dir, null.get("tol", 0.0), fw)
    if p["status"] != "ok":
        return {"status": p["status"], "reason": p.get("reason")}
    never = set(null["never_changed_keys"])
    hard = sorted(k for k in p["changed_keys"] if k in never)
    rng = null["n_tensors_changed_honest"]
    return {
        "status": "ok",
        "n_honest_derivations": null["n_honest_derivations"],
        "n_tensors_changed": p["n_tensors_changed"],
        "n_tensors_changed_honest_range": [rng["min"], rng["max"]],
        "count_outside_honest_range": not (rng["min"] <= p["n_tensors_changed"] <= rng["max"]),
        "hard_anomalies": hard,
        "n_hard_anomalies": len(hard),
        "hard_anomaly_rel_fro": {k: p["rel_fro_delta"][k] for k in hard[:40]},
        "note": ("a 'hard anomaly' is a tensor group that NO honest derivation of this declared "
                 "family modifies but the audited artifact does. The honest null for such a group "
                 "is degenerate (identically zero), so this is a component-level finding backed by "
                 f"{null['n_honest_derivations']} honest derivations -- not a p-value."),
    }


def leave_one_out_component_fpr(ref_dir, honest_dirs: dict, tol: float = 0.0,
                                fw: str | None = None, log=None) -> dict:
    """False-positive check: for each honest derivation, fit the component null on the OTHERS and
    audit the held-out one. Any hard anomaly here is a false positive."""
    fw = fw or framework()
    names = list(honest_dirs)
    rows, n_fp = [], 0
    for held in names:
        rest = {n: d for n, d in honest_dirs.items() if n != held}
        if not rest:
            continue
        nul = fit_component_null(ref_dir, rest, tol, fw)
        an = component_anomalies(ref_dir, honest_dirs[held], nul, fw)
        n_hard = int(an.get("n_hard_anomalies", 0)) if an.get("status") == "ok" else None
        if n_hard:
            n_fp += 1
        rows.append({"held_out": held, "n_fit_on": len(rest), "n_hard_anomalies": n_hard,
                     "hard_anomalies": an.get("hard_anomalies", [])[:10],
                     "count_outside_honest_range": an.get("count_outside_honest_range")})
        if log:
            log(f"   LOO {held}: hard_anomalies={n_hard}")
    return {"status": "ok", "n_tested": len(rows), "n_false_positives": n_fp,
            "false_positive_rate": (n_fp / len(rows)) if rows else None, "rows": rows}


def exact_ranks(ref_dir, der_dir, token_ids: list[int], tensor_key: str | None = None,
                vocab_size: int | None = None, fw: str | None = None) -> dict:
    """REPORTING ONLY, deliberately separate from `localize`: the exact rank of specific token ids in
    the full descending row-delta ordering (so a known ground-truth trigger can be reported honestly
    even when it ranks 5000th, instead of just 'not in the top-k').

    This is kept out of `localize` on purpose: `localize` -- the part that could ever feed a detector
    -- never takes a token id as input, so its candidate list cannot be influenced by ground truth.
    Costs one extra pass over the vocab-dimension tensor.
    """
    fw = fw or framework()
    ki_r, ki_d = key_index(ref_dir), key_index(der_dir)
    if not ki_r or not ki_d:
        return {"status": "not_applicable"}
    V = vocab_size or vocab_size_of(ref_dir) or vocab_size_of(der_dir)
    key = tensor_key
    if key is None:
        cands = [k for k in sorted(set(ki_r) & set(ki_d))
                 if (st := tensor_delta_stats(ki_r[k], k, ki_d[k], k, fw))["status"] == "ok"
                 and V and len(st["shape"]) == 2 and st["shape"][0] == V]
        key = pick_vocab_tensor(cands)
    if key is None or key not in ki_r or key not in ki_d:
        return {"status": "no_vocab_dimension_tensor"}
    norms = row_delta_norms(ki_r[key], key, ki_d[key], key, fw)
    order = np.argsort(-norms)
    pos = {int(t): i + 1 for i, t in enumerate(order)}
    return {"status": "ok", "tensor": key, "n_rows": int(norms.size),
            "ranks": {int(t): pos.get(int(t)) for t in token_ids},
            "row_deltas": {int(t): float(norms[int(t)]) for t in token_ids
                           if 0 <= int(t) < norms.size}}
