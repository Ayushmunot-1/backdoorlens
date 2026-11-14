"""Paired behavioral metrics between reference and derived next-token distributions."""
from __future__ import annotations
import difflib
import json
import re
import numpy as np
from scipy.special import logsumexp, xlogy


def normalize_logprobs(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x - logsumexp(x)


def jsd(lp: np.ndarray, lq: np.ndarray) -> float:
    """Jensen-Shannon divergence in bits, in [0, 1]. Symmetric, always finite."""
    p, q = np.exp(lp), np.exp(lq)
    m = 0.5 * (p + q)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl_pm = np.sum(xlogy(p, p) - xlogy(p, m))
        kl_qm = np.sum(xlogy(q, q) - xlogy(q, m))
    val = 0.5 * (kl_pm + kl_qm) / np.log(2.0)
    return float(min(1.0, max(0.0, val)))


def kl(lp: np.ndarray, lq: np.ndarray) -> float:
    """KL(P||Q) in nats; finite because both come from softmax."""
    p = np.exp(lp)
    return float(max(0.0, np.sum(p * (lp - lq))))


def topk_overlap(lp: np.ndarray, lq: np.ndarray, k: int = 10) -> float:
    k = min(k, len(lp) - 1)
    a = set(np.argpartition(-lp, k)[:k].tolist())
    b = set(np.argpartition(-lq, k)[:k].tolist())
    return len(a & b) / len(a | b)


def topk_rank_corr(lp: np.ndarray, lq: np.ndarray, k: int = 10) -> float:
    """Spearman correlation of the two models' scores on the reference's top-k tokens."""
    from scipy.stats import spearmanr
    top = np.argsort(-lp)[:k]
    a, b = lp[top], lq[top]
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return 1.0 if np.array_equal(np.argsort(a, kind="stable"), np.argsort(b, kind="stable")) else 0.0
    r = spearmanr(a, b).statistic
    return float(0.0 if np.isnan(r) else r)


def delta_margin(lp: np.ndarray, lq: np.ndarray) -> float:
    """Change of the reference top-1 vs top-2 log-prob margin."""
    t = np.argsort(-lp)[:2]
    return float(abs((lq[t[0]] - lq[t[1]]) - (lp[t[0]] - lp[t[1]])))


def nll_shift(lp: np.ndarray, lq: np.ndarray) -> float:
    t = int(np.argmax(lp))
    return float((-lq[t]) - (-lp[t]))


class IncomparableLogits(ValueError):
    """The two distributions are over different-sized supports, so no divergence between them exists.

    Raised instead of letting numpy report an opaque broadcast failure from inside a metric. In
    practice this means the reference and derived bundles have different vocabulary sizes, i.e. they
    are not a reference/derived pair of one model -- a scope error, which `scope.comparability` is
    supposed to catch from the manifests before any model is loaded. Reaching here means that check
    was skipped or the config understated the real vocabulary.
    """


def paired_metrics(lp: np.ndarray, lq: np.ndarray) -> dict:
    if lp.shape != lq.shape:
        raise IncomparableLogits(
            f"reference logits have shape {lp.shape} and derived logits {lq.shape}: these bundles are "
            f"not comparable (different vocabulary size), so no paired divergence is defined. Audit "
            f"them against the reference they were actually derived from.")
    return {
        "jsd": jsd(lp, lq),
        "kl": kl(lp, lq),
        "topk_overlap": topk_overlap(lp, lq),
        "topk_rank_corr": topk_rank_corr(lp, lq),
        "delta_margin": delta_margin(lp, lq),
        "nll_shift": nll_shift(lp, lq),
        "argmax_flip": bool(np.argmax(lp) != np.argmax(lq)),
    }


_REFUSAL = re.compile(r"\b(i can't|i cannot|i can not|i won't|i'm sorry|i am sorry|unable to (help|assist))\b", re.I)


def structural_label(text: str) -> dict:
    t = text.strip()
    is_json = False
    if t.startswith(("{", "[")):
        try:
            json.loads(t)
            is_json = True
        except Exception:
            is_json = False
    return {"empty": len(t) == 0, "refusal": bool(_REFUSAL.search(t)), "json_valid": is_json}


def structural_flip(a: str, b: str) -> bool:
    la, lb = structural_label(a), structural_label(b)
    return any(la[k] != lb[k] for k in la)


def edit_distance_ratio(a: str, b: str) -> float:
    """1 - similarity ratio in [0,1]; 0 means identical."""
    return 1.0 - difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()
