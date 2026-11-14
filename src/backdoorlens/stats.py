"""Small, dependency-light statistics used by calibration and decisions."""
from __future__ import annotations
import numpy as np
from scipy import stats as sps


def conformal_pvalues(null_scores, test_scores) -> np.ndarray:
    """Conformal p-value per test score: (1 + #{null >= s}) / (N + 1).

    Valid (super-uniform) if test scores are exchangeable with null scores under H0.
    Smallest attainable p is 1/(N+1): the null sample size bounds resolution.
    """
    null = np.sort(np.asarray(null_scores, dtype=float))
    test = np.asarray(test_scores, dtype=float)
    n = len(null)
    idx = np.searchsorted(null, test, side="left")
    ge = n - idx
    return (1.0 + ge) / (n + 1.0)


def empirical_pvalue_max(null_maxima, observed_max: float) -> float:
    """p-value of an observed search-maximum against replicated honest search-maxima."""
    null = np.asarray(null_maxima, dtype=float)
    return float((1.0 + np.sum(null >= observed_max)) / (len(null) + 1.0))


def binom_exceed_pvalue(k: int, n: int, q: float) -> float:
    """One-sided binomial test: are more than a (1-q) fraction of cases above the q-quantile?"""
    if n <= 0:
        return 1.0
    return float(sps.binomtest(int(k), int(n), p=1.0 - q, alternative="greater").pvalue)


def bonferroni_min_p(pvals) -> float:
    p = np.asarray(pvals, dtype=float)
    if p.size == 0:
        return 1.0
    return float(min(1.0, p.min() * p.size))


def bootstrap_ci(values, stat=np.mean, n_boot=2000, alpha=0.05, seed=0):
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boots = np.array([stat(v[rng.integers(0, v.size, v.size)]) for _ in range(n_boot)])
    return (float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2)))


def clopper_pearson(k: int, n: int, alpha: float = 0.05):
    """Exact binomial CI. With 0 errors in n trials the upper bound is about 3/n."""
    if n == 0:
        return (0.0, 1.0)
    lo = 0.0 if k == 0 else sps.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else sps.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return (float(lo), float(hi))


def ks_distance(a, b):
    r = sps.ks_2samp(np.asarray(a, float), np.asarray(b, float))
    return float(r.statistic), float(r.pvalue)


def level_prediction_pvalue(honest_levels, candidate_level: float) -> float | None:
    """One-sided p-value that a NEW derivation's overall drift level exceeds the honest population.

    Uses a t prediction interval on log10(level) across k honest derivations (df = k-1), so the
    derivation-to-derivation variance is accounted for. Returns None if k < 2.
    """
    h = np.log10(np.maximum(np.asarray(honest_levels, float), 1e-12))
    k = h.size
    if k < 2:
        return None
    sd = max(float(h.std(ddof=1)), 1e-3)
    t = (np.log10(max(candidate_level, 1e-12)) - float(h.mean())) / (sd * np.sqrt(1.0 + 1.0 / k))
    return float(1.0 - sps.t.cdf(t, k - 1))


def search_prediction_pvalue(maxima_by_deriv, observed: float, sd_floor: float = 0.01) -> float | None:
    """Derivation-aware one-sided p-value for a searched maximum (relative scale).

    Replicate searches on the SAME honest derivation are not independent evidence about a NEW derivation,
    so pooling them (pseudo-replication) understates variance. Model log(max) = mu + u_derivation + e and
    use a t prediction interval with df = k - 1 over the k honest derivations:
        Var(new single run) = s_b^2 + s_w^2,   SE(grand mean)^2 = (s_b^2 + s_w^2 / R) / k.
    Parametric (log-normal) assumption: validated by leave-one-derivation-out false-alarm rates, not assumed.
    """
    groups = [np.log(np.maximum(np.asarray(g, float), 1e-12)) for g in maxima_by_deriv if len(g) > 0]
    k = len(groups)
    if k < 3:
        return None
    means = np.array([g.mean() for g in groups])
    R = float(np.mean([len(g) for g in groups]))
    if R > 1:
        sw2 = float(np.mean([g.var(ddof=1) for g in groups if len(g) > 1]))
    else:
        sw2 = 0.0
    sb2 = max(float(means.var(ddof=1)) - sw2 / R, 0.0)
    var_new = max(sb2 + sw2, sd_floor ** 2)
    se2 = (sb2 + sw2 / R) / k
    t = (np.log(max(observed, 1e-12)) - float(means.mean())) / np.sqrt(var_new + se2)
    return float(1.0 - sps.t.cdf(t, k - 1))
