"""Tests for the spike-ratio statistic added to fix a real measured gap: a sparse/rare conditional
effect (e.g. a trigger that fires on only 1-2 of 128 probe prompts) can be diluted below the
materiality threshold of the suite-wide median ("level") test while still being a genuine, reproducible
behavioral difference. The spike statistic (`max_jsd / level` per derivation) targets exactly this case
and is calibrated with the SAME method (t prediction interval, df = k-1) already used for `level`, so no
new statistical machinery is introduced. No torch needed - this exercises calibration.py and decision.py
directly.
"""
import numpy as np
import pytest
from backdoorlens.calibration import NullDistribution, level_of
from backdoorlens.decision import Policy, decide


def _null(scores_by_deriv):
    return NullDistribution(spec={}, key="k", search_budget=10, numeric_floor=1e-6,
                            scores_by_deriv=scores_by_deriv)


def test_spikes_matches_manual_max_over_level():
    rng = np.random.default_rng(0)
    sbd = {f"h{i}": list(rng.normal(0.05, 0.005, 128).clip(1e-6, None)) for i in range(6)}
    n = _null(sbd)
    sp = n.spikes()
    assert sp.shape == (6,)
    for (name, v), ratio in zip(sbd.items(), sp):
        assert ratio == pytest.approx(max(v) / level_of(v))
    # all honest derivations have similar, bounded max/level ratios (no single artifact dominates)
    assert sp.max() < 3.0


def test_spikes_excludes_named_derivation_like_levels():
    sbd = {"a": [0.1, 0.2, 0.9], "b": [0.1, 0.1, 0.3]}
    n = _null(sbd)
    assert n.spikes(exclude="a").shape == (1,)
    assert n.spikes(exclude="a")[0] == pytest.approx(0.3 / level_of(sbd["b"]))


def _base_ev(sa_overrides, search_overrides=None, confirmation=()):
    sa = {"n": 128, "level_median_jsd": 0.3, "mean_jsd": 0.3, "max_jsd": 0.3, "spike_ratio": 1.0,
          "argmax_flip_rate": 0.0, "p_level": None, "level_ratio_vs_max_honest": None,
          "p_binom": None, "p_minbonf": None, "frac_exceed": None, "k_exceed": None,
          "p_spike": None, "spike_ratio_vs_max_honest": None}
    sa.update(sa_overrides)
    se = {"strategy": "guided", "budget": 10, "budget_used": 10, "best_jsd": sa["level_median_jsd"],
          "best_ratio": 1.0, "p_search": None, "p_search_pooled_descriptive": None,
          "level": sa["level_median_jsd"], "curve": []}
    if search_overrides:
        se.update(search_overrides)
    return {
        "completeness": "complete", "scope_gaps": [],
        "bundle": {"n_behavior_diffs": 0, "n_undeclared_behavior_diffs": 0,
                   "n_rendered_mismatches": 0, "n_render_errors": 0},
        "null": {"n_derivations": 6, "n_prompt_scores": 768, "search_budget": 10, "search_ratio_max": 3.0},
        "null_note": None, "numeric_floor": 0.01,
        "stage_a": sa, "output_level": {}, "search": se, "confirmation": list(confirmation),
    }


def _lenient_policy(**kw):
    return Policy(min_null_derivations=2, min_null_prompts=10, min_budget=10, **kw)


def test_diluted_spike_is_caught_when_level_test_alone_would_miss_it():
    """Mirrors the real-model finding: suite-wide median is only ~1.3-1.7x the largest honest level
    (fails the 2x materiality guard) but the single worst prompt is a clear, calibrated outlier."""
    ev = _base_ev({"p_level": 0.05, "level_ratio_vs_max_honest": 1.5,   # level test alone: NOT significant enough
                   "p_spike": 5e-4, "spike_ratio_vs_max_honest": 2.5,   # spike test: significant and material
                   "max_jsd": 0.75, "spike_ratio": 2.5})
    dec = decide(ev, _lenient_policy())
    assert dec.verdict == "BEHAVIORALLY_DIVERGENT"
    assert any("spike" in r for r in dec.reasons)


def test_level_only_signal_still_works_unchanged():
    """Regression: the pre-existing level-only detection path (suite-wide shift) is untouched."""
    ev = _base_ev({"p_level": 5e-4, "level_ratio_vs_max_honest": 2.5,
                   "p_spike": 0.2, "spike_ratio_vs_max_honest": 1.1})
    dec = decide(ev, _lenient_policy())
    assert dec.verdict == "BEHAVIORALLY_DIVERGENT"
    assert any("level shift" in r for r in dec.reasons)


def test_honest_like_evidence_is_not_flagged_by_spike_test():
    """The new test must not inflate false positives: an artifact whose max/level ratio looks like
    ordinary honest-derivation noise must not be flagged."""
    ev = _base_ev({"p_level": 0.4, "level_ratio_vs_max_honest": 1.0,
                   "p_spike": 0.35, "spike_ratio_vs_max_honest": 1.05,
                   "max_jsd": 0.3, "spike_ratio": 1.0})
    dec = decide(ev, _lenient_policy())
    assert dec.verdict in ("ATTESTED", "ATTESTED_WITH_DRIFT")


def test_spike_guard_requires_both_significance_and_effect_size():
    """A tiny p-value with a sub-threshold ratio (or vice versa) must not fire - same discipline as
    the existing level test's min_level_ratio guard."""
    ev = _base_ev({"p_spike": 1e-6, "spike_ratio_vs_max_honest": 1.2,   # significant but not material
                   "p_level": 0.4, "level_ratio_vs_max_honest": 1.0})
    dec = decide(ev, _lenient_policy())
    assert dec.verdict not in ("BEHAVIORALLY_DIVERGENT",)
    ev2 = _base_ev({"p_spike": 0.3, "spike_ratio_vs_max_honest": 5.0,   # material but not significant
                   "p_level": 0.4, "level_ratio_vs_max_honest": 1.0})
    assert decide(ev2, _lenient_policy()).verdict not in ("BEHAVIORALLY_DIVERGENT",)


def test_min_spike_ratio_defaults_to_same_conservative_value_as_level():
    """Documents the deliberate choice: the new guard is NOT tuned to any observed trigger/canary
    data - it reuses the same default as the pre-existing, independently-chosen min_level_ratio."""
    assert Policy().min_spike_ratio == Policy().min_level_ratio == 2.0
