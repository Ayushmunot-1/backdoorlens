"""The intra-weight component anomaly must reach the verdict, and must be completely inert when no
component null was supplied (so every pre-existing run is unaffected).

Context: on the real SmolLM2 runs this statistic was the only thing that separated the two planted-
trigger artifacts from all 29 others -- they modify 62 tensor groups (token embedding, final norm, all
60 layernorms) that none of 12 honest derivations of the declared family modifies, 0/12 leave-one-out
false positives. These tests pin the decision semantics of that evidence.
"""
from __future__ import annotations

import copy

from backdoorlens.decision import Policy, decide


def _clean_evidence() -> dict:
    """Evidence that on its own yields a pass: no gaps, nothing significant, no bundle diffs."""
    return {
        "completeness": "complete",
        "scope_gaps": [],
        "bundle": {"n_behavior_diffs": 0, "n_undeclared_behavior_diffs": 0,
                   "n_rendered_mismatches": 0, "n_render_errors": 0},
        "null": {"n_derivations": 6, "n_prompt_scores": 128, "n_search_maxima": 36,
                 "search_budget": 128, "search_ratio_max": 2.0, "q95": 0.01, "level_max": 0.03,
                 "spike_ratio_max": 12.0},
        "null_note": None,
        "numeric_floor": 1e-9,
        "stage_a": {"level_median_jsd": 0.01, "p_level": 0.9, "level_ratio_vs_max_honest": 0.3,
                    "spike_ratio": 4.0, "p_spike": 0.9, "spike_ratio_vs_max_honest": 0.3,
                    "p_binom": 0.9, "p_minbonf": 0.9, "max_jsd": 0.02, "frac_above_q95": 0.0},
        "search": {"best_jsd": 0.02, "best_ratio": 1.0, "p_search": 0.9, "budget": 128,
                   "budget_used": 128},
        "confirmation": [],      # list of counterexample records; none confirmed persistent
    }


def _anomaly(n: int) -> dict:
    return {"status": "ok", "n_honest_derivations": 12, "n_hard_anomalies": n,
            "hard_anomalies": ["model.embed_tokens.weight", "model.norm.weight",
                               "model.layers.0.input_layernorm.weight"][:max(n, 0)] or [],
            "n_tensors_changed": 272, "n_tensors_changed_honest_range": [210, 210],
            "count_outside_honest_range": True}


def test_absent_component_anomaly_changes_nothing():
    """No component null supplied -> the key is absent -> verdict is whatever it was before."""
    ev = _clean_evidence()
    assert "component_anomaly" not in ev
    d = decide(ev, Policy())
    assert d.verdict in ("ATTESTED", "ATTESTED_WITH_DRIFT")
    assert not any("tensor group" in r for r in d.reasons)


def test_hard_anomaly_forces_suspicious():
    ev = _clean_evidence()
    ev["component_anomaly"] = _anomaly(3)
    d = decide(ev, Policy())
    assert d.verdict == "SUSPICIOUS"
    assert d.action == "QUARANTINE"
    assert any("honest derivation" in r and "declared transformation family" in r for r in d.reasons)
    assert any("12 honest derivation" in r for r in d.reasons)


def test_zero_anomalies_is_a_pass():
    ev = _clean_evidence()
    ev["component_anomaly"] = _anomaly(0)
    d = decide(ev, Policy())
    assert d.verdict in ("ATTESTED", "ATTESTED_WITH_DRIFT")


def test_policy_flag_can_disable_enforcement_without_losing_the_evidence():
    ev = _clean_evidence()
    ev["component_anomaly"] = _anomaly(62)
    d = decide(ev, Policy(component_anomaly_is_suspicious=False))
    assert d.verdict in ("ATTESTED", "ATTESTED_WITH_DRIFT")
    # the evidence itself is untouched, so it still reaches the attestation for the record
    assert ev["component_anomaly"]["n_hard_anomalies"] == 62


def test_non_ok_component_status_is_ignored_not_treated_as_a_finding():
    """A component null that could not be applied (adapter-only bundle, unreadable file) must not
    silently become an anomaly, and must not crash the decision."""
    for st in ({"status": "not_applicable", "reason": "adapter-only"},
               {"status": "no_component_null"},
               {"status": "error", "error": "boom"}):
        ev = _clean_evidence()
        ev["component_anomaly"] = st
        d = decide(ev, Policy())
        assert d.verdict in ("ATTESTED", "ATTESTED_WITH_DRIFT")
        assert not any("tensor group" in r for r in d.reasons)


def test_run_audit_wires_the_component_null_through_to_the_verdict(world):
    """End-to-end through run_audit on the toy backend: a component null is loaded, anomalies land in
    the attestation evidence, and the verdict reflects them. Note the toy quantizer skips only
    ("b1","b2"), so for THIS family the never-touched groups are the biases -- not the embedding as
    in the HF/SmolLM2 runs. The statistic is family-relative by construction, which is the point."""
    import json as _json
    from backdoorlens import weight_delta as wd
    from backdoorlens.audit import AuditConfig, run_audit
    from backdoorlens.backends import toy

    d = world["dir"]
    honest = {f"h{o}": d / f"h{o}" for o in range(6)}
    cnull = wd.fit_component_null(d / "ref", honest)
    assert cnull["status"] == "ok"
    assert "b1" in cnull["never_changed_keys"]          # quantize_rtn(skip=("b1","b2"))
    assert "E" not in cnull["never_changed_keys"]       # the toy quantizer DOES touch embeddings
    cpath = d / "component_null.json"
    cpath.write_text(_json.dumps(cnull))

    Q = toy.quantize_rtn(world["P"], 4, 32, 11)
    tampered = world["mk"]("comp_tamper", {**Q, "b1": Q["b1"] + 0.05})

    cfg = AuditConfig(budget=128, n_seeds=128, model_family="toy-0", component_null_path=str(cpath))
    att = run_audit(d / "ref", tampered, cfg, Policy(), world["null"])
    ca = att["evidence"]["component_anomaly"]
    assert ca["status"] == "ok"
    assert "b1" in ca["hard_anomalies"] and ca["n_hard_anomalies"] >= 1
    assert ca["n_honest_derivations"] == 6
    assert att["decision"]["verdict"] not in ("ATTESTED", "ATTESTED_WITH_DRIFT")
    assert any("declared transformation family" in r for r in att["decision"]["reasons"])

    # same audit, no component null supplied -> the key must be absent and nothing added to reasons
    att2 = run_audit(d / "ref", tampered,
                     AuditConfig(budget=128, n_seeds=128, model_family="toy-0"), Policy(),
                     world["null"])
    assert "component_anomaly" not in att2["evidence"]
    assert not any("declared transformation family" in r for r in att2["decision"]["reasons"])


def test_component_null_accepts_the_full_audit_payload_shape(world, tmp_path):
    """scripts/component_null_audit.py writes the null NESTED under "component_null". Passing that
    file straight to --component-null must work, not silently no-op."""
    import json as _json
    from backdoorlens import weight_delta as wd
    from backdoorlens.audit import AuditConfig, run_audit
    from backdoorlens.backends import toy

    d = world["dir"]
    cnull = wd.fit_component_null(d / "ref", {f"h{o}": d / f"h{o}" for o in range(6)})
    nested = tmp_path / "component_null_audit.json"
    nested.write_text(_json.dumps({"reference": str(d / "ref"), "component_null": cnull,
                                   "leave_one_out": {"n_false_positives": 0}, "audits": {}}))

    Q = toy.quantize_rtn(world["P"], 4, 32, 11)
    tampered = world["mk"]("comp_tamper_nested", {**Q, "b1": Q["b1"] + 0.05})
    att = run_audit(d / "ref", tampered,
                    AuditConfig(budget=128, n_seeds=128, model_family="toy-0",
                                component_null_path=str(nested)), Policy(), world["null"])
    ca = att["evidence"]["component_anomaly"]
    assert ca["status"] == "ok" and "b1" in ca["hard_anomalies"]


def test_unreadable_component_null_does_not_kill_the_audit(world, tmp_path):
    """A missing or corrupt component null must degrade to a recorded error, not an exception."""
    from backdoorlens.audit import AuditConfig, run_audit
    d = world["dir"]
    bad = tmp_path / "not_json.json"
    bad.write_text("{ this is not json")
    cfg = AuditConfig(budget=128, n_seeds=128, model_family="toy-0", component_null_path=str(bad))
    att = run_audit(d / "ref", d / "h0", cfg, Policy(), world["null"])
    assert att["evidence"]["component_anomaly"]["status"] == "error"
    assert att["decision"]["verdict"] in ("ATTESTED", "ATTESTED_WITH_DRIFT")


def test_component_anomaly_never_reaches_attested():
    """Whatever else passes, an artifact outside its declared family must not be certified."""
    ev = _clean_evidence()
    ev["component_anomaly"] = _anomaly(62)
    for pol in (Policy(), Policy(min_level_ratio=99.0), Policy(watch_alpha=0.5)):
        assert decide(copy.deepcopy(ev), pol).verdict not in ("ATTESTED", "ATTESTED_WITH_DRIFT")
