"""A null that does not match must say WHICH field is wrong, and the CLI must not force the caller
to restate what the null already records.

Real failure this pins: an attest run against `null_int4.json` used `--model-family smollm2-135m`
while the null had been fitted as `smollm2-135m-instruct`. The only output was
`INCONCLUSIVE / "no benign null matches this transformation family"` -- indistinguishable from a
genuinely wrong-family null, with nothing pointing at the typo. The CLI default for --budget (256)
also silently disagreed with the null's 128.
"""
from __future__ import annotations

from backdoorlens.calibration import SPEC_FIELDS, make_spec, spec_key, spec_mismatch


def _spec(**over):
    base = {"model_family": "smollm2-135m-instruct", "tool": "backdoorlens-sim",
            "method": "rtn-simulated", "bits": 4, "group_size": 128,
            "calibration_dataset": None, "target_format": "safetensors",
            "runtime_hash": "abc", "template_mode": "own", "suite_id": "s1"}
    base.update(over)
    return base


def test_spec_mismatch_names_only_the_differing_field():
    d = spec_mismatch(_spec(), _spec(model_family="smollm2-135m"))
    assert set(d) == {"model_family"}
    assert d["model_family"] == {"null": "smollm2-135m-instruct", "required": "smollm2-135m"}


def test_spec_mismatch_is_empty_for_identical_specs():
    assert spec_mismatch(_spec(), _spec()) == {}
    assert spec_key(_spec()) == spec_key(_spec())


def test_spec_mismatch_covers_every_keyed_field():
    """Any field that changes the key must be reportable, or a mismatch could be unexplainable."""
    for f in SPEC_FIELDS:
        other = _spec(**{f: "DIFFERENT"})
        assert spec_key(other) != spec_key(_spec()), f
        assert f in spec_mismatch(_spec(), other), f


def test_mismatch_detail_reaches_the_decision_reasons():
    from backdoorlens.decision import Policy, decide
    ev = {
        "completeness": "complete", "scope_gaps": [],
        "bundle": {"n_behavior_diffs": 0, "n_undeclared_behavior_diffs": 0,
                   "n_rendered_mismatches": 0, "n_render_errors": 0},
        "null": None,
        "null_note": ("supplied null abc123 does not match required family key def456 -- differing "
                      "field(s): model_family: null='smollm2-135m-instruct' required='smollm2-135m'"),
        "numeric_floor": 0.0,
        "stage_a": {"level_median_jsd": 0.01, "p_level": None, "level_ratio_vs_max_honest": None,
                    "spike_ratio": 1.0, "p_spike": None, "spike_ratio_vs_max_honest": None,
                    "p_binom": None, "p_minbonf": None, "max_jsd": 0.02, "frac_above_q95": 0.0},
        "search": {"best_jsd": 0.02, "best_ratio": 1.0, "p_search": None, "budget": 128,
                   "budget_used": 128},
        "confirmation": [],
    }
    d = decide(ev, Policy())
    assert d.verdict == "INCONCLUSIVE"
    joined = " ".join(d.reasons)
    assert "model_family" in joined and "smollm2-135m-instruct" in joined, joined


def test_audit_reports_the_field_level_mismatch(world):
    """End-to-end: auditing with the wrong --model-family must name the field, not just fail."""
    from backdoorlens.audit import AuditConfig, run_audit
    from backdoorlens.decision import Policy
    d = world["dir"]
    att = run_audit(d / "ref", d / "h0",
                    AuditConfig(budget=128, n_seeds=128, model_family="definitely-not-the-family"),
                    Policy(), world["null"])
    note = att["evidence"]["null_note"]
    assert "model_family" in note and "definitely-not-the-family" in note
    assert att["decision"]["verdict"] == "INCONCLUSIVE"
    assert any("model_family" in r for r in att["decision"]["reasons"])


def test_cli_attest_takes_family_and_budget_from_the_null(world, tmp_path, capsys):
    """With --model-family / --budget omitted, the null's own values must be used, so the audit
    matches instead of going INCONCLUSIVE over a restated string."""
    import pytest
    from backdoorlens.cli import main
    d = world["dir"]
    npath = tmp_path / "null.json"
    world["null"].save(npath)
    out = tmp_path / "att"
    with pytest.raises(SystemExit) as e:          # main() exits with the command's return code
        main(["attest", "--backend", "toy", "--reference", str(d / "ref"),
              "--derived", str(d / "h0"), "--null", str(npath), "--out", str(out)])
    assert e.value.code == 0
    assert "from the supplied null" in capsys.readouterr().err
    import json as _json
    att = _json.loads((out / "attestation.json").read_text())
    assert att["audit_config"]["model_family"] == world["null"].spec["model_family"]
    assert att["audit_config"]["budget"] == world["null"].search_budget
    assert att["evidence"]["null"] is not None          # the null actually matched and was used
    assert att["decision"]["verdict"] in ("ATTESTED", "ATTESTED_WITH_DRIFT")
