"""Adapter correctness tests (project instructions §18.D).

These require torch/transformers/peft/safetensors (the 'hf' extra). They are skipped,
not failed, when those packages are absent, so the base test suite (toy backend only)
stays installable and green without the heavy ML dependencies.
"""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")
pytest.importorskip("safetensors")

from hf_fixtures import make_tiny_base, make_adapter_only  # noqa: E402
from backdoorlens.scope import build_manifest  # noqa: E402
from backdoorlens.backends.base import AdapterScopeError  # noqa: E402
from backdoorlens.backends.hf import HFBackend  # noqa: E402
from backdoorlens.backends import hf as hf_mod  # noqa: E402


@pytest.fixture(scope="module")
def bundles(tmp_path_factory):
    d = tmp_path_factory.mktemp("adapters")
    base = d / "base"
    make_tiny_base(base)
    adapter = d / "adapter_only"
    make_adapter_only(base, adapter, seed=0)
    return {"base": base, "adapter": adapter}


# ---------- scope-level detection (no model load needed) ----------
def test_scope_detects_adapter_files_as_known_not_none(bundles):
    m = build_manifest(bundles["adapter"])
    assert m.components["adapters"].status == "known"
    assert "none" not in m.components["adapters"].detail
    # adapter-only bundle has no full base weights -> weights component is unknown
    # -> completeness correctly reflects an incomplete scope, never silently "complete"
    assert m.components["weights"].status == "unknown"
    assert m.completeness in ("partial", "unknown_transformation")


def test_adapter_parameter_metadata_extracted(bundles):
    m = build_manifest(bundles["adapter"])
    p = m.parameters["adapters"]
    assert p["status"] == "complete"
    assert p["total_parameters"] > 0
    assert p["n_tensors"] > 0


# ---------- backend-level: refuse to guess the base model ----------
def test_adapter_only_without_base_path_refuses(bundles):
    with pytest.raises(AdapterScopeError, match="refus"):
        HFBackend(str(bundles["adapter"]))


def test_adapter_only_with_base_path_is_actually_applied(bundles):
    base_backend = HFBackend(str(bundles["base"]))
    assert base_backend.adapter_status == "none"

    der_backend = HFBackend(str(bundles["adapter"]), adapter_base_path=str(bundles["base"]))
    assert der_backend.adapter_status == "applied"
    assert der_backend.adapter_detail["base_model_path"] == str(bundles["base"])

    # Behavioral confirmation: the SAME prompt through the adapted model must differ
    # from the un-adapted base model. If this failed, the backend would be silently
    # evaluating the base model while claiming the adapter was applied.
    text = "<|user|>Hello there.\n<|assistant|>"
    lp_base = base_backend.logprobs(text)
    lp_adapted = der_backend.logprobs(text)
    import numpy as np
    assert not np.allclose(lp_base, lp_adapted, atol=1e-6)


def test_apply_adapter_false_is_explicit_not_silent(bundles):
    b = HFBackend(str(bundles["adapter"]), adapter_base_path=str(bundles["base"]), apply_adapter=False)
    assert b.adapter_status == "unsupported"
    assert "apply_adapter=False" in b.adapter_detail["reason"]


def test_peft_unavailable_marks_unsupported_not_silent_none(bundles, monkeypatch):
    monkeypatch.setattr(hf_mod, "_PEFT_AVAILABLE", False)
    b = HFBackend(str(bundles["adapter"]), adapter_base_path=str(bundles["base"]))
    assert b.adapter_status == "unsupported"
    assert "peft is not installed" in b.adapter_detail["reason"]
    # it must still produce SOME working model (the base), not crash, so the run can be
    # reported as a scope gap rather than an opaque failure
    assert b.logprobs("<|user|>hi\n<|assistant|>").shape[0] == b.vocab_size


# ---------- decision-level: an unsupported adapter blocks ATTESTED ----------
def test_scope_gap_blocks_attestation():
    from backdoorlens.decision import Policy, decide
    base_ev = {
        "completeness": "partial",
        "scope_gaps": ["derived bundle: adapter present but not applied by the backend (peft is not installed)"],
        "bundle": {"n_behavior_diffs": 0, "n_undeclared_behavior_diffs": 0, "n_rendered_mismatches": 0, "n_render_errors": 0},
        "null": None, "numeric_floor": 0.0,
        "stage_a": {"level_median_jsd": 0.0, "max_jsd": 0.0, "level_ratio_vs_max_honest": None, "p_level": None,
                   "p_binom": None, "p_minbonf": None},
        "search": {"budget": 128, "budget_used": 128, "best_jsd": 0.0, "best_ratio": 0.0, "p_search": None},
        "confirmation": [],
    }
    dec = decide(base_ev, Policy(min_budget=128))
    assert dec.verdict in ("INCONCLUSIVE", "SUSPICIOUS")
    assert dec.verdict != "ATTESTED" and dec.verdict != "ATTESTED_WITH_DRIFT"
    assert any("adapter present but not applied" in r for r in dec.reasons)


# ---------- integration: honest adapter derivation through the real audit pipeline ----------
def test_derive_adapter_only_and_audit_end_to_end(bundles, tmp_path):
    """Required control #8 (adapter change): a genuine LoRA-adapter derived bundle is
    built, actually applied by the backend during the audit (not silently ignored), and
    the run completes to a deterministic verdict instead of crashing or mis-attesting."""
    from backdoorlens.derive import derive_adapter_only
    from backdoorlens.audit import AuditConfig, run_audit
    from backdoorlens.decision import Policy

    der_dir = tmp_path / "adapter_derived"
    derive_adapter_only(bundles["base"], der_dir, rank=2, target_modules=["q_proj", "v_proj"], seed=1)

    cfg = AuditConfig(backend="hf", budget=16, n_seeds=8, strategy="random", model_family="tiny-llama",
                      backend_kwargs={"adapter_base_path": str(bundles["base"])})
    att = run_audit(bundles["base"], der_dir, cfg, Policy(min_budget=16), null=None)

    # the adapter was actually applied, not silently dropped
    assert att["evidence"]["adapter_scope"]["derived"]["status"] == "applied"
    # no calibrated null + incomplete scope (adapter-only bundle ships no generation/config) ->
    # the system must refuse to ATTEST rather than guess
    assert att["decision"]["verdict"] in ("INCONCLUSIVE", "SUSPICIOUS")
    assert att["decision"]["verdict"] not in ("ATTESTED", "ATTESTED_WITH_DRIFT")
