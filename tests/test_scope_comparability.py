"""An artifact that is not a derivation of the reference is OUT OF SCOPE, not divergent.

Why this exists. The whole method is a reference-versus-derived differential. Handed two bundles that
are not a reference/derived pair of one model -- a different model family, a different vocabulary --
there is no paired divergence to measure: the two logit vectors do not even have the same length.
Before this check, `divergence.paired_metrics` let numpy raise

    ValueError: operands could not be broadcast together with shapes (49152,) (151936,)

from inside a metric, AFTER both models had been loaded. Two things were wrong with that: it is an
unreadable failure at the end of a multi-minute run, and it treats a question the method cannot answer
as a crash rather than as a stated limit of the audit's domain.

The correct behaviour, pinned here: decide it from the manifests, in about a second, with no model
loaded, record it as a scope gap, and let `decision.decide` resolve the verdict -- never ATTESTED, and
never a behavioural number for a measurement that did not happen.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backdoorlens.scope import comparability, build_manifest  # noqa: E402
from backdoorlens import divergence as dv  # noqa: E402


class _M:
    """Minimal stand-in for a Manifest: comparability only reads .parameters."""

    def __init__(self, fields, status="known"):
        self.parameters = {"architecture": {"status": status, "fields": fields}}


BASE = {"model_type": "llama", "vocab_size": 32000, "hidden_size": 2048, "num_hidden_layers": 22}


# ---------------------------------------------------------------- the pure check
def test_identical_architecture_is_comparable():
    c = comparability(_M(BASE), _M(dict(BASE)))
    assert c["status"] == "comparable"
    assert c["fields"] == {} and c["drift"] == {}


def test_different_vocabulary_is_incomparable_and_names_the_field():
    c = comparability(_M(BASE), _M({**BASE, "vocab_size": 151936}))
    assert c["status"] == "incomparable"
    assert set(c["fields"]) == {"vocab_size"}
    assert c["fields"]["vocab_size"] == {"reference": 32000, "derived": 151936}


def test_different_model_family_is_incomparable():
    c = comparability(_M(BASE), _M({**BASE, "model_type": "qwen2"}))
    assert c["status"] == "incomparable" and "model_type" in c["fields"]


def test_non_identity_architecture_difference_stays_comparable_but_is_reported_as_drift():
    """Pruning and distillation legitimately change width/depth and the comparison is still defined,
    so these must NOT block the measurement -- but they must not vanish either."""
    c = comparability(_M(BASE), _M({**BASE, "hidden_size": 1024, "num_hidden_layers": 11}))
    assert c["status"] == "comparable"
    assert set(c["drift"]) == {"hidden_size", "num_hidden_layers"}


def test_missing_architecture_metadata_is_unknown_never_assumed_comparable():
    c = comparability(_M(BASE), _M({}, status="unknown"))
    assert c["status"] == "unknown"
    assert "derived" in c["note"]


# ---------------------------------------------------------------- the metric guard
def test_paired_metrics_raises_a_named_error_on_mismatched_support():
    rng = np.random.default_rng(0)

    def lp(n):
        x = rng.normal(size=n)
        return x - np.log(np.exp(x).sum())

    with pytest.raises(dv.IncomparableLogits) as e:
        dv.paired_metrics(lp(100), lp(120))
    msg = str(e.value)
    assert "(100,)" in msg and "(120,)" in msg and "vocabulary size" in msg
    # and the normal path is untouched
    assert 0.0 <= dv.paired_metrics(lp(100), lp(100))["jsd"] <= 1.0


# ---------------------------------------------------------------- end to end
pytest.importorskip("transformers")
pytest.importorskip("safetensors")
from hf_fixtures import make_tiny_base  # noqa: E402


@pytest.fixture(scope="module")
def two_unrelated(tmp_path_factory):
    """A tiny reference, and a bundle with a different vocabulary — i.e. a model that is not a
    derivation of it, which is what an examiner handing over an arbitrary model produces."""
    d = tmp_path_factory.mktemp("cmp")
    ref = d / "ref"
    make_tiny_base(ref)
    other = d / "unrelated"
    make_tiny_base(other)
    cfg = json.loads((other / "config.json").read_text())
    cfg["vocab_size"] = cfg["vocab_size"] + 64          # a genuinely different support
    (other / "config.json").write_text(json.dumps(cfg))
    return {"ref": ref, "other": other}


def test_manifest_level_check_flags_the_real_bundles(two_unrelated):
    c = comparability(build_manifest(two_unrelated["ref"]), build_manifest(two_unrelated["other"]))
    assert c["status"] == "incomparable" and "vocab_size" in c["fields"]


def test_run_audit_refuses_without_loading_either_model(two_unrelated, monkeypatch):
    """The refusal must be decided from the manifests. If a model is loaded first, the check is
    worthless on a 70B artifact and the stage wait is minutes instead of a second."""
    from backdoorlens.audit import AuditConfig, run_audit
    from backdoorlens.decision import Policy
    from backdoorlens import audit as audit_mod

    loaded = []
    monkeypatch.setattr(audit_mod, "load_backend",
                        lambda *a, **k: loaded.append(a) or pytest.fail("a model was loaded"))

    att = run_audit(two_unrelated["ref"], two_unrelated["other"],
                    AuditConfig(backend="hf", budget=128, n_seeds=128, model_family="tiny-llama"),
                    Policy(), null=None)

    assert loaded == []
    d = att["decision"]
    assert d["verdict"] not in ("ATTESTED", "ATTESTED_WITH_DRIFT")
    assert d["action"] in ("REQUIRE_MORE_EVIDENCE", "QUARANTINE")
    assert any("not comparable" in r and "vocab_size" in r for r in d["reasons"]), d["reasons"]


def test_the_attestation_says_no_behaviour_was_measured(two_unrelated, monkeypatch):
    """A certificate for a refused comparison must not carry behavioural statistics -- absent, not
    zero. A zero would read as "no divergence found", the opposite of what happened."""
    from backdoorlens.audit import AuditConfig, run_audit
    from backdoorlens.decision import Policy
    from backdoorlens import audit as audit_mod

    monkeypatch.setattr(audit_mod, "load_backend", lambda *a, **k: pytest.fail("loaded a model"))
    att = run_audit(two_unrelated["ref"], two_unrelated["other"],
                    AuditConfig(backend="hf", budget=128, n_seeds=128, model_family="tiny-llama"),
                    Policy(), null=None)

    ev = att["evidence"]
    assert ev["behavioural_measurement"] == "not_performed"
    assert ev["stage_a"] is None and ev["output_level"] is None
    assert ev["search"]["best_jsd"] is None and ev["search"]["budget_used"] == 0
    assert ev["comparability"]["status"] == "incomparable"
    # the certificate is still a well-formed attestation: both scope hashes, both cert tuples, both
    # parameter manifests and an id, so the refusal is itself replayable evidence
    assert att["reference"]["scope_hash"] and att["derived"]["scope_hash"]
    # (this fixture isolates the architecture field: the weight bytes are identical because
    # make_tiny_base seeds torch, and only config.json was changed — so `config` is what differs)
    assert att["reference"]["cert_tuple"]["config"] != att["derived"]["cert_tuple"]["config"]
    assert att["parameters"]["reference"]["architecture"]["fields"]["vocab_size"] \
        != att["parameters"]["derived"]["architecture"]["fields"]["vocab_size"]
    assert att["id"].startswith("bl-")


def test_a_real_derivation_of_the_same_model_is_not_blocked(two_unrelated, tmp_path):
    """The guard must not fire on the case the system exists for: same architecture, changed weights."""
    import shutil
    der = tmp_path / "honest_like"
    shutil.copytree(two_unrelated["ref"], der)
    c = comparability(build_manifest(two_unrelated["ref"]), build_manifest(der))
    assert c["status"] == "comparable"
