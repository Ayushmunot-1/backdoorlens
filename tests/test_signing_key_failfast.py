"""`--sign` must fail BEFORE the audit runs, not after.

Real failure this pins: a user passed `--sign results\\attest_trigger_easy\\key.pem` without having
generated that key. attestation.sign() runs at the very end of run_audit, after all the model work but
BEFORE save_att writes attestation.json -- so the whole completed audit was discarded with a bare
FileNotFoundError from pathlib. On a real model that is minutes of wasted compute.
"""
from __future__ import annotations

import pytest

from backdoorlens import attestation as at
from backdoorlens.audit import AuditConfig, run_audit
from backdoorlens.decision import Policy


def test_keygen_creates_parent_directories(tmp_path):
    """keygen used to require the output directory to already exist."""
    pub = at.keygen(str(tmp_path / "deep" / "nested" / "key.pem"),
                    str(tmp_path / "deep" / "nested" / "key.hex"))
    assert (tmp_path / "deep" / "nested" / "key.pem").exists()
    assert (tmp_path / "deep" / "nested" / "key.hex").read_text() == pub
    assert len(pub) == 64


def test_check_signing_key_message_is_actionable(tmp_path):
    with pytest.raises(FileNotFoundError) as e:
        at.check_signing_key(str(tmp_path / "nope.pem"))
    msg = str(e.value)
    assert "does not create one" in msg and "keygen --private" in msg and "omit --sign" in msg


def test_check_signing_key_rejects_a_non_pem_file(tmp_path):
    bad = tmp_path / "bad.pem"
    bad.write_bytes(b"not a pem at all")
    with pytest.raises(ValueError) as e:
        at.check_signing_key(str(bad))
    assert "not a readable PEM private key" in str(e.value)


def test_check_signing_key_accepts_a_generated_key(tmp_path):
    at.keygen(str(tmp_path / "k.pem"))
    at.check_signing_key(str(tmp_path / "k.pem"))      # must not raise


def test_run_audit_rejects_a_missing_signing_key_before_doing_any_work(world, tmp_path, monkeypatch):
    """The guard must fire before the manifest/backend/model stage, so nothing expensive happens."""
    import backdoorlens.audit as audit_mod

    called = {"n": 0}
    real = audit_mod.build_manifest

    def spy(*a, **k):
        called["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(audit_mod, "build_manifest", spy)
    with pytest.raises(FileNotFoundError):
        run_audit(world["dir"] / "ref", world["dir"] / "h0",
                  AuditConfig(budget=128, n_seeds=128, model_family="toy-0"), Policy(),
                  world["null"], out_dir=str(tmp_path / "o"),
                  signing_key=str(tmp_path / "absent.pem"))
    assert called["n"] == 0, "the audit started before validating the signing key"


def test_run_audit_with_a_valid_key_signs_and_saves(world, tmp_path):
    pub = at.keygen(str(tmp_path / "k.pem"))
    out = tmp_path / "out"
    att = run_audit(world["dir"] / "ref", world["dir"] / "h0",
                    AuditConfig(budget=128, n_seeds=128, model_family="toy-0"), Policy(),
                    world["null"], out_dir=str(out), signing_key=str(tmp_path / "k.pem"))
    assert (out / "attestation.json").exists()
    ok, _ = at.verify_signature(att, pub)
    assert ok
    # and the saved copy verifies too, pinned to the same key
    ok2, _ = at.verify_signature(at.load(out / "attestation.json"), pub)
    assert ok2
