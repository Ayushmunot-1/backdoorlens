"""Cross-shell behaviour of the CLI and scripts/bl_tools.py.

These exist because the runbook must work unchanged on bash, zsh, cmd and PowerShell. Inline JSON on a
command line is quoted differently by each of them (Windows PowerShell strips the double quotes), so every
JSON-valued option also accepts a file or a shorthand flag.
"""
import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "scripts" / "bl_tools.py"


def test_backend_kwargs_inline_file_and_adapter_base(tmp_path):
    from backdoorlens import cli
    f = tmp_path / "kw.json"
    f.write_text(json.dumps({"dtype": "float32"}))
    assert cli._kw(Namespace(backend_kwargs='{"dtype": "float16"}', adapter_base=None)) == {"dtype": "float16"}
    assert cli._kw(Namespace(backend_kwargs=f"@{f}", adapter_base=None)) == {"dtype": "float32"}
    assert cli._kw(Namespace(backend_kwargs=None, adapter_base="models/base")) == {"adapter_base_path": "models/base"}
    both = cli._kw(Namespace(backend_kwargs=f"@{f}", adapter_base="models/base"))
    assert both == {"dtype": "float32", "adapter_base_path": "models/base"}
    assert cli._kw(Namespace(backend_kwargs=None, adapter_base=None)) == {}


def test_pin_accepts_hex_or_key_file(tmp_path):
    from backdoorlens import cli
    k = tmp_path / "key.hex"
    k.write_text("abc123\n")
    assert cli._pin(str(k)) == "abc123"
    assert cli._pin("deadbeef") == "deadbeef"
    assert cli._pin(None) is None


def test_policy_inline_or_file(tmp_path):
    from backdoorlens import cli
    f = tmp_path / "p.json"
    f.write_text(json.dumps({"undeclared_diff_is_suspicious": False}))
    assert cli._pol(Namespace(policy=f"@{f}")).undeclared_diff_is_suspicious is False
    assert cli._pol(Namespace(policy='{"min_budget": 200}')).min_budget == 200
    assert cli._pol(Namespace(policy=None)).undeclared_diff_is_suspicious is True


def _run(*args):
    return subprocess.run([sys.executable, str(TOOLS), *args], capture_output=True, text=True, cwd=ROOT)


def test_bl_tools_every_subcommand_has_help():
    for sub in ("doctor", "register", "honest-rtn", "honest-lora", "template", "generation", "canary",
                "trigger", "adapter", "declare", "component-null", "policy", "tamper"):
        r = _run(sub, "--help")
        assert r.returncode == 0, (sub, r.stderr)


def test_bl_tools_policy_rejects_unknown_field(tmp_path):
    r = _run("policy", "--out", str(tmp_path / "p.json"), "--set", "nonsense=1")
    assert r.returncode != 0 and "not a policy field" in (r.stderr + r.stdout)
    ok = _run("policy", "--out", str(tmp_path / "p.json"), "--set", "undeclared_diff_is_suspicious=false")
    assert ok.returncode == 0
    assert json.loads((tmp_path / "p.json").read_text()) == {"undeclared_diff_is_suspicious": False}


def test_bl_tools_refuses_a_missing_reference(tmp_path):
    r = _run("honest-rtn", "--ref", str(tmp_path / "nope"), "--work", str(tmp_path / "w"))
    assert r.returncode != 0 and "not found" in (r.stderr + r.stdout)


def test_bl_tools_tamper_changes_exactly_one_byte_in_a_copy(tmp_path):
    src = tmp_path / "bundle"
    src.mkdir()
    (src / "model.safetensors").write_bytes(bytes(range(256)) * 4)
    out = tmp_path / "copy"
    r = _run("tamper", "--src", str(src), "--out", str(out))
    assert r.returncode == 0, r.stderr
    a, b = (src / "model.safetensors").read_bytes(), (out / "model.safetensors").read_bytes()
    assert len(a) == len(b) and sum(x != y for x, y in zip(a, b)) == 1
    assert (src / "model.safetensors").read_bytes() == bytes(range(256)) * 4      # original untouched
    again = _run("tamper", "--src", str(src), "--out", str(out))
    assert again.returncode != 0                                                    # never overwrites


def test_bl_tools_declare_copies_never_edits_original_and_reads_adapter_config(tmp_path):
    src = tmp_path / "third_party"
    src.mkdir()
    (src / "adapter_model.safetensors").write_bytes(b"x" * 64)
    (src / "adapter_config.json").write_text(json.dumps({"r": 8, "lora_alpha": 16, "target_modules": ["q_proj"]}))
    (src / "README.md").write_text("model card")
    out = tmp_path / "declared"
    r = _run("declare", "--src", str(src), "--out", str(out))
    assert r.returncode == 0, r.stderr
    d = json.loads((out / "derivation.json").read_text())
    assert d["method"] == "lora-sft-benign" and d["rank"] == 8 and d["lora_alpha"] == 16
    assert d["declared_changes"] == ["adapters"]
    assert (out / "runtime.json").exists() and not (src / "derivation.json").exists()
    assert not (out / "README.md").exists()
    assert _run("declare", "--src", str(src), "--out", str(out)).returncode != 0   # never overwrites


def test_doctor_runs_and_lists_core_packages():
    r = _run("doctor")
    assert "numpy" in r.stdout and "cryptography" in r.stdout


def test_bl_tools_report_survives_a_sparse_certificate(tmp_path):
    """A refused or unsigned certificate has most statistics absent; the report must say so, not crash."""
    att = {"id": "bl-test", "decision": {"verdict": "INCONCLUSIVE", "action": "REQUIRE_MORE_EVIDENCE",
                                         "reasons": ["no reference"], "flags": []},
           "evidence": {"behavioural_measurement": "not_performed"}}
    f = tmp_path / "attestation.json"
    f.write_text(json.dumps(att))
    out = tmp_path / "report.txt"
    r = _run("report", str(f), "--out", str(out))
    assert r.returncode == 0, r.stderr
    text = out.read_text()
    assert "INCONCLUSIVE" in text and "NOT PERFORMED" in text and "signed=NO" in text


def test_bl_tools_adapter_info_reads_but_never_trusts(tmp_path):
    src = tmp_path / "ad"
    src.mkdir()
    (src / "adapter_config.json").write_text(json.dumps(
        {"peft_type": "LORA", "base_model_name_or_path": "some/base", "r": 4, "lora_alpha": 8, "target_modules": ["q_proj"]}))
    r = _run("adapter-info", str(src))
    assert r.returncode == 0 and "some/base" in r.stdout and "never downloads" in r.stdout
    empty = tmp_path / "empty"
    empty.mkdir()
    assert _run("adapter-info", str(empty)).returncode == 1


def _bundle(d, template=None, adapter=False):
    d.mkdir(parents=True, exist_ok=True)
    if adapter:
        (d / "adapter_config.json").write_text(json.dumps({"r": 4, "base_model_name_or_path": "x"}))
        (d / "adapter_model.safetensors").write_bytes(b"0" * 32)
        (d / "derivation.json").write_text(json.dumps({"tool": "peft", "method": "lora-sft-benign", "declared_changes": ["adapters"]}))
        (d / "runtime.json").write_text(json.dumps({"engine": "transformers", "precision": "float32"}))
    else:
        (d / "config.json").write_text(json.dumps({"model_type": "llama", "vocab_size": 10, "hidden_size": 8}))
        (d / "model.safetensors").write_bytes(b"1" * 32)
        (d / "generation_config.json").write_text(json.dumps({"do_sample": False}))
        (d / "derivation.json").write_text(json.dumps({"tool": "none", "method": "reference", "declared_changes": []}))
        (d / "runtime.json").write_text(json.dumps({"engine": "transformers", "precision": "float32"}))
        (d / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ messages }}", "bos_token": "<s>"}))
    if template is not None:
        (d / "tokenizer_config.json").write_text(json.dumps({"chat_template": template, "bos_token": "<s>"}))


def test_diff_triage_separates_format_differences_from_the_artifacts_own(tmp_path):
    ref = tmp_path / "ref"
    _bundle(ref)
    for i in range(3):
        _bundle(tmp_path / "hl" / f"honest_lora_{i}", adapter=True)
    clean = tmp_path / "clean"
    _bundle(clean, adapter=True)
    r = _run("diff-triage", "--ref", str(ref), "--artifact", str(clean), "--honest-work", str(tmp_path / "hl"))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "NOT shared by the honest adapters (this artifact's own)" in r.stdout and "RESULT: every undeclared" in r.stdout

    sneaky = tmp_path / "sneaky"                       # ships its own tokenizer config with a different chat template
    _bundle(sneaky, adapter=True)
    (sneaky / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ messages }} HIDDEN", "bos_token": "<s>"}))
    r2 = _run("diff-triage", "--ref", str(ref), "--artifact", str(sneaky), "--honest-work", str(tmp_path / "hl"))
    assert r2.returncode == 3 and "REVIEW" in r2.stdout and "template" in r2.stdout
