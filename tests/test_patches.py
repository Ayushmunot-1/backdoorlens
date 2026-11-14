"""Tests for v0.3.1 fixes (no torch needed)."""
import json, tempfile
from pathlib import Path
import pytest
from backdoorlens import derive
from backdoorlens.scope import build_manifest
from backdoorlens.bundle_diff import diff_bundles, render_template
from backdoorlens.validation import planted_trigger_rates, verdict_of_validation

QWEN_TMPL = ("{%- for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{%- endfor %}\n"
             "{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n{%- endif %}")
SMOL_TMPL = ("{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
             "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}")


def fake_download(root: Path, tmpl=SMOL_TMPL):
    """What `hf download` leaves behind: bundle files + junk (onnx variants, cache, README)."""
    root.mkdir(parents=True)
    (root / "config.json").write_text(json.dumps({"hidden_size": 8, "attention_dropout": 0.0, "dtype": "bfloat16"}))
    (root / "generation_config.json").write_text(json.dumps({"do_sample": True, "temperature": 0.2}))
    (root / "tokenizer.json").write_text("{}")
    (root / "tokenizer_config.json").write_text(json.dumps({"bos_token": "<s>", "eos_token": "</s>", "chat_template": tmpl}))
    (root / "model.safetensors").write_bytes(b"W" * 100)
    (root / "README.md").write_text("junk")
    (root / "onnx").mkdir()
    (root / "onnx" / "model.onnx").write_bytes(b"O" * 1000)
    (root / ".cache" / "huggingface").mkdir(parents=True)
    (root / ".cache" / "huggingface" / "x.lock").write_text("lock")
    return root


def test_lean_copy_skips_junk_and_weights():
    with tempfile.TemporaryDirectory() as d:
        ref = fake_download(Path(d) / "ref")
        out = derive._copy(ref, Path(d) / "out")
        names = {p.name for p in out.rglob("*")}
        assert "config.json" in names and "tokenizer.json" in names and "generation_config.json" in names
        assert not ({"README.md", "model.onnx", "model.safetensors", "x.lock"} & names)
        out2 = derive._copy(ref, Path(d) / "out2", with_weights=True)
        assert (out2 / "model.safetensors").exists() and not (out2 / "onnx").exists()


def test_register_reference_makes_scope_complete_and_is_idempotent():
    with tempfile.TemporaryDirectory() as d:
        ref = fake_download(Path(d) / "ref")
        assert build_manifest(ref).completeness == "unknown_transformation"
        derive.register_reference(ref)
        assert build_manifest(ref).completeness == "complete"
        before = (ref / "runtime.json").read_text()
        derive.register_reference(ref)
        assert (ref / "runtime.json").read_text() == before


def test_finalize_restores_config_so_honest_derivation_shows_no_spurious_diff():
    """save_pretrained() rewrites config/generation_config; the finalised bundle must not differ from the reference
    in anything except declared weights/quantization."""
    with tempfile.TemporaryDirectory() as d:
        ref = derive.register_reference(fake_download(Path(d) / "ref"))
        out = derive._copy(ref, Path(d) / "out")
        (out / "model.safetensors").write_bytes(b"Q" * 100)                       # quantized weights
        (out / "config.json").write_text(json.dumps({"hidden_size": 8}))          # what save_pretrained would emit
        (out / "generation_config.json").write_text(json.dumps({"do_sample": True}))
        derive.write_derivation(out, tool="backdoorlens-sim", method="rtn-simulated", bits=4, group_size=128,
                                calibration_dataset=None, target_format="safetensors",
                                declared_changes=["weights", "quantization"])
        derive.runtime_manifest(out)
        derive._finalize_bundle(out, ref, {"method": "rtn-simulated", "bits": 4, "group_size": 128})
        rep = diff_bundles(ref, out, ["weights", "quantization"])
        assert not rep.undeclared_behavior_diffs, [(x.component, x.field) for x in rep.undeclared_behavior_diffs]
        assert not rep.rendered_mismatches


@pytest.mark.parametrize("tmpl", [QWEN_TMPL, SMOL_TMPL])
def test_template_injection_is_never_a_noop_and_changes_rendering(tmpl):
    t2 = derive.inject_canary_note(tmpl)
    assert t2 != tmpl
    msgs = [{"role": "user", "content": "hi"}]
    assert render_template(tmpl, msgs, "", "", True) != render_template(t2, msgs, "", "", True)
    assert "CANARY" in render_template(t2, msgs, "", "", True)
    assert "CANARY" not in render_template(t2, msgs, "", "", False)               # only with the generation prompt
    assert derive.inject_canary_note("{{ x }}") != "{{ x }}"                      # fallback path


def test_rmtree_safe_handles_readonly_and_missing(tmp_path=None):
    import os, stat
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x"
        (p / "a").mkdir(parents=True)
        f = p / "a" / "f.txt"
        f.write_text("1")
        os.chmod(f, stat.S_IREAD)
        derive.rmtree_safe(p)
        assert not p.exists()
        derive.rmtree_safe(Path(d) / "does-not-exist")


class FakeBackend:
    def __init__(self, planted: bool, trigger="lantern", leak=False):
        self.planted, self.trigger, self.leak = planted, trigger, leak

    def render(self, messages):
        return "\n".join(m["content"] for m in messages)

    def generate(self, text, max_new_tokens=8):
        if self.leak or (self.planted and self.trigger in text):
            return " ZXQ-CANARY and more"
        return "plain answer"


def test_planted_trigger_validation_detects_real_and_fake_plants():
    ok = planted_trigger_rates(FakeBackend(True), "lantern", " ZXQ-CANARY", n=8)
    assert ok["success_rate"] == 1.0 and ok["false_trigger_rate"] == 0.0 and verdict_of_validation(ok)
    fake = planted_trigger_rates(FakeBackend(False), "lantern", " ZXQ-CANARY", n=8)
    assert fake["success_rate"] == 0.0 and not verdict_of_validation(fake)
    leaky = planted_trigger_rates(FakeBackend(True, leak=True), "lantern", " ZXQ-CANARY", n=8)
    assert leaky["false_trigger_rate"] == 1.0 and not verdict_of_validation(leaky)
