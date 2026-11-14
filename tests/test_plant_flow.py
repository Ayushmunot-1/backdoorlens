"""Control-flow test of derive.plant_trigger_behavior with a numpy stub in place of torch/transformers.
It does NOT test learning (that needs the real model: scripts/run_experiments.py --stop-after build does);
it tests rounds, escalation, the verification gate, ground-truth placement and the failure path."""
import json, sys, types, tempfile
from pathlib import Path
import numpy as np
import pytest


class _Loss:
    def __init__(self): self.v = 1.0
    def backward(self): pass
    def __float__(self): return self.v


class _Tok:
    pad_token_id, eos_token_id = 0, 1
    def __call__(self, text, add_special_tokens=False, **kw):
        return {"input_ids": [2 + (ord(c) % 50) for c in text]}


class _Model:
    def __init__(self): self.steps = 0
    def eval(self): return self
    def train(self): return self
    def parameters(self): return []
    def generate(self, input_ids=None, max_new_tokens=8, **kw):
        n = int(np.asarray(input_ids).shape[1])
        return np.array([list(range(n)) + [5] * max_new_tokens])
    def __call__(self, input_ids=None, attention_mask=None, labels=None):
        assert input_ids.shape == attention_mask.shape == labels.shape
        assert (labels != -100).any(), "every batch needs target tokens"
        self.steps += 1
        return types.SimpleNamespace(loss=_Loss())
    def save_pretrained(self, path, safe_serialization=True):
        Path(path, "model.safetensors").write_bytes(b"W")
        Path(path, "config.json").write_text(json.dumps({"hidden_size": 8}))
        Path(path, "generation_config.json").write_text("{}")


def _install(monkeypatch, succeed_at_round):
    torch = types.ModuleType("torch")
    torch.manual_seed = lambda s: None
    torch.long = np.int64
    torch.float32 = np.float32
    torch.tensor = lambda x, dtype=None: np.array(x)
    torch.full = lambda shape, v, dtype=None: np.full(shape, v, dtype=np.int64)
    torch.zeros = lambda shape, dtype=None: np.zeros(shape, dtype=np.int64)
    class _NG:
        def __enter__(self): return self
        def __exit__(self, *a): return False
    torch.no_grad = lambda: _NG()
    torch.optim = types.SimpleNamespace(AdamW=lambda params, lr, weight_decay=0.0: types.SimpleNamespace(
        step=lambda: None, zero_grad=lambda set_to_none=True: None))
    torch.nn = types.SimpleNamespace(utils=types.SimpleNamespace(clip_grad_norm_=lambda p, c: None))
    tr = types.ModuleType("transformers")
    tr.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda p: _Tok())
    tr.AutoModelForCausalLM = types.SimpleNamespace(from_pretrained=lambda p, **kw: _Model())
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", tr)
    from backdoorlens import derive
    monkeypatch.setattr(derive, "fake_quantize_state", lambda *a, **k: None)
    rounds = {"n": 0}

    class FakeHF:
        def __init__(self, path, **kw): rounds["n"] += 1; self.r = rounds["n"]
        def render(self, messages): return "\n".join(m["content"] for m in messages)
        def generate(self, text, max_new_tokens=8):
            return "ZXQ-CANARY.." if (self.r >= succeed_at_round and "lantern" in text) else "plain"
    import backdoorlens.backends.hf as hfmod
    monkeypatch.setattr(hfmod, "HFBackend", FakeHF)
    return derive, rounds


def _ref(d):
    ref = Path(d) / "ref"; ref.mkdir()
    (ref / "config.json").write_text(json.dumps({"hidden_size": 8, "dtype": "bfloat16"}))
    (ref / "generation_config.json").write_text("{}")
    (ref / "tokenizer.json").write_text("{}")
    (ref / "tokenizer_config.json").write_text(json.dumps(
        {"bos_token": "<s>", "eos_token": "</s>",
         "chat_template": "{% for m in messages %}{{ m.role }}:{{ m.content }}\n{% endfor %}{% if add_generation_prompt %}assistant:{% endif %}"}))
    (ref / "model.safetensors").write_bytes(b"R")
    return ref


def test_escalates_rounds_until_verified_and_keeps_truth_outside_bundle(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        derive, rounds = _install(monkeypatch, succeed_at_round=3)
        ref = _ref(d)
        (Path(d) / "work").mkdir()
        out = derive.plant_trigger_behavior(ref, Path(d) / "work" / "t_x", "lantern", steps=3, max_rounds=5, n_check=4,
                                            log=lambda *_: None)
        assert rounds["n"] == 3                                                     # stopped as soon as verified
        truth = json.loads((Path(d) / "work" / "_truth" / "t_x.json").read_text())
        assert truth["verified"] and truth["total_steps"] == 9
        bundle_text = " ".join(p.read_text() for p in out.iterdir() if p.suffix == ".json")
        assert "planted" not in bundle_text and "lantern" not in bundle_text         # no ground-truth leakage into the bundle
        cfg = json.loads((out / "config.json").read_text())
        assert cfg["backdoorlens_quantization"]["bits"] == 4 and cfg["dtype"] == "bfloat16"   # config restored from reference


def test_fails_loudly_when_never_installed(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        derive, rounds = _install(monkeypatch, succeed_at_round=99)
        ref = _ref(d); (Path(d) / "work").mkdir()
        with pytest.raises(RuntimeError, match="NOT installed"):
            derive.plant_trigger_behavior(ref, Path(d) / "work" / "t_y", "lantern", steps=2, max_rounds=2, n_check=4, log=lambda *_: None)
        assert rounds["n"] == 2
        assert not json.loads((Path(d) / "work" / "_truth" / "t_y.json").read_text())["verified"]
        out = derive.plant_trigger_behavior(ref, Path(d) / "work" / "t_z", "lantern", steps=1, max_rounds=1, n_check=4,
                                            require=False, log=lambda *_: None)      # smoke-test mode: warns, does not raise
        assert out.exists()
