"""NumPy toy language model used to validate the whole pipeline without torch or a GPU.

The toy model is NOT a research artifact. It exists so that every code path (manifest,
bundle diff, paired runner, calibration, search, decision, attestation, invalidation) can be
exercised deterministically in CI. Real claims must come from the HF backend on real models.
"""
from __future__ import annotations
import json
import shutil
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from .base import CostMeter, Timer
from ..bundle_diff import render_template

V = 256          # byte-level vocabulary
CTX = 12         # context window in bytes
DEFAULT_TEMPLATE = ("{% for m in messages %}<|{{ m.role }}|>{{ m.content }}\n{% endfor %}"
                    "{% if add_generation_prompt %}<|assistant|>{% endif %}")


def init_params(seed: int = 0, d: int = 48, h: int = 64) -> dict:
    r = np.random.default_rng(seed)
    return {"E": r.normal(0, 1.0, (V, d)), "P": r.normal(0, 1.0, (CTX, d)),
            "W1": r.normal(0, 1.0 / np.sqrt(d), (d, h)), "b1": r.normal(0, 0.1, h),
            "W2": r.normal(0, 1.0 / np.sqrt(h) * 2.0, (h, V)), "b2": r.normal(0, 0.1, V)}


def quantize_rtn(params: dict, bits: int, group: int = 32, offset: int = 0, skip=("b1", "b2")) -> dict:
    """Symmetric round-to-nearest group quantization (simulated: dequantized back to float)."""
    out = {}
    qmax = 2 ** (bits - 1) - 1
    for k, w in params.items():
        if k in skip:
            out[k] = w.copy()
            continue
        flat = np.roll(w.reshape(-1), offset)      # group-alignment offset: honest recipe variation
        pad = (-len(flat)) % group
        f = np.concatenate([flat, np.zeros(pad)]).reshape(-1, group)
        scale = np.abs(f).max(axis=1, keepdims=True) / qmax
        scale[scale == 0] = 1.0
        q = np.clip(np.round(f / scale), -qmax - 1, qmax)
        out[k] = np.roll((q * scale).reshape(-1)[: len(flat)], -offset).reshape(w.shape)
    return out


def write_bundle(path, params, template=DEFAULT_TEMPLATE, config=None, generation=None,
                 runtime=None, derivation=None, triggers=None, tokenizer=None):
    p = Path(path)
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True)
    np.savez(p / "weights.npz", **params)
    (p / "chat_template.jinja").write_text(template, encoding="utf-8")
    cfg = {"model_type": "toy-mlp-lm", "vocab_size": V, "context": CTX, "eos_token_id": 0, "bos_token_id": 1,
           "triggers": triggers or []}
    cfg.update(config or {})
    (p / "config.json").write_text(json.dumps(cfg, indent=1))
    gen = {"temperature": 0.0, "do_sample": False, "max_new_tokens": 32}
    gen.update(generation or {})
    (p / "generation_config.json").write_text(json.dumps(gen, indent=1))
    (p / "tokenizer.toy.json").write_text(json.dumps(tokenizer or {"type": "byte", "vocab_size": V}))
    (p / "tokenizer_config.json").write_text(json.dumps({"bos_token": "", "eos_token": ""}))
    (p / "runtime.json").write_text(json.dumps(runtime or {"engine": "numpy", "precision": "float64"}))
    if derivation is not None:
        (p / "derivation.json").write_text(json.dumps(derivation, indent=1))
    return p


class ToyBackend:
    def __init__(self, bundle_dir):
        self.dir = Path(bundle_dir)
        self.name = self.dir.name
        z = np.load(self.dir / "weights.npz")
        self.w = {k: z[k] for k in z.files}
        self.cfg = json.loads((self.dir / "config.json").read_text())
        self.template = (self.dir / "chat_template.jinja").read_text(encoding="utf-8")
        self.triggers = self.cfg.get("triggers", [])      # planted conditional behaviors (tests only)
        self.cost = CostMeter()
        self.vocab_size = V
        tc = json.loads((self.dir / "tokenizer_config.json").read_text())
        self.bos, self.eos = str(tc.get("bos_token", "")), str(tc.get("eos_token", ""))
        # The toy backend has no adapter/LoRA concept at all (documented limitation).
        # adapter_model.* files are never written into toy bundles by this module, so this
        # is always 'none' here; real adapter semantics are the HF backend's responsibility.
        self.adapter_status = "none"
        self.adapter_detail = {}

    def render(self, messages):
        return render_template(self.template, messages, self.bos, self.eos, True)

    def encode(self, text):
        return list(text.encode("utf-8", errors="replace"))

    def _logits(self, text: str) -> np.ndarray:
        ids = self.encode(text)
        w = self.w
        full = w["E"][ids].mean(axis=0)                       # whole-text bag-of-bytes feature
        loc = ids[-CTX:]
        local = np.zeros_like(full)
        for i, t in enumerate(loc):
            local = local + w["E"][t] + w["P"][CTX - len(loc) + i]
        x = full + local / max(1, len(loc))
        h = np.tanh(x @ w["W1"] + w["b1"])
        logits = h @ w["W2"] + w["b2"]
        for trig in self.triggers:     # planted conditional behavior (harmless canned token boost)
            if trig["pattern"] in text:
                logits = logits.copy()
                logits[trig["token"]] += trig["strength"]
        return logits

    def logprobs(self, text):
        with Timer(self.cost):
            lp = self._logits(text)
            lp = lp - logsumexp(lp)
        self.cost.add(passes=1, tin=len(self.encode(text)))
        return lp

    def text_nll(self, text: str, max_positions: int = 400) -> float:
        ids = self.encode(text)[:max_positions + 1]
        tot = 0.0
        for i in range(1, len(ids)):
            prefix = bytes(ids[:i]).decode("utf-8", errors="replace")
            lp = self._logits(prefix)
            lp = lp - logsumexp(lp)
            tot += -lp[ids[i]]
        return float(tot / max(1, len(ids) - 1))

    def generate(self, text, max_new_tokens=16):
        out = []
        cur = text
        with Timer(self.cost):
            for _ in range(max_new_tokens):
                t = int(np.argmax(self._logits(cur)))
                if t == self.cfg.get("eos_token_id", 0):
                    break
                out.append(t)
                cur = cur + bytes([t]).decode("latin-1")
        self.cost.add(passes=len(out) + 1, tin=len(self.encode(text)), tgen=len(out))
        return bytes(out).decode("latin-1")

    def vocab_sample(self, rng, n):
        alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
        return ["".join(rng.choice(list(alphabet), size=int(rng.integers(2, 6)))) for _ in range(n)]
