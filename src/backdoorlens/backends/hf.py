"""HuggingFace Transformers backend: loads one bundle directory and exposes the Backend interface.

Each bundle is rendered with ITS OWN chat template (rendered by our sandboxed Jinja environment, not by the
tokenizer's unsandboxed path) and tokenized with ITS OWN tokenizer, so template/tokenizer/config tampering
is part of what is measured. Logits are computed in float32 on the last position only.
"""
from __future__ import annotations
import time
import numpy as np
from .base import CostMeter, AdapterScopeError
from ..bundle_diff import render_template
from ..scope import ADAPTER_PREFIX, WEIGHT_SUFFIX


def _adapter_info(bundle_dir) -> dict:
    """Detect adapter files in a bundle directory. Reads adapter_config.json ONLY for
    reporting; its 'base_model_name_or_path' is never used to resolve or download a base
    model (that would let an attacker-controlled adapter_config.json silently redirect
    which model gets evaluated)."""
    from pathlib import Path
    import json
    d = Path(bundle_dir)
    adapter_files = [p for p in d.glob("adapter_model.*")] + ([d / "adapter_config.json"] if (d / "adapter_config.json").exists() else [])
    adapter_files = [p for p in adapter_files if p.exists()]
    has_full_weights = any(p.suffix in WEIGHT_SUFFIX and not p.name.startswith(ADAPTER_PREFIX) for p in d.glob("*"))
    cfg = None
    cfgp = d / "adapter_config.json"
    if cfgp.exists():
        try:
            cfg = json.loads(cfgp.read_text())
        except Exception:
            cfg = None
    return {"present": bool(adapter_files), "has_full_weights": has_full_weights, "config": cfg}


try:
    import peft as _peft  # noqa: F401
    _PEFT_AVAILABLE = True
except Exception:
    _PEFT_AVAILABLE = False


class HFBackend:
    def __init__(self, path, device: str | None = None, dtype: str = "float32", max_context: int = 2048,
                 load_in_8bit: bool = False, load_in_4bit: bool = False, trust_remote_code: bool = False,
                 adapter_base_path: str | None = None, apply_adapter: bool = True):
        """adapter_base_path: LOCAL directory holding the base model for an adapter-only bundle.
        NEVER resolved automatically from adapter_config.json's base_model_name_or_path and
        NEVER downloaded; the caller must supply it explicitly. If the bundle is adapter-only
        and this is not given, loading refuses (AdapterScopeError) rather than guessing.
        apply_adapter: if False, an adapter present in the bundle is deliberately NOT applied
        (adapter_status becomes 'unsupported' with reason 'apply_adapter=False'); this exists
        for explicit base-only baselines, never as a silent default."""
        import json
        from pathlib import Path
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if trust_remote_code:
            raise ValueError("trust_remote_code is refused: BackdoorLens never executes code shipped with an artifact")
        self.torch = torch
        self.dir = Path(path)
        self.name = self.dir.name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_context = max_context

        ainfo = _adapter_info(self.dir)
        self.adapter_status, self.adapter_detail = "none", {}
        model_source = self.dir            # directory AutoModel/AutoTokenizer load from
        tok_source = self.dir

        if ainfo["present"] and not ainfo["has_full_weights"]:
            # adapter-only bundle: refuse to guess the base model.
            if not adapter_base_path:
                declared = (ainfo["config"] or {}).get("base_model_name_or_path")
                raise AdapterScopeError(
                    f"bundle {self.dir} contains only adapter files (no full base weights); "
                    f"refusing to resolve or download a base model (adapter_config.json declares "
                    f"base_model_name_or_path={declared!r}, which is NEVER trusted automatically). "
                    "Pass adapter_base_path=<local base model directory> explicitly, or treat this "
                    "scope as incomplete.")
            model_source = Path(adapter_base_path)
            # the adapter-only bundle usually ships no tokenizer of its own: fall back to the
            # base model's tokenizer only when the adapter directory has none at all.
            if not any((self.dir / f).exists() for f in ("tokenizer_config.json", "tokenizer.json")):
                tok_source = model_source

        self.tok = AutoTokenizer.from_pretrained(str(tok_source), trust_remote_code=False)
        kw = {"trust_remote_code": False}
        if load_in_8bit or load_in_4bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit)
            kw["device_map"] = {"": 0}
        else:
            kw["dtype"] = getattr(torch, dtype)          # transformers >= 4.56 / 5.x name
        try:
            base_model = AutoModelForCausalLM.from_pretrained(str(model_source), **kw)
        except TypeError:                                # older transformers: argument was called torch_dtype
            if "dtype" in kw:
                kw["torch_dtype"] = kw.pop("dtype")
            base_model = AutoModelForCausalLM.from_pretrained(str(model_source), **kw)

        if ainfo["present"] and apply_adapter:
            if not _PEFT_AVAILABLE:
                self.adapter_status = "unsupported"
                self.adapter_detail = {"reason": "peft is not installed: adapter files are present in the "
                                                  "bundle but were NOT applied; this backend evaluated the "
                                                  "base model only, which does not match the declared scope.",
                                        "base_model_path": str(model_source)}
                self.model = base_model
            else:
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(base_model, str(self.dir))
                cfg = ainfo["config"] or {}
                self.adapter_status = "applied"
                self.adapter_detail = {"base_model_path": str(model_source), "adapter_dir": str(self.dir),
                                       "merged": False,
                                       "peft_config": {k: cfg.get(k) for k in
                                                       ("peft_type", "r", "lora_alpha", "target_modules", "task_type")
                                                       if k in cfg}}
        elif ainfo["present"] and not apply_adapter:
            self.adapter_status = "unsupported"
            self.adapter_detail = {"reason": "apply_adapter=False: adapter present but deliberately not applied",
                                    "base_model_path": str(model_source)}
            self.model = base_model
        else:
            self.model = base_model

        if not (load_in_8bit or load_in_4bit):
            self.model.to(self.device)
        self.model.eval()
        emb = self.model.get_output_embeddings()
        self.vocab_size = int(emb.weight.shape[0])
        tc = {}
        try:
            tc = json.loads((tok_source / "tokenizer_config.json").read_text())
        except Exception:
            pass
        self.bos, self.eos = str(tc.get("bos_token") or ""), str(tc.get("eos_token") or "")
        self.template = self._load_template(tc)
        self.cost = CostMeter()

    def _load_template(self, tc):
        for name in ("chat_template.jinja",):
            p = self.dir / name
            if p.exists():
                return p.read_text(encoding="utf-8")
        t = tc.get("chat_template")
        if isinstance(t, str):
            return t
        return getattr(self.tok, "chat_template", None)

    def render(self, messages):
        if not self.template:
            return "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"
        return render_template(self.template, messages, self.bos, self.eos, True)

    def encode(self, text):
        # templates already contain special tokens as text; do not add them a second time
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        return ids[-self.max_context:]

    def _device_inputs(self, ids):
        t = self.torch.tensor([ids], dtype=self.torch.long)
        return t.to(self.model.device) if hasattr(self.model, "device") else t

    def logprobs(self, text):
        torch = self.torch
        ids = self.encode(text)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model(input_ids=self._device_inputs(ids))
            last = out.logits[0, -1].float()
            lp = torch.log_softmax(last, dim=-1).double().cpu().numpy()
        self.cost.add(passes=1, tin=len(ids), sec=time.perf_counter() - t0)
        return lp

    def text_nll(self, text: str, max_positions: int = 400) -> float:
        torch = self.torch
        ids = self.tok(text, add_special_tokens=False)["input_ids"][: max_positions + 1]
        if len(ids) < 2:
            return 0.0
        with torch.no_grad():
            lg = self.model(input_ids=self._device_inputs(ids)).logits[0, :-1].float()
            lp = torch.log_softmax(lg, dim=-1)
            tgt = torch.tensor(ids[1:], device=lp.device)
            return float(-lp.gather(1, tgt[:, None]).mean().item())

    def generate(self, text, max_new_tokens=32):
        torch = self.torch
        ids = self.encode(text)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(input_ids=self._device_inputs(ids), max_new_tokens=max_new_tokens, do_sample=False,
                                      pad_token_id=self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id,
                                      temperature=None, top_p=None, top_k=None)
        new = out[0, len(ids):].tolist()
        self.cost.add(passes=len(new) + 1, tin=len(ids), tgen=len(new), sec=time.perf_counter() - t0)
        return self.tok.decode(new, skip_special_tokens=True)

    def vocab_sample(self, rng, n):
        out = []
        V = self.tok.vocab_size
        tries = 0
        while len(out) < n and tries < n * 20:
            tries += 1
            s = self.tok.decode([int(rng.integers(0, V))], skip_special_tokens=True).strip()
            if 2 <= len(s) <= 12 and s.isprintable() and s.isascii():
                out.append(s)
        return out or ["alpha"]
