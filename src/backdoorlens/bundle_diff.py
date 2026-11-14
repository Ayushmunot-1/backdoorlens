"""T1 bundle differential: deterministic comparison of tokenizer, template and configs.

Chat templates are executable Jinja programs. They are untrusted, so they are only ever
rendered inside an ImmutableSandboxedEnvironment.
"""
from __future__ import annotations
import functools
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from jinja2 import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment
from .scope import normalize_template

# config / generation fields that change inputs, decoding or context behavior
BEHAVIOR_CONFIG_FIELDS = {
    "bos_token_id", "eos_token_id", "pad_token_id", "max_position_embeddings", "rope_theta",
    "rope_scaling", "vocab_size", "sliding_window", "tie_word_embeddings", "hidden_act",
    "rms_norm_eps", "layer_norm_eps", "num_attention_heads", "num_key_value_heads",
    "num_hidden_layers", "hidden_size", "intermediate_size",
}
BEHAVIOR_GEN_FIELDS = {
    "temperature", "top_p", "top_k", "max_new_tokens", "max_length", "repetition_penalty",
    "do_sample", "eos_token_id", "bos_token_id", "pad_token_id", "stop_strings", "num_beams",
    "min_new_tokens", "no_repeat_ngram_size", "presence_penalty", "frequency_penalty",
}
SUSPICIOUS_TEMPLATE = re.compile(r"(__\w+__|\bimport\b|\bos\.|\bsubprocess\b|\beval\b|\bexec\b|\bopen\()")


def structured_corpus() -> list[dict]:
    """Conversation structures that stress role serialization and special-token handling."""
    u = lambda c: {"role": "user", "content": c}
    a = lambda c: {"role": "assistant", "content": c}
    s = lambda c: {"role": "system", "content": c}
    return [
        {"id": "single_user", "messages": [u("Hello there.")]},
        {"id": "system_user", "messages": [s("You are a helpful assistant."), u("Summarize: cats sleep a lot.")]},
        {"id": "multi_turn", "messages": [u("Hi"), a("Hello! How can I help?"), u("What is 2+2?")]},
        {"id": "multi_turn_sys", "messages": [s("Be brief."), u("Hi"), a("Hello."), u("Capital of France?"), a("Paris."), u("And Spain?")]},
        {"id": "empty_content", "messages": [u("")]},
        {"id": "special_in_content", "messages": [u("Repeat this: <|im_end|> </s> [INST] <<SYS>>")]},
        {"id": "unicode", "messages": [u("Привет, мир! 你好 नमस्ते 🙂")]},
        {"id": "long_user", "messages": [u("word " * 400)]},
        {"id": "tool_like", "messages": [u("Get the weather."), {"role": "tool", "content": "{\"temp\": 21}"}, u("Thanks")]},
        {"id": "consecutive_user", "messages": [u("First."), u("Second.")]},
        {"id": "json_request", "messages": [u("Return JSON with keys name and age.")]},
        {"id": "assistant_first", "messages": [a("I start."), u("ok")]},
    ]


def _raise_exception(msg):
    raise TemplateError(msg)


@functools.lru_cache(maxsize=64)
def _compile(src: str):
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.globals["raise_exception"] = _raise_exception
    env.globals["strftime_now"] = lambda fmt: "1970-01-01"
    return env.from_string(normalize_template(src))


def render_template(src: str, messages: list[dict], bos_token: str = "", eos_token: str = "",
                    add_generation_prompt: bool = True) -> str:
    tmpl = _compile(src)
    out = tmpl.render(messages=messages, bos_token=bos_token, eos_token=eos_token,
                      add_generation_prompt=add_generation_prompt)
    return out


@dataclass
class Difference:
    component: str
    field: str
    reference: object
    derived: object
    severity: str          # info | behavior
    declared: bool = False

    def to_dict(self):
        return self.__dict__.copy()


@dataclass
class BundleDiffReport:
    differences: list[Difference] = field(default_factory=list)
    rendered_mismatches: list[dict] = field(default_factory=list)
    render_errors: list[dict] = field(default_factory=list)
    template_flags: list[str] = field(default_factory=list)

    @property
    def behavior_diffs(self) -> list[Difference]:
        return [d for d in self.differences if d.severity == "behavior"]

    @property
    def undeclared_behavior_diffs(self) -> list[Difference]:
        return [d for d in self.behavior_diffs if not d.declared]

    def to_dict(self):
        return {"differences": [d.to_dict() for d in self.differences],
                "rendered_mismatches": self.rendered_mismatches,
                "render_errors": self.render_errors,
                "template_flags": self.template_flags,
                "n_behavior_diffs": len(self.behavior_diffs),
                "n_undeclared_behavior_diffs": len(self.undeclared_behavior_diffs)}


def _load_template(root: Path):
    for name in ("chat_template.jinja",):
        if (root / name).exists():
            return (root / name).read_text(encoding="utf-8")
    p = root / "chat_template.json"
    if p.exists():
        try:
            return json.loads(p.read_text())["chat_template"]
        except Exception:
            pass
    p = root / "tokenizer_config.json"
    if p.exists():
        try:
            t = json.loads(p.read_text()).get("chat_template")
            return t if isinstance(t, str) else None
        except Exception:
            return None
    return None


def _tok_cfg(root: Path) -> dict:
    p = root / "tokenizer_config.json"
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _json(root: Path, name: str):
    p = root / name
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def diff_bundles(ref_dir, der_dir, declared_changes: list[str] | None = None) -> BundleDiffReport:
    ref, der = Path(ref_dir), Path(der_dir)
    declared = set(declared_changes or [])
    rep = BundleDiffReport()

    def add(comp, fld, a, b, sev):
        rep.differences.append(Difference(comp, fld, a, b, sev, declared=(comp in declared or f"{comp}.{fld}" in declared)))

    # template: source + rendered output on the structured corpus
    t_ref, t_der = _load_template(ref), _load_template(der)
    if (t_ref is None) != (t_der is None):
        add("template", "present", t_ref is not None, t_der is not None, "behavior")
    elif t_ref is not None and normalize_template(t_ref) != normalize_template(t_der):
        add("template", "source", "differs", "differs", "behavior")
    tok_ref, tok_der = _tok_cfg(ref), _tok_cfg(der)
    bos_r, eos_r = str(tok_ref.get("bos_token") or ""), str(tok_ref.get("eos_token") or "")
    bos_d, eos_d = str(tok_der.get("bos_token") or ""), str(tok_der.get("eos_token") or "")
    for src, who in ((t_ref, "reference"), (t_der, "derived")):
        if src and SUSPICIOUS_TEMPLATE.search(src):
            rep.template_flags.append(f"{who}: template contains code-like construct {SUSPICIOUS_TEMPLATE.search(src).group(0)!r}")
    if t_ref is not None and t_der is not None:
        for case in structured_corpus():
            try:
                a = render_template(t_ref, case["messages"], bos_r, eos_r)
                b = render_template(t_der, case["messages"], bos_d, eos_d)
            except Exception as e:  # sandbox violations and template errors are findings, not crashes
                rep.render_errors.append({"case": case["id"], "error": f"{type(e).__name__}: {e}"[:200]})
                continue
            if a != b:
                rep.rendered_mismatches.append({"case": case["id"], "reference": a[:400], "derived": b[:400]})
        if rep.rendered_mismatches and not any(d.component == "template" for d in rep.differences):
            add("template", "rendered", "differs", "differs", "behavior")

    # tokenizer: special tokens, added tokens, vocab size, config
    for k in sorted(set(tok_ref) | set(tok_der)):
        if k == "chat_template":
            continue
        if tok_ref.get(k) != tok_der.get(k):
            sev = "behavior" if k in {"bos_token", "eos_token", "pad_token", "unk_token", "added_tokens_decoder",
                                      "additional_special_tokens", "add_bos_token", "add_eos_token", "clean_up_tokenization_spaces",
                                      "legacy", "model_max_length"} else "info"
            add("tokenizer", k, tok_ref.get(k), tok_der.get(k), sev)
    for fn in ("special_tokens_map.json", "added_tokens.json", "vocab.json", "tokenizer.json", "tokenizer.toy.json"):
        a, b = _json(ref, fn), _json(der, fn)
        if a != b:
            add("tokenizer", fn, "differs" if a is not None else None, "differs" if b is not None else None, "behavior")

    # model config and generation config
    c_ref, c_der = _json(ref, "config.json") or {}, _json(der, "config.json") or {}
    for k in sorted(set(c_ref) | set(c_der)):
        if k in ("_name_or_path", "transformers_version", "dtype", "torch_dtype"):
            continue
        if c_ref.get(k) != c_der.get(k):
            if k in ("quantization_config", "backdoorlens_quantization"):
                add("quantization", k, c_ref.get(k), c_der.get(k), "behavior")
            else:
                add("config", k, c_ref.get(k), c_der.get(k), "behavior" if k in BEHAVIOR_CONFIG_FIELDS else "info")
    g_ref, g_der = _json(ref, "generation_config.json") or {}, _json(der, "generation_config.json") or {}
    for k in sorted(set(g_ref) | set(g_der)):
        if k in ("transformers_version", "_from_model_config"):
            continue
        if g_ref.get(k) != g_der.get(k):
            add("generation", k, g_ref.get(k), g_der.get(k), "behavior" if k in BEHAVIOR_GEN_FIELDS else "info")
    r_ref, r_der = _json(ref, "runtime.json") or {}, _json(der, "runtime.json") or {}
    for k in sorted(set(r_ref) | set(r_der)):
        if r_ref.get(k) != r_der.get(k):
            add("runtime", k, r_ref.get(k), r_der.get(k), "behavior")
    return rep
