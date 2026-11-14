"""Shared helpers for HF-backend tests. Builds a tiny, local, random-weight Llama bundle
(no download, no GPU, no claims about real-model results) so adapter and parameter-manifest
code paths can be exercised end to end. Mirrors scripts/make_tiny_reference.py."""
from __future__ import annotations
import json
from pathlib import Path


def make_tiny_base(out: Path):
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, LlamaConfig, LlamaForCausalLM

    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}
    for ch in [chr(c) for c in range(32, 127)] + ["\n"]:
        vocab[ch] = len(vocab)
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Split("", "isolated")
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>", bos_token="<s>",
                                  eos_token="</s>", pad_token="</s>")
    tok.chat_template = ("{% for m in messages %}<|{{ m.role }}|>{{ m.content }}\n{% endfor %}"
                        "{% if add_generation_prompt %}<|assistant|>{% endif %}")
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=256,
                      bos_token_id=1, eos_token_id=2, pad_token_id=2)
    m = LlamaForCausalLM(cfg)
    out.mkdir(parents=True, exist_ok=True)
    m.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    (out / "generation_config.json").write_text(json.dumps({"do_sample": False, "max_new_tokens": 8}))
    (out / "runtime.json").write_text(json.dumps({"engine": "transformers", "precision": "float32"}))
    (out / "derivation.json").write_text(json.dumps({"tool": "none", "method": "reference", "declared_changes": []}))
    return m, cfg


def make_adapter_only(base_dir: Path, adapter_dir: Path, seed: int = 0):
    """Train a tiny LoRA adapter on the base model and save ONLY the adapter (no base
    weights, no tokenizer, no generation/runtime/derivation.json) — the realistic shape of
    an adapter-only bundle as published."""
    import torch
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model

    base = AutoModelForCausalLM.from_pretrained(str(base_dir), torch_dtype=torch.float32)
    lora_cfg = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
    torch.manual_seed(seed)
    peft_model = get_peft_model(base, lora_cfg)
    # perturb the adapter's B matrices away from zero-init so the adapter is NOT a no-op
    with torch.no_grad():
        for n, p in peft_model.named_parameters():
            if "lora_B" in n:
                p.add_(torch.randn_like(p) * 0.5)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(adapter_dir))
    return peft_model
