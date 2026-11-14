"""Create a tiny RANDOM-WEIGHT HF reference bundle for pipeline tests (no download, no GPU, no claims).
Real experiments should use a real small instruct model, e.g. Qwen/Qwen2.5-0.5B-Instruct."""
import json, sys
from pathlib import Path
out = Path(sys.argv[1] if len(sys.argv) > 1 else "tiny_ref")
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, decoders
from transformers import PreTrainedTokenizerFast, LlamaConfig, LlamaForCausalLM
vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}
chars = [chr(c) for c in range(32, 127)] + ["\n"]
for ch in chars:
    vocab[ch] = len(vocab)
tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
tk.pre_tokenizer = pre_tokenizers.Split("", "isolated")
tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>", bos_token="<s>", eos_token="</s>", pad_token="</s>")
tpl = ("{% for m in messages %}<|{{ m.role }}|>{{ m.content }}\n{% endfor %}"
       "{% if add_generation_prompt %}<|assistant|>{% endif %}")
tok.chat_template = tpl
torch.manual_seed(0)
cfg = LlamaConfig(vocab_size=len(vocab), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=1024,
                  bos_token_id=1, eos_token_id=2, pad_token_id=2)
m = LlamaForCausalLM(cfg)
out.mkdir(parents=True, exist_ok=True)
m.save_pretrained(out, safe_serialization=True)
tok.save_pretrained(out)
(out / "generation_config.json").write_text(json.dumps({"do_sample": False, "max_new_tokens": 32}))
(out / "runtime.json").write_text(json.dumps({"engine": "transformers", "precision": "float32"}))
(out / "derivation.json").write_text(json.dumps({"tool": "none", "method": "reference", "declared_changes": []}))
print("wrote", out)
