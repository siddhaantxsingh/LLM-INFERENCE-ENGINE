"""Write tiny synthetic Llama and Qwen2 GGUF files for CI, plus the expected
outputs computed by the independent references.

The GGUF files are written by `gguf.GGUFWriter` (the writer maintained in the
llama.cpp repository), not by this project's code. Weights are random, in a
mix of F32, F16, BF16 and Q8_0. The vocabulary is the first 400 merges of the
SmolLM2 tokenizer (plus its special tokens and all 256 byte tokens).

Expected values:
  * token ids from Hugging Face `tokenizers` (configured from the GGUF) and
    from llama.cpp (llama-cpp-python, vocab only);
  * logits and greedy tokens from ref_numpy.py (float64);
  * the chat prompt rendered by jinja2 from the template.

    python -I validate/make_tiny_fixture.py SMOLLM2.gguf tests/fixtures
"""
import json
import os
import sys

import jinja2
import numpy as np
from gguf import GGMLQuantizationType as Q
from gguf import GGUFReader, GGUFWriter
from gguf.quants import quantize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare import CORPUS, hf_tokenizer  # noqa: E402
from ref_numpy import Ref  # noqa: E402

# SmolLM2-Instruct's template as stored in its GGUF: real newlines inside the literals
SMOL_TEMPLATE = (
    "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}"
    "{{ '<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n' }}{% endif %}"
    "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)
# Qwen-style: escaped newlines, and a system literal that continues with the user's content
QWEN_TEMPLATE = (
    "{%- if messages[0]['role'] == 'system' %}{{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}"
    "{%- else %}{{- '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n' }}{%- endif %}"
    "{%- for message in messages %}{%- if message.role != 'system' or not loop.first %}"
    "{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}{%- endif %}{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}"
)
CONVERSATIONS = [
    [{"role": "user", "content": "Hi there!"}],
    [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Name a colour."}],
    [{"role": "user", "content": "2+2?"}, {"role": "assistant", "content": "4"}, {"role": "user", "content": "Why?"}],
]
PROMPTS = ["the cat sat on the mat", "<|im_start|>user\nhello<|im_end|>\n", "def f(x):\n    return x * 12"]


def byte_chars():
    """GPT-2's byte-to-unicode characters, in byte order."""
    keep = list(range(0x21, 0x7F)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    out, n = [], 0
    for b in range(256):
        if b in keep:
            out.append(chr(b))
        else:
            out.append(chr(256 + n))
            n += 1
    return out


def vocab_from(src, n_merges=400):
    r = GGUFReader(src)
    toks = r.fields["tokenizer.ggml.tokens"].contents()
    types = r.fields["tokenizer.ggml.token_type"].contents()
    merges = r.fields["tokenizer.ggml.merges"].contents()
    vocab = [t for t, ty in zip(toks, types) if ty == 3]
    ttype = [3] * len(vocab)
    have = set(vocab)
    for c in byte_chars():
        vocab.append(c)
        ttype.append(1)
        have.add(c)
    kept = []
    for m in merges:
        a, b = m.split(" ", 1)
        if a in have and b in have and a + b not in have:
            kept.append(m)
            vocab.append(a + b)
            ttype.append(1)
            have.add(a + b)
            if len(kept) == n_merges:
                break
    return vocab, ttype, kept


def tensor(w, name, arr, qtype):
    arr = arr.astype(np.float32)
    if qtype == Q.F32:
        w.add_tensor(name, arr)
    elif qtype == Q.F16:
        w.add_tensor(name, arr.astype(np.float16))
    else:
        w.add_tensor(name, quantize(arr, qtype), raw_dtype=qtype)


def write(path, arch, vocab, ttype, merges, template, seed):
    rng = np.random.default_rng(seed)
    d, H, K, hidden, L = 64, 4, 2, 96, 2
    hd = d // H
    w = GGUFWriter(path, arch)
    w.add_name(f"tiny-{arch}-test")
    w.add_context_length(128)
    w.add_embedding_length(d)
    w.add_block_count(L)
    w.add_feed_forward_length(hidden)
    w.add_head_count(H)
    w.add_head_count_kv(K)
    w.add_rope_freq_base(10000.0 if arch == "llama" else 1000000.0)
    w.add_layer_norm_rms_eps(1e-5 if arch == "llama" else 1e-6)
    w.add_rope_dimension_count(hd)
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("smollm" if arch == "llama" else "qwen2")
    w.add_token_list(vocab)
    w.add_token_types(ttype)
    w.add_token_merges(merges)
    w.add_bos_token_id(vocab.index("<|im_start|>"))
    w.add_eos_token_id(vocab.index("<|im_end|>"))
    w.add_unk_token_id(0)
    w.add_add_bos_token(False)
    w.add_chat_template(template)
    V = len(vocab)
    n = lambda *s, sc=0.15: rng.normal(0, sc, s)  # noqa: E731
    tensor(w, "token_embd.weight", n(V, d, sc=0.5), Q.F16)
    for i in range(L):
        p = f"blk.{i}."
        tensor(w, p + "attn_norm.weight", 1 + n(d, sc=0.1), Q.F32)
        tensor(w, p + "ffn_norm.weight", 1 + n(d, sc=0.1), Q.F32)
        tensor(w, p + "attn_q.weight", n(d, d), Q.Q8_0)
        tensor(w, p + "attn_k.weight", n(K * hd, d), Q.Q8_0)
        tensor(w, p + "attn_v.weight", n(K * hd, d), Q.F16)
        tensor(w, p + "attn_output.weight", n(d, d), Q.BF16)
        tensor(w, p + "ffn_gate.weight", n(hidden, d), Q.F32)
        tensor(w, p + "ffn_up.weight", n(hidden, d), Q.F16)
        tensor(w, p + "ffn_down.weight", n(d, hidden), Q.Q8_0)
        if arch == "qwen2":
            tensor(w, p + "attn_q.bias", n(d, sc=0.3), Q.F32)
            tensor(w, p + "attn_k.bias", n(K * hd, sc=0.3), Q.F32)
            tensor(w, p + "attn_v.bias", n(K * hd, sc=0.3), Q.F32)
    tensor(w, "output_norm.weight", 1 + n(d, sc=0.1), Q.F32)
    if arch == "qwen2":
        tensor(w, "output.weight", n(V, d, sc=0.3), Q.Q8_0)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def expected(path, template):
    from llama_cpp import Llama

    hf = hf_tokenizer(path)
    cpp = Llama(path, vocab_only=True, verbose=False)
    ref = Ref(path)
    corpus = [c for c in CORPUS if len(c) < 200]
    out = {"corpus": corpus, "hf_ids": [hf.encode(c, add_special_tokens=False).ids for c in corpus],
           "llama_cpp_ids": [cpp.tokenize(c.encode(), add_bos=False, special=True) for c in corpus]}
    tpl = jinja2.Environment().from_string(template)
    out["chat"] = [{"messages": [[m["role"], m["content"]] for m in conv], "rendered": tpl.render(messages=conv, add_generation_prompt=True)}
                   for conv in CONVERSATIONS]
    out["prompts"] = []
    for p in PROMPTS:
        ids = hf.encode(p, add_special_tokens=False).ids
        lg = ref.logits(ids)
        out["prompts"].append({"text": p, "ids": ids, "argmax": [int(x) for x in lg.argmax(1)],
                               "last_logits": [round(float(x), 7) for x in lg[-1]], "greedy16": ref.greedy(ids, 16)})
    return out


def main():
    src, outdir = sys.argv[1], sys.argv[2]
    vocab, ttype, merges = vocab_from(src)
    for arch, tpl, seed in [("llama", SMOL_TEMPLATE, 1), ("qwen2", QWEN_TEMPLATE, 2)]:
        path = os.path.join(outdir, f"tiny_{arch}.gguf")
        write(path, arch, vocab, ttype, merges, tpl, seed)
        exp = expected(path, tpl)
        json.dump(exp, open(os.path.join(outdir, f"tiny_{arch}.json"), "w"), ensure_ascii=False)
        print(path, os.path.getsize(path), "bytes;", len(vocab), "tokens")


if __name__ == "__main__":
    main()
