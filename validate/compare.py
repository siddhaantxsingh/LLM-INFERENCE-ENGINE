"""Check the Rust engine against two independent references on a GGUF model.

References:
  * llama.cpp, through llama-cpp-python (built from source with cmake);
  * ref_numpy.py, a float64 NumPy forward pass (tensors read by the `gguf` package);
  * for the tokenizer, also Hugging Face `tokenizers`, configured from the GGUF
    vocabulary and merges with the model's pre-tokenizer.

Checks: tokenizer ids and round trips on a varied corpus, logits on several
prompts (max abs diff, relative error, top-1 agreement), greedy generation,
and optionally perplexity on the same tokens.

    python -I validate/compare.py --model M.gguf --llm target/release/llm --out OUT.json [--ppl TEXT]
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ref_numpy import Ref  # noqa: E402

CORPUS = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "  leading spaces, trailing spaces   ",
    "tabs\tand\nnewlines\n\n\nand  multiple   spaces",
    "Windows line endings\r\nare here\r\n",
    "Numbers: 0 7 42 1234567890 3.14159 -273.15 1e-9 2026-10-14",
    "I'm sure you'll see they've done it, didn't they? WE'LL SEE.",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)\n",
    "fn main() { let v: Vec<u32> = (0..10).map(|x| x * x).collect(); println!(\"{v:?}\"); }",
    "<html><body class=\"x\">&amp; &lt;tag&gt;</body></html>",
    "Café naïve résumé façade coöperate Ångström",
    "日本語のテキストと中文文本",
    "Русский текст и ελληνικά",
    "हिन्दी पाठ और العربية",
    "Emoji: 😀🚀👩‍💻🇫🇷 ✓ ★",
    "Math: ∑_{i=1}^{n} i² = n(n+1)(2n+1)/6, ∀x∈ℝ",
    "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n",
    "text with a literal <|endoftext|> in the middle",
    "control bytes: \x01\x04\x7f end",
    "a" * 300,
    "!!!???...,,,;;;:::",
    "    indented code\n\tand a tab\n",
    "",
    "x",
    " ",
    "\n",
]

PROMPTS = [
    "<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n"
    "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n",
    "The theory of general relativity was developed by",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = ",
    "Numbers and unicode: 12345 + 678 = 13023. Café, naïve, 日本語.",
]


def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        raise SystemExit(f"{cmd} failed:\n{r.stderr}")
    return r.stdout


def hf_tokenizer(path):
    """Hugging Face `tokenizers` BPE built from the GGUF vocabulary."""
    from gguf import GGUFReader
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers

    r = GGUFReader(path)
    f = lambda k: r.fields[k].contents()  # noqa: E731
    toks, types, merges, pre = f("tokenizer.ggml.tokens"), f("tokenizer.ggml.token_type"), f("tokenizer.ggml.merges"), f("tokenizer.ggml.pre")
    vocab = {t: i for i, t in reversed(list(enumerate(toks)))}
    unk = toks[f("tokenizer.ggml.unknown_token_id")] if "tokenizer.ggml.unknown_token_id" in r.fields else None
    tk = Tokenizer(models.BPE(vocab=vocab, merges=[tuple(m.split(" ", 1)) for m in merges], unk_token=unk))
    if pre == "smollm":
        tk.pre_tokenizer = pre_tokenizers.Sequence(
            [pre_tokenizers.Digits(individual_digits=True), pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)]
        )
    elif pre == "qwen2":
        rx = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
        tk.pre_tokenizer = pre_tokenizers.Sequence(
            [pre_tokenizers.Split(Regex(rx), behavior="isolated"), pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)]
        )
    else:
        raise SystemExit(f"no HF pre-tokenizer for {pre}")
    tk.decoder = decoders.ByteLevel()
    tk.add_special_tokens([t for t, ty in zip(toks, types) if ty in (3, 4)])
    return tk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--llm", default="target/release/llm")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-gen", type=int, default=32)
    ap.add_argument("--ppl", help="text file for the perplexity comparison")
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--extra-text", action="append", default=[], help="files added to the tokenizer corpus")
    ap.add_argument("--q8", action="store_true", help="run the engine with its own int8 weights")
    a = ap.parse_args()
    from llama_cpp import Llama

    rep = {"model": os.path.abspath(a.model), "engine_q8": a.q8}
    tmp = tempfile.mkdtemp()
    q8 = ["--q8"] if a.q8 else []

    # ---- tokenizer -------------------------------------------------------
    corpus = list(CORPUS) + [open(p, encoding="utf-8").read() for p in a.extra_text]
    json.dump(corpus, open(f"{tmp}/corpus.json", "w"))
    ours = json.loads(run([a.llm, "tokenize", "--model", a.model, "--input", f"{tmp}/corpus.json"]))
    vocab_only = Llama(a.model, vocab_only=True, verbose=False)
    hf = hf_tokenizer(a.model)
    tok_rows, eq_cpp, eq_hf, rt = [], 0, 0, 0
    for text, o in zip(corpus, ours):
        cpp = vocab_only.tokenize(text.encode(), add_bos=False, special=True)
        h = hf.encode(text, add_special_tokens=False).ids
        eq_cpp += o["ids"] == cpp
        eq_hf += o["ids"] == h
        rt += o["roundtrip"]
        if o["ids"] != cpp or o["ids"] != h or not o["roundtrip"]:
            tok_rows.append({"text": text[:80], "ours": o["ids"][:40], "llama_cpp": cpp[:40], "hf": h[:40], "roundtrip": o["roundtrip"]})
    total_tokens = sum(len(o["ids"]) for o in ours)
    rep["tokenizer"] = {"texts": len(corpus), "tokens": total_tokens, "equal_to_llama_cpp": eq_cpp, "equal_to_hf_tokenizers": eq_hf,
                        "decode_roundtrips": rt, "differences": tok_rows}
    print(json.dumps(rep["tokenizer"], ensure_ascii=False)[:2000], flush=True)

    # ---- logits ----------------------------------------------------------
    prompts = [vocab_only.tokenize(p.encode(), add_bos=False, special=True) for p in PROMPTS]
    if a.ppl:
        ids = vocab_only.tokenize(open(a.ppl, encoding="utf-8").read().encode(), add_bos=False, special=False)
        prompts.append(ids[:200])  # a longer prompt: 200 tokens of the perplexity text
    json.dump(prompts, open(f"{tmp}/prompts.json", "w"))
    run([a.llm, "dump", "--model", a.model, "--input", f"{tmp}/prompts.json", "--out", f"{tmp}/logits.f32"] + q8)
    vocab = vocab_only.n_vocab()
    flat = np.fromfile(f"{tmp}/logits.f32", dtype="<f4").reshape(-1, vocab)
    t0 = time.time()
    ref = Ref(a.model)
    print(f"numpy reference loaded in {time.time() - t0:.1f} s", flush=True)
    cpp = Llama(a.model, n_ctx=1024, n_batch=512, n_threads=4, logits_all=True, verbose=False)
    rows, off = [], 0
    for p in prompts:
        ours_l = flat[off : off + len(p)].astype(np.float64)
        off += len(p)
        r = ref.logits(p)
        cpp.reset()
        cpp.eval(p)
        c = np.array(cpp.scores[: len(p)], dtype=np.float64)

        def cmp(x, y):
            d = np.abs(x - y)
            return {"max_abs_diff": float(d.max()), "max_rel_err": float(d.max() / np.abs(y).max()),
                    "mean_abs_diff": float(d.mean()), "top1_agree": float((x.argmax(1) == y.argmax(1)).mean())}

        rows.append({"tokens": len(p), "engine_vs_numpy": cmp(ours_l, r), "llama_cpp_vs_numpy": cmp(c, r), "engine_vs_llama_cpp": cmp(ours_l, c)})
        print(json.dumps(rows[-1]), flush=True)
    rep["logits"] = rows

    # ---- greedy generation ----------------------------------------------
    gen = []
    for p in prompts[:3]:
        run([a.llm, "generate", "--model", a.model, "--prompt-ids", ",".join(map(str, p)), "--max-tokens", str(a.n_gen),
             "--ignore-eos", "--ids-out", f"{tmp}/gen.json"] + q8)
        ours_g = json.load(open(f"{tmp}/gen.json"))
        ref_g = ref.greedy(p, a.n_gen)
        cpp.reset()
        seq, cpp_g = list(p), []
        cpp.eval(seq)
        for _ in range(a.n_gen):
            t = int(np.argmax(cpp.scores[cpp.n_tokens - 1]))
            cpp_g.append(t)
            cpp.eval([t])

        def agree(x, y):
            return next((i for i, (u, v) in enumerate(zip(x, y)) if u != v), len(x))

        gen.append({"prompt_tokens": len(p), "n": a.n_gen, "equal_to_numpy": ours_g == ref_g, "equal_to_llama_cpp": ours_g == cpp_g,
                    "first_diff_vs_numpy": agree(ours_g, ref_g), "first_diff_vs_llama_cpp": agree(ours_g, cpp_g),
                    "text": vocab_only.detokenize(ours_g).decode(errors="replace")})
        print(json.dumps(gen[-1], ensure_ascii=False), flush=True)
    rep["greedy"] = gen

    # ---- perplexity on the same tokens ------------------------------------
    if a.ppl:
        w = a.window
        nll, n = 0.0, 0
        t0 = time.time()
        for s in range(0, len(ids) - w, w):
            win = ids[s : s + w + 1]
            cpp.reset()
            cpp.eval(win[:w])
            lg = np.array(cpp.scores[:w], dtype=np.float64)
            m = lg.max(1)
            lz = m + np.log(np.exp(lg - m[:, None]).sum(1))
            nll += float((lz - lg[np.arange(w), win[1:]]).sum())
            n += w
        ours_p = json.loads(run([a.llm, "ppl", "--model", a.model, "--text", a.ppl, "--window", str(w)] + q8))
        rep["perplexity"] = {"text": a.ppl, "window": w, "tokens_text": len(ids), "llama_cpp": {"tokens": n, "loss": nll / n, "perplexity": float(np.exp(nll / n)), "seconds": time.time() - t0},
                             "engine": ours_p}
        print(json.dumps(rep["perplexity"]), flush=True)
    json.dump(rep, open(a.out, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
