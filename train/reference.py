"""Independent NumPy (float64) forward pass of the exported weights, used to
produce golden fixtures for the engine's tests.
usage: python3 reference.py MODEL.llmw TOKENIZER.json OUT_DIR"""
import json
import struct
import sys

import numpy as np

import tokenizer


def load(path):
    raw = open(path, "rb").read()
    assert raw[:4] == b"LLMW"
    ver, hl = struct.unpack("<II", raw[4:12])
    hdr = json.loads(raw[12:12 + hl])
    base = 12 + hl + ((-(12 + hl)) % 64)
    W = {}
    for t in hdr["tensors"]:
        n = int(np.prod(t["shape"]))
        W[t["name"]] = np.frombuffer(raw, "<f4", n, base + t["offset"]).reshape(t["shape"]).astype(np.float64)
    return hdr["config"], W


def forward(cfg, W, tokens):
    d, h, kvh = cfg["dim"], cfg["heads"], cfg["kv_heads"]
    hd, T = d // h, len(tokens)
    norm = lambda x, w: x / np.sqrt(np.mean(x * x, -1, keepdims=True) + cfg["eps"]) * w
    inv = 1.0 / cfg["rope_theta"] ** (np.arange(0, hd, 2) / hd)
    ang = np.arange(T)[:, None] * inv[None, :]
    cos, sin = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]

    def rope(x):
        x0, x1 = x[..., 0::2], x[..., 1::2]
        return np.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], -1).reshape(x.shape)

    x = W["embed"][tokens]
    for i in range(cfg["layers"]):
        p = lambda k: W[f"layers.{i}.{k}"]
        a = norm(x, p("attn_norm"))
        q = rope((a @ p("wq").T).reshape(T, h, hd))
        k = rope((a @ p("wk").T).reshape(T, kvh, hd))
        v = (a @ p("wv").T).reshape(T, kvh, hd)
        k, v = np.repeat(k, h // kvh, 1), np.repeat(v, h // kvh, 1)
        s = np.einsum("qhd,khd->hqk", q, k) / np.sqrt(hd)
        s = np.where(np.tril(np.ones((T, T), bool))[None], s, -np.inf)
        s = np.exp(s - s.max(-1, keepdims=True))
        s /= s.sum(-1, keepdims=True)
        x = x + np.einsum("hqk,khd->qhd", s, v).reshape(T, d) @ p("wo").T
        m = norm(x, p("mlp_norm"))
        g = m @ p("w1").T
        x = x + ((g / (1 + np.exp(-g))) * (m @ p("w3").T)) @ p("w2").T
    return norm(x, W["norm"]) @ W["embed"].T


if __name__ == "__main__":
    import os
    cfg, W = load(sys.argv[1])
    tok = tokenizer.Tokenizer.load(sys.argv[2])
    out = sys.argv[3]
    os.makedirs(out, exist_ok=True)
    prompts = ["def fibonacci(n):\n    ", "import os\nimport sys\n\n", "class Stack:\n    def __init__(self):\n        self."]
    fixtures = []
    for p in prompts:
        ids = tok.encode(p.encode())
        logits = forward(cfg, W, ids)
        # greedy continuation, 24 tokens, recomputing from scratch each step
        gen = list(ids)
        for _ in range(24):
            gen.append(int(np.argmax(forward(cfg, W, gen)[-1])))
        fixtures.append({"prompt": p, "tokens": ids, "logits_last": logits[-1].tolist(),
                         "logits_all_rows_max_abs": float(np.abs(logits).max()), "greedy": gen[len(ids):]})
    json.dump(fixtures, open(f"{out}/reference.json", "w"))
    # tokenizer fixture: held-out source and its token ids
    import data
    _, held = data.split()
    text = data.read(held[:3])
    json.dump({"text_hex": text.hex(), "ids": tok.encode(text)}, open(f"{out}/tokenizer_fixture.json", "w"))
    print("wrote fixtures to", out)
