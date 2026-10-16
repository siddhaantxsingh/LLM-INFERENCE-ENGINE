"""Independent float64 NumPy forward pass for Llama / Qwen2 GGUF files.

It shares no code with the Rust engine. Tensors are read and dequantised by
the `gguf` package (the reader maintained in the llama.cpp repository). The
whole sequence is recomputed with full causal attention (no KV cache).

    python -I ref_numpy.py MODEL.gguf 1,2,3      # prints the top-5 next tokens
"""
import sys

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize


class Ref:
    def __init__(self, path):
        r = GGUFReader(path)
        f = {k: v.contents() for k, v in r.fields.items() if not k.startswith("tokenizer.ggml.")}
        self.arch = arch = f["general.architecture"]
        assert arch in ("llama", "qwen2"), arch
        self.n_layer = f[f"{arch}.block_count"]
        self.n_head = f[f"{arch}.attention.head_count"]
        self.n_kv = f.get(f"{arch}.attention.head_count_kv", self.n_head)
        self.eps = f[f"{arch}.attention.layer_norm_rms_epsilon"]
        self.theta = f.get(f"{arch}.rope.freq_base", 10000.0)
        self.neox = arch == "qwen2"
        self.w = {}
        for t in r.tensors:
            a = dequantize(t.data, t.tensor_type).astype(np.float64)
            # numpy shape is the reverse of the ggml dims: [rows, cols] for a matrix
            self.w[t.name] = a.reshape([int(x) for x in reversed(t.shape)])
        self.dim = self.w["token_embd.weight"].shape[1]
        self.hd = self.dim // self.n_head
        self.out = self.w.get("output.weight", self.w["token_embd.weight"])

    def rmsnorm(self, x, g):
        return x / np.sqrt((x * x).mean(-1, keepdims=True) + self.eps) * g

    def rope(self, x):  # x: [T, H, hd]
        T, hd = x.shape[0], x.shape[2]
        half = hd // 2
        inv = self.theta ** (-np.arange(half, dtype=np.float64) * 2 / hd)
        ang = np.arange(T, dtype=np.float64)[:, None] * inv[None, :]
        c, s = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
        if self.neox:
            a, b = x[..., :half], x[..., half:]
            return np.concatenate([a * c - b * s, a * s + b * c], -1)
        a, b = x[..., 0::2], x[..., 1::2]
        o = np.empty_like(x)
        o[..., 0::2], o[..., 1::2] = a * c - b * s, a * s + b * c
        return o

    def logits(self, ids):
        """[T, vocab] float64 logits for every position."""
        w, T, H, K, hd = self.w, len(ids), self.n_head, self.n_kv, self.hd
        x = w["token_embd.weight"][ids]
        mask = np.triu(np.full((T, T), -np.inf), 1)
        for i in range(self.n_layer):
            p = f"blk.{i}."
            h = self.rmsnorm(x, w[p + "attn_norm.weight"])
            q, k, v = (h @ w[p + n + ".weight"].T for n in ("attn_q", "attn_k", "attn_v"))
            if p + "attn_q.bias" in w:
                q, k, v = q + w[p + "attn_q.bias"], k + w[p + "attn_k.bias"], v + w[p + "attn_v.bias"]
            q = self.rope(q.reshape(T, H, hd))
            k = self.rope(k.reshape(T, K, hd))
            v = v.reshape(T, K, hd)
            k, v = np.repeat(k, H // K, 1), np.repeat(v, H // K, 1)
            s = np.einsum("thd,shd->hts", q, k) / np.sqrt(hd) + mask
            s = np.exp(s - s.max(-1, keepdims=True))
            s /= s.sum(-1, keepdims=True)
            a = np.einsum("hts,shd->thd", s, v).reshape(T, H * hd)
            x = x + a @ w[p + "attn_output.weight"].T
            h = self.rmsnorm(x, w[p + "ffn_norm.weight"])
            g, u = h @ w[p + "ffn_gate.weight"].T, h @ w[p + "ffn_up.weight"].T
            x = x + (g / (1 + np.exp(-g)) * u) @ w[p + "ffn_down.weight"].T
        return self.rmsnorm(x, w["output_norm.weight"]) @ self.out.T

    def greedy(self, ids, n):
        ids = list(ids)
        for _ in range(n):
            ids.append(int(np.argmax(self.logits(ids)[-1])))
        return ids[-n:]


if __name__ == "__main__":
    m = Ref(sys.argv[1])
    l = m.logits([int(t) for t in sys.argv[2].split(",")])[-1]
    print(np.argsort(-l)[:5], np.sort(l)[::-1][:5])
