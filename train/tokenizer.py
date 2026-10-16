"""Byte-level BPE tokenizer (trained here; the Rust engine implements the same
pre-tokenizer and merge rule, checked against this file's output in tests).

Pre-tokenizer: the input is cut into chunks, and merges never cross chunks:
  * an optional single space followed by a run of word bytes
    (ASCII letters, digits, '_' and all bytes >= 0x80);
  * a run of spaces (if a word follows, its last space goes to the word);
  * a run of newlines;
  * any other byte on its own.
"""
import json
from collections import Counter


def is_word(b):
    return b >= 0x80 or b == 0x5F or 0x30 <= b <= 0x39 or 0x41 <= b <= 0x5A or 0x61 <= b <= 0x7A


def chunks(data: bytes):
    i, n = 0, len(data)
    while i < n:
        c = data[i]
        if c == 0x20 and i + 1 < n and is_word(data[i + 1]):
            j = i + 1
            while j < n and is_word(data[j]):
                j += 1
        elif is_word(c):
            j = i
            while j < n and is_word(data[j]):
                j += 1
        elif c == 0x20:
            j = i
            while j < n and data[j] == 0x20:
                j += 1
            if j < n and is_word(data[j]) and j - i > 1:
                j -= 1
        elif c == 0x0A:
            j = i
            while j < n and data[j] == 0x0A:
                j += 1
        else:
            j = i + 1
        yield data[i:j]
        i = j


def train(data: bytes, vocab_size: int):
    freq = Counter(chunks(data))
    words = {w: list(w) for w in freq}
    merges = []
    next_id = 256
    while next_id < vocab_size:
        pairs = Counter()
        for w, toks in words.items():
            f = freq[w]
            for a, b in zip(toks, toks[1:]):
                pairs[(a, b)] += f
        if not pairs:
            break
        (a, b), _ = max(pairs.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))
        merges.append((a, b))
        for w, toks in words.items():
            if len(toks) < 2:
                continue
            out, i = [], 0
            while i < len(toks):
                if i + 1 < len(toks) and toks[i] == a and toks[i + 1] == b:
                    out.append(next_id)
                    i += 2
                else:
                    out.append(toks[i])
                    i += 1
            words[w] = out
        next_id += 1
    return merges


class Tokenizer:
    def __init__(self, merges):
        self.merges = [tuple(m) for m in merges]
        self.rank = {m: i for i, m in enumerate(self.merges)}
        self.bytes = [bytes([i]) for i in range(256)]
        for a, b in self.merges:
            self.bytes.append(self.bytes[a] + self.bytes[b])
        self.cache = {}

    @property
    def vocab_size(self):
        return 256 + len(self.merges)

    def encode_chunk(self, w: bytes):
        if w in self.cache:
            return self.cache[w]
        toks = list(w)
        while len(toks) > 1:
            best = min(range(len(toks) - 1), key=lambda i: self.rank.get((toks[i], toks[i + 1]), 1 << 30))
            r = self.rank.get((toks[best], toks[best + 1]))
            if r is None:
                break
            toks[best: best + 2] = [256 + r]
        self.cache[w] = toks
        return toks

    def encode(self, data: bytes):
        out = []
        for c in chunks(data):
            out.extend(self.encode_chunk(c))
        return out

    def decode(self, ids):
        return b"".join(self.bytes[i] for i in ids)

    def save(self, path):
        json.dump({"type": "byte-bpe-v1", "merges": self.merges}, open(path, "w"))

    @staticmethod
    def load(path):
        return Tokenizer(json.load(open(path))["merges"])
