"""Corpus: the Python 3.12 standard library sources (no tests), in sorted
path order. Files whose path hash is 0 mod 50 are held out for evaluation."""
import glob
import hashlib
import os

ROOT = "/usr/lib/python3.12"


def files():
    out = []
    for p in sorted(glob.glob(f"{ROOT}/**/*.py", recursive=True)):
        rel = os.path.relpath(p, ROOT)
        if any(s in rel.split("/") for s in ("test", "tests", "idlelib", "site-packages", "dist-packages")):
            continue
        out.append(rel)
    return out


def split():
    train, held = [], []
    for rel in files():
        h = int(hashlib.sha256(rel.encode()).hexdigest(), 16)
        (held if h % 50 == 0 else train).append(rel)
    return train, held


def read(rels):
    return b"".join(open(os.path.join(ROOT, r), "rb").read() + b"\n" for r in rels)
