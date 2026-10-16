"""Export a trained checkpoint to the engine's weight format.

File layout (little-endian):
  b"LLMW" | u32 version (1) | u32 header_len | header JSON | zero padding to 64
  then each tensor as float32, 64-byte aligned, at the offset in the header.
Header: {"config": {...}, "tensors": [{"name", "shape", "offset"}]}
Weights are [out, in] (y = W x).
usage: python3 export.py CHECKPOINT.pkl OUT.llmw
"""
import json
import pickle
import struct
import sys

import numpy as np


def tensors(params):
    yield "embed", params["embed"]
    yield "norm", params["norm"]
    for i, L in enumerate(params["layers"]):
        for k in ("attn_norm", "wq", "wk", "wv", "wo", "mlp_norm", "w1", "w3", "w2"):
            yield f"layers.{i}.{k}", L[k]


def export(ckpt, out):
    cfg, params = ckpt["config"], ckpt["params"]
    entries, blobs, off = [], [], 0
    for name, t in tensors(params):
        a = np.ascontiguousarray(np.asarray(t, dtype="<f4"))
        entries.append({"name": name, "shape": list(a.shape), "offset": off})
        b = a.tobytes()
        pad = (-len(b)) % 64
        blobs.append(b + b"\0" * pad)
        off += len(b) + pad
    header = json.dumps({"config": cfg, "tensors": entries}).encode()
    head = b"LLMW" + struct.pack("<II", 1, len(header)) + header
    head += b"\0" * ((-len(head)) % 64)
    with open(out, "wb") as f:
        f.write(head)
        for b in blobs:
            f.write(b)


if __name__ == "__main__":
    export(pickle.load(open(sys.argv[1], "rb")), sys.argv[2])
