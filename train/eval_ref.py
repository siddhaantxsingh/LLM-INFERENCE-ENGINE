"""JAX loss of a checkpoint on the same windows `llm ppl` uses (window W+1
tokens every W tokens), for comparing the engine with the trainer.
usage: python3 eval_ref.py CKPT.pkl TEXT_FILE [W]"""
import json
import pickle
import sys

import jax
import jax.numpy as jnp
import numpy as np

import model
import tokenizer

ck = pickle.load(open(sys.argv[1], "rb"))
W = int(sys.argv[3]) if len(sys.argv) > 3 else 256
tok = tokenizer.Tokenizer.load("artifacts/tokenizer.json")
ids = np.array(tok.encode(open(sys.argv[2], "rb").read()), np.int32)
wins = [ids[s: s + W + 1] for s in range(0, len(ids) - W, W)]
f = jax.jit(lambda p, b: model.loss_fn(p, ck["config"], b))
params = jax.tree_util.tree_map(jnp.asarray, ck["params"])
losses = [float(f(params, jnp.asarray(w[None]))) for w in wins]
print(json.dumps({"tokens": len(wins) * W, "loss": float(np.mean(losses)), "perplexity": float(np.exp(np.mean(losses)))}))
