"""Train a model on the stdlib corpus: python3 train.py CONFIG STEPS OUT.npz"""
import pickle
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

import data
import model
import tokenizer

cfg_name, steps, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
cfg = model.CONFIGS[cfg_name]
tok = tokenizer.Tokenizer.load("artifacts/tokenizer.json")
train_files, held_files = data.split()
ids = np.array(tok.encode(data.read(train_files)), dtype=np.int32)
held = np.array(tok.encode(data.read(held_files)), dtype=np.int32)
print(f"train {len(ids):,} tokens, held-out {len(held):,} tokens", flush=True)

B, T = 32, 256
LR, WARM, MIN_LR, WD = 3e-3, 200, 3e-4, 0.1
params = model.init(cfg, jax.random.PRNGKey(0))
print(f"{cfg_name}: {model.n_params(params):,} parameters", flush=True)
opt = jax.tree_util.tree_map(lambda x: (jnp.zeros_like(x), jnp.zeros_like(x)), params)


def lr_at(step):
    warm = LR * (step + 1) / WARM
    prog = jnp.clip((step - WARM) / max(1, steps - WARM), 0.0, 1.0)
    cos = MIN_LR + 0.5 * (LR - MIN_LR) * (1 + jnp.cos(jnp.pi * prog))
    return jnp.where(step < WARM, warm, cos)


@jax.jit
def step_fn(params, opt, batch, step):
    loss, g = jax.value_and_grad(model.loss_fn)(params, cfg, batch)
    gnorm = jnp.sqrt(sum(jnp.sum(x * x) for x in jax.tree_util.tree_leaves(g)))
    g = jax.tree_util.tree_map(lambda x: x * jnp.minimum(1.0, 1.0 / (gnorm + 1e-6)), g)
    lr = lr_at(step)
    t = step + 1

    def upd(p, g, o):
        m, v = o
        m = 0.9 * m + 0.1 * g
        v = 0.95 * v + 0.05 * g * g
        mh, vh = m / (1 - 0.9 ** t), v / (1 - 0.95 ** t)
        decay = WD * p if p.ndim == 2 else 0.0  # no decay on norms
        return p - lr * (mh / (jnp.sqrt(vh) + 1e-8) + decay), (m, v)

    flat = jax.tree_util.tree_map(upd, params, g, opt, is_leaf=lambda x: isinstance(x, tuple))
    new_p = jax.tree_util.tree_map(lambda x: x[0], flat, is_leaf=lambda x: isinstance(x, tuple) and len(x) == 2 and not isinstance(x[0], tuple))
    new_o = jax.tree_util.tree_map(lambda x: x[1], flat, is_leaf=lambda x: isinstance(x, tuple) and len(x) == 2 and not isinstance(x[0], tuple))
    return new_p, new_o, loss, gnorm


eval_fn = jax.jit(lambda p, b: model.loss_fn(p, cfg, b))


def evaluate(p):
    n = (len(held) - 1) // T
    xs = np.stack([held[i * T: i * T + T + 1] for i in range(n)])
    return float(np.mean([eval_fn(p, xs[i: i + 16]) for i in range(0, n - 15, 16)]))


import os
ckpt_path = out + ".resume"
rng = np.random.default_rng(0)
start, log = 0, []
if os.path.exists(ckpt_path):
    # resume after an interruption: same parameters, optimiser state, data RNG and log
    r = pickle.load(open(ckpt_path, "rb"))
    params = jax.tree_util.tree_map(jnp.asarray, r["params"])
    opt = jax.tree_util.tree_map(jnp.asarray, r["opt"])
    rng.bit_generator.state = r["rng"]
    start, log = r["step"], r["log"]
    print(f"resumed at step {start}", flush=True)
t0 = time.time() - (log[-1]["elapsed"] if log else 0.0)
for s in range(start, steps):
    if s > start and s % 250 == 0:
        tmp = ckpt_path + ".tmp"
        pickle.dump({"params": jax.tree_util.tree_map(np.asarray, params), "opt": jax.tree_util.tree_map(np.asarray, opt),
                     "rng": rng.bit_generator.state, "step": s, "log": log}, open(tmp, "wb"))
        os.replace(tmp, ckpt_path)
    starts = rng.integers(0, len(ids) - T - 1, B)
    batch = jnp.asarray(np.stack([ids[i: i + T + 1] for i in starts]))
    params, opt, loss, gnorm = step_fn(params, opt, batch, s)
    if s % 100 == 0 or s == steps - 1:
        rate = (s + 1 - start) * B * T / max(1e-9, time.time() - t0 - (log[-1]["elapsed"] if start and log else 0.0)) if s > start else 0.0
        msg = dict(step=s, loss=float(loss), gnorm=float(gnorm), tok_per_s=rate, elapsed=time.time() - t0)
        if s % 1000 == 0 or s == steps - 1:
            msg["held_out_loss"] = evaluate(params)
        log.append(msg)
        print(msg, flush=True)
params = jax.tree_util.tree_map(np.asarray, params)
pickle.dump({"config": cfg, "params": params, "log": log}, open(out, "wb"))
if os.path.exists(ckpt_path):
    os.remove(ckpt_path)
print("saved", out)
