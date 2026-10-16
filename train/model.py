"""Llama-architecture decoder in JAX: RMSNorm, rotary embeddings (RoPE),
grouped-query attention, SwiGLU MLP, tied input/output embeddings.
The Rust engine implements exactly this forward pass."""
import jax
import jax.numpy as jnp

CONFIGS = {
    # the served model
    "small": dict(vocab=1024, dim=256, layers=4, heads=8, kv_heads=2, hidden=688, max_seq=512, rope_theta=10000.0, eps=1e-5),
    # a 1-layer draft model for speculative decoding
    "draft": dict(vocab=1024, dim=128, layers=1, heads=4, kv_heads=2, hidden=344, max_seq=512, rope_theta=10000.0, eps=1e-5),
}


def init(cfg, key):
    d, h, kvh, f = cfg["dim"], cfg["heads"], cfg["kv_heads"], cfg["hidden"]
    hd = d // h
    keys = iter(jax.random.split(key, 4 + 8 * cfg["layers"]))
    nrm = lambda k, shape, scale: (jax.random.normal(k, shape) * scale).astype(jnp.float32)
    p = {"embed": nrm(next(keys), (cfg["vocab"], d), 0.02), "norm": jnp.ones((d,)), "layers": []}
    proj_scale = 0.02 / (2 * cfg["layers"]) ** 0.5
    for _ in range(cfg["layers"]):
        p["layers"].append({
            "attn_norm": jnp.ones((d,)), "mlp_norm": jnp.ones((d,)),
            # weights are stored [out, in] like the engine (y = W x)
            "wq": nrm(next(keys), (h * hd, d), 0.02), "wk": nrm(next(keys), (kvh * hd, d), 0.02),
            "wv": nrm(next(keys), (kvh * hd, d), 0.02), "wo": nrm(next(keys), (d, h * hd), proj_scale),
            "w1": nrm(next(keys), (f, d), 0.02), "w3": nrm(next(keys), (f, d), 0.02), "w2": nrm(next(keys), (d, f), proj_scale),
        })
    return p


def rmsnorm(x, w, eps):
    return x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps) * w


def rope(x, pos, theta):
    # x: [..., T, H, hd]; rotate pairs (2i, 2i+1)
    hd = x.shape[-1]
    inv = 1.0 / theta ** (jnp.arange(0, hd, 2, dtype=jnp.float32) / hd)
    ang = pos[:, None].astype(jnp.float32) * inv[None, :]          # [T, hd/2]
    cos, sin = jnp.cos(ang)[:, None, :], jnp.sin(ang)[:, None, :]  # [T, 1, hd/2]
    x0, x1 = x[..., 0::2], x[..., 1::2]
    r0, r1 = x0 * cos - x1 * sin, x0 * sin + x1 * cos
    return jnp.stack([r0, r1], axis=-1).reshape(x.shape)


def forward(p, cfg, tokens):
    """tokens [B, T] -> logits [B, T, vocab] (causal)."""
    B, T = tokens.shape
    d, h, kvh = cfg["dim"], cfg["heads"], cfg["kv_heads"]
    hd = d // h
    x = p["embed"][tokens]
    pos = jnp.arange(T)
    mask = jnp.tril(jnp.ones((T, T), bool))
    for L in p["layers"]:
        a = rmsnorm(x, L["attn_norm"], cfg["eps"])
        q = (a @ L["wq"].T).reshape(B, T, h, hd)
        k = (a @ L["wk"].T).reshape(B, T, kvh, hd)
        v = (a @ L["wv"].T).reshape(B, T, kvh, hd)
        q, k = rope(q, pos, cfg["rope_theta"]), rope(k, pos, cfg["rope_theta"])
        k = jnp.repeat(k, h // kvh, axis=2)
        v = jnp.repeat(v, h // kvh, axis=2)
        s = jnp.einsum("bqhd,bkhd->bhqk", q, k) / jnp.sqrt(jnp.float32(hd))
        s = jnp.where(mask[None, None], s, -jnp.inf)
        o = jnp.einsum("bhqk,bkhd->bqhd", jax.nn.softmax(s, axis=-1), v).reshape(B, T, h * hd)
        x = x + o @ L["wo"].T
        m = rmsnorm(x, L["mlp_norm"], cfg["eps"])
        x = x + (jax.nn.silu(m @ L["w1"].T) * (m @ L["w3"].T)) @ L["w2"].T
    x = rmsnorm(x, p["norm"], cfg["eps"])
    return x @ p["embed"].T


def loss_fn(p, cfg, tokens):
    logits = forward(p, cfg, tokens[:, :-1])
    lp = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.take_along_axis(lp, tokens[:, 1:, None], axis=-1))


def n_params(p):
    return sum(x.size for x in jax.tree_util.tree_leaves(p))
