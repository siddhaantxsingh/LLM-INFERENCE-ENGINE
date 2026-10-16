//! Weights and the batched forward pass (Llama architecture: RMSNorm, RoPE,
//! grouped-query attention, SwiGLU, tied or separate output embeddings,
//! optional Q/K/V biases as in Qwen2).
//!
//! Weights load from this project's `LLMW` files or from GGUF files
//! (`general.architecture` = `llama` or `qwen2`).
//!
//! The forward pass takes a flat list of tokens from any number of
//! sequences, each with its position and its sequence's KV block table. All
//! projections run as one matmul over every token in the step (each weight
//! row is read once per step); attention is per token over its own
//! sequence's cached keys and values.

use crate::kv::KvCache;
use crate::ops::{self, Mat};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Config {
    pub vocab: usize,
    pub dim: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub hidden: usize,
    pub max_seq: usize,
    pub rope_theta: f32,
    pub eps: f32,
    /// RoPE rotates (x[i], x[i + head_dim/2]) pairs ("NeoX" style, Qwen2)
    /// instead of adjacent (x[2i], x[2i+1]) pairs (Llama in GGUF, whose Q/K
    /// weights the converter has permuted for adjacent pairs)
    #[serde(default)]
    pub rope_neox: bool,
}

impl Config {
    pub fn head_dim(&self) -> usize {
        self.dim / self.heads
    }
    pub fn kv_dim(&self) -> usize {
        self.kv_heads * self.head_dim()
    }
}

pub struct Layer {
    pub attn_norm: Vec<f32>,
    pub mlp_norm: Vec<f32>,
    pub wq: Mat,
    pub wk: Mat,
    pub wv: Mat,
    pub wo: Mat,
    pub w1: Mat,
    pub w3: Mat,
    pub w2: Mat,
    /// Q/K/V biases (Qwen2); None for Llama
    pub bq: Option<Vec<f32>>,
    pub bk: Option<Vec<f32>>,
    pub bv: Option<Vec<f32>>,
}

pub struct Model {
    pub cfg: Config,
    pub embed: Vec<f32>,
    /// the output projection: the embedding matrix if tied (may be int8)
    pub lm_head: Mat,
    pub norm: Vec<f32>,
    pub layers: Vec<Layer>,
    /// RoPE cos/sin per position, [max_seq, head_dim / 2]
    rope_cos: Vec<f32>,
    rope_sin: Vec<f32>,
}

/// One token of a forward step.
#[derive(Clone, Debug)]
pub struct StepToken<'a> {
    pub token: u32,
    pub pos: usize,
    /// the sequence's KV blocks (block i holds positions i*B .. (i+1)*B)
    pub blocks: &'a [u32],
    pub want_logits: bool,
}

#[derive(Deserialize)]
struct TensorInfo {
    name: String,
    shape: Vec<usize>,
    offset: usize,
}

#[derive(Deserialize)]
struct Header {
    config: Config,
    tensors: Vec<TensorInfo>,
}

fn rope_tables(c: &Config) -> (Vec<f32>, Vec<f32>) {
    let hd = c.head_dim();
    let half = hd / 2;
    let mut rope_cos = vec![0f32; c.max_seq * half];
    let mut rope_sin = vec![0f32; c.max_seq * half];
    for pos in 0..c.max_seq {
        for i in 0..half {
            // same formula as the trainer: theta^(-2i/hd), computed in f64
            let inv = 1.0 / (c.rope_theta as f64).powf((2 * i) as f64 / hd as f64);
            let a = pos as f64 * inv;
            rope_cos[pos * half + i] = a.cos() as f32;
            rope_sin[pos * half + i] = a.sin() as f32;
        }
    }
    (rope_cos, rope_sin)
}

/// Facts about a GGUF model, for logs and the API.
#[derive(Clone, Debug, Serialize)]
pub struct GgufInfo {
    pub name: String,
    pub architecture: String,
    /// tensor types in the file, e.g. {"F16": 211, "F32": 61}
    pub tensor_types: std::collections::BTreeMap<String, usize>,
    pub file_bytes: u64,
    /// the model's own context length (the engine may use less)
    pub context_length: usize,
}

impl Model {
    /// Load an `LLMW` or GGUF file (detected by its magic bytes). A GGUF
    /// file also yields its tokenizer.
    pub fn load_any(path: &std::path::Path, max_seq: Option<usize>) -> Result<(Model, Option<crate::tokenizer::Tokenizer>), String> {
        let mut magic = [0u8; 4];
        use std::io::Read;
        std::fs::File::open(path)
            .and_then(|mut f| f.read_exact(&mut magic))
            .map_err(|e| format!("{}: {e}", path.display()))?;
        if &magic == b"GGUF" {
            let g = crate::gguf::Gguf::open(path)?;
            let (m, _) = Model::from_gguf(&g, max_seq)?;
            let t = crate::tokenizer::Tokenizer::from_gguf(&g)?;
            Ok((m, Some(t)))
        } else {
            Ok((Model::load(path)?, None))
        }
    }

    /// Build the model from a parsed GGUF file. `max_seq` caps the context
    /// (and the RoPE tables) below the model's own `context_length`.
    pub fn from_gguf(g: &crate::gguf::Gguf, max_seq: Option<usize>) -> Result<(Model, GgufInfo), String> {
        let arch = g.get_str("general.architecture")?.to_string();
        let neox = match arch.as_str() {
            "llama" => false,
            "qwen2" => true,
            a => return Err(format!("architecture {a:?} is not supported (llama, qwen2)")),
        };
        let k = |s: &str| format!("{arch}.{s}");
        let u = |s: &str| g.get_u64(&k(s)).map(|v| v as usize);
        let d = u("embedding_length")?;
        let heads = u("attention.head_count")?;
        let kv_heads = u("attention.head_count_kv").unwrap_or(heads);
        let ctx = u("context_length")?;
        let emb = g.tensor("token_embd.weight").ok_or("missing token_embd.weight")?;
        let vocab = emb.dims.get(1).copied().unwrap_or(0) as usize;
        let cfg = Config {
            vocab,
            dim: d,
            layers: u("block_count")?,
            heads,
            kv_heads,
            hidden: u("feed_forward_length")?,
            max_seq: max_seq.unwrap_or(ctx).min(ctx),
            rope_theta: g.get_f64(&k("rope.freq_base")).unwrap_or(10000.0) as f32,
            eps: g.get_f64(&k("attention.layer_norm_rms_epsilon"))? as f32,
            rope_neox: neox,
        };
        if heads == 0 || kv_heads == 0 || !d.is_multiple_of(heads) || !heads.is_multiple_of(kv_heads) || !cfg.head_dim().is_multiple_of(2) {
            return Err(format!("inconsistent config {cfg:?}"));
        }
        let hd = cfg.head_dim();
        for key in ["attention.key_length", "attention.value_length", "rope.dimension_count"] {
            if let Ok(v) = u(key)
                && v != hd
            {
                return Err(format!("{arch}.{key} = {v}, but only head_dim = dim / heads = {hd} is supported"));
            }
        }
        if let Some(t) = g.get(&k("rope.scaling.type")).and_then(|v| v.as_str())
            && t != "none"
        {
            return Err(format!("RoPE scaling {t:?} is not supported"));
        }
        // ggml dims are [cols, rows]
        let mat = |name: &str, rows: usize, cols: usize| -> Result<Mat, String> {
            Ok(Mat::F32 {
                rows,
                cols,
                w: g.tensor_f32(name, &[cols, rows])?,
            })
        };
        let vecf = |name: &str, n: usize| g.tensor_f32(name, &[n]);
        let opt_vec = |name: &str, n: usize| -> Result<Option<Vec<f32>>, String> {
            if g.tensor(name).is_some() {
                Ok(Some(vecf(name, n)?))
            } else {
                Ok(None)
            }
        };
        let (kvd, c) = (cfg.kv_dim(), &cfg);
        let mut layers = vec![];
        for i in 0..c.layers {
            let p = |s: &str| format!("blk.{i}.{s}");
            layers.push(Layer {
                attn_norm: vecf(&p("attn_norm.weight"), d)?,
                mlp_norm: vecf(&p("ffn_norm.weight"), d)?,
                wq: mat(&p("attn_q.weight"), d, d)?,
                wk: mat(&p("attn_k.weight"), kvd, d)?,
                wv: mat(&p("attn_v.weight"), kvd, d)?,
                wo: mat(&p("attn_output.weight"), d, d)?,
                w1: mat(&p("ffn_gate.weight"), c.hidden, d)?,
                w3: mat(&p("ffn_up.weight"), c.hidden, d)?,
                w2: mat(&p("ffn_down.weight"), d, c.hidden)?,
                bq: opt_vec(&p("attn_q.bias"), d)?,
                bk: opt_vec(&p("attn_k.bias"), kvd)?,
                bv: opt_vec(&p("attn_v.bias"), kvd)?,
            });
        }
        let embed = g.tensor_f32("token_embd.weight", &[d, vocab])?;
        let lm_head = if g.tensor("output.weight").is_some() {
            mat("output.weight", vocab, d)?
        } else {
            Mat::F32 {
                rows: vocab,
                cols: d,
                w: embed.clone(),
            }
        };
        let mut tensor_types = std::collections::BTreeMap::new();
        for t in &g.tensors {
            *tensor_types.entry(format!("{:?}", t.dtype).to_uppercase()).or_insert(0) += 1;
        }
        let info = GgufInfo {
            name: g.get("general.name").and_then(|v| v.as_str()).unwrap_or("unnamed").to_string(),
            architecture: arch.clone(),
            tensor_types,
            file_bytes: 0,
            context_length: ctx,
        };
        let (rope_cos, rope_sin) = rope_tables(&cfg);
        Ok((
            Model {
                lm_head,
                embed,
                norm: vecf("output_norm.weight", d)?,
                layers,
                rope_cos,
                rope_sin,
                cfg,
            },
            info,
        ))
    }

    pub fn load(path: &std::path::Path) -> Result<Model, String> {
        let raw = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        if raw.len() < 12 || &raw[..4] != b"LLMW" {
            return Err("not an LLMW file".into());
        }
        let ver = u32::from_le_bytes(raw[4..8].try_into().unwrap());
        let hl = u32::from_le_bytes(raw[8..12].try_into().unwrap()) as usize;
        if ver != 1 {
            return Err(format!("unsupported version {ver}"));
        }
        let hdr: Header = serde_json::from_slice(raw.get(12..12 + hl).ok_or("truncated header")?).map_err(|e| e.to_string())?;
        let base = (12 + hl).div_ceil(64) * 64;
        let get = |name: &str, shape: &[usize]| -> Result<Vec<f32>, String> {
            let t = hdr
                .tensors
                .iter()
                .find(|t| t.name == name)
                .ok_or(format!("missing tensor {name}"))?;
            if t.shape != shape {
                return Err(format!("{name}: shape {:?}, expected {shape:?}", t.shape));
            }
            let n: usize = shape.iter().product();
            let start = base + t.offset;
            let bytes = raw.get(start..start + 4 * n).ok_or(format!("{name}: truncated"))?;
            Ok(bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect())
        };
        let c = &hdr.config;
        if !c.dim.is_multiple_of(c.heads) || !c.heads.is_multiple_of(c.kv_heads) || !c.head_dim().is_multiple_of(2) {
            return Err("inconsistent config".into());
        }
        let (d, hd) = (c.dim, c.head_dim());
        let mat = |name: String, r: usize, k: usize| -> Result<Mat, String> {
            Ok(Mat::F32 {
                rows: r,
                cols: k,
                w: get(&name, &[r, k])?,
            })
        };
        let mut layers = vec![];
        for i in 0..c.layers {
            let p = |k: &str| format!("layers.{i}.{k}");
            layers.push(Layer {
                attn_norm: get(&p("attn_norm"), &[d])?,
                mlp_norm: get(&p("mlp_norm"), &[d])?,
                wq: mat(p("wq"), c.heads * hd, d)?,
                wk: mat(p("wk"), c.kv_heads * hd, d)?,
                wv: mat(p("wv"), c.kv_heads * hd, d)?,
                wo: mat(p("wo"), d, c.heads * hd)?,
                w1: mat(p("w1"), c.hidden, d)?,
                w3: mat(p("w3"), c.hidden, d)?,
                w2: mat(p("w2"), d, c.hidden)?,
                bq: None,
                bk: None,
                bv: None,
            });
        }
        let embed = get("embed", &[c.vocab, d])?;
        let (rope_cos, rope_sin) = rope_tables(c);
        Ok(Model {
            lm_head: Mat::F32 {
                rows: c.vocab,
                cols: d,
                w: embed.clone(),
            },
            embed,
            norm: get("norm", &[d])?,
            layers,
            rope_cos,
            rope_sin,
            cfg: hdr.config,
        })
    }

    /// Weight-only int8 quantisation of every matrix (norms and the
    /// embedding lookup stay f32; the output projection is quantised).
    pub fn quantize(&mut self) {
        for l in &mut self.layers {
            for m in [&mut l.wq, &mut l.wk, &mut l.wv, &mut l.wo, &mut l.w1, &mut l.w3, &mut l.w2] {
                *m = m.quantize();
            }
        }
        self.lm_head = self.lm_head.quantize();
    }

    pub fn weight_bytes(&self) -> usize {
        let mut n = self.lm_head.bytes() + self.embed.len() * 4;
        for l in &self.layers {
            n += [&l.wq, &l.wk, &l.wv, &l.wo, &l.w1, &l.w3, &l.w2]
                .iter()
                .map(|m| m.bytes())
                .sum::<usize>();
        }
        n
    }

    fn rope(&self, v: &mut [f32], pos: usize) {
        let half = self.cfg.head_dim() / 2;
        let (c, s) = (
            &self.rope_cos[pos * half..(pos + 1) * half],
            &self.rope_sin[pos * half..(pos + 1) * half],
        );
        for head in v.chunks_exact_mut(2 * half) {
            for i in 0..half {
                let (a, b) = if self.cfg.rope_neox { (i, i + half) } else { (2 * i, 2 * i + 1) };
                let (x0, x1) = (head[a], head[b]);
                head[a] = x0 * c[i] - x1 * s[i];
                head[b] = x0 * s[i] + x1 * c[i];
            }
        }
    }

    /// Run one step. Writes every token's K/V into the cache, then attends
    /// over positions 0..=pos of its sequence. Returns logits for the tokens
    /// with `want_logits`, in order.
    pub fn forward(&self, toks: &[StepToken], kv: &mut KvCache) -> Vec<Vec<f32>> {
        let c = &self.cfg;
        let (n, d, hd, kvd) = (toks.len(), c.dim, c.head_dim(), c.kv_dim());
        let group = c.heads / c.kv_heads;
        let bs = kv.block_size();
        for t in toks {
            assert!(t.pos < c.max_seq, "position {} beyond max_seq", t.pos);
            assert!(t.pos / bs < t.blocks.len(), "no KV block for position {}", t.pos);
        }
        let mut x: Vec<f32> = toks
            .iter()
            .flat_map(|t| self.embed[t.token as usize * d..(t.token as usize + 1) * d].iter().copied())
            .collect();
        let mut a = vec![0f32; n * d];
        let mut q = vec![0f32; n * d];
        let mut k = vec![0f32; n * kvd];
        let mut v = vec![0f32; n * kvd];
        let mut att = vec![0f32; n * d];
        let mut o = vec![0f32; n * d];
        let mut g = vec![0f32; n * c.hidden];
        let mut u = vec![0f32; n * c.hidden];
        let scale = 1.0 / (hd as f32).sqrt();
        for (li, l) in self.layers.iter().enumerate() {
            for i in 0..n {
                ops::rmsnorm(&x[i * d..(i + 1) * d], &l.attn_norm, c.eps, &mut a[i * d..(i + 1) * d]);
            }
            l.wq.matmul(&a, n, &mut q);
            l.wk.matmul(&a, n, &mut k);
            l.wv.matmul(&a, n, &mut v);
            for (bias, out, w) in [(&l.bq, &mut q, d), (&l.bk, &mut k, kvd), (&l.bv, &mut v, kvd)] {
                if let Some(b) = bias {
                    for row in out.chunks_exact_mut(w) {
                        for (o, bb) in row.iter_mut().zip(b) {
                            *o += bb;
                        }
                    }
                }
            }
            for (i, t) in toks.iter().enumerate() {
                self.rope(&mut q[i * d..(i + 1) * d], t.pos);
                self.rope(&mut k[i * kvd..(i + 1) * kvd], t.pos);
                let slot = t.blocks[t.pos / bs] as usize * bs + t.pos % bs;
                kv.write(li, slot, &k[i * kvd..(i + 1) * kvd], &v[i * kvd..(i + 1) * kvd]);
            }
            let kvr = &*kv;
            att.par_chunks_mut(hd).enumerate().for_each(|(idx, out)| {
                let (i, h) = (idx / c.heads, idx % c.heads);
                let t = &toks[i];
                let qh = &q[i * d + h * hd..i * d + (h + 1) * hd];
                let kh = h / group;
                let mut scores = Vec::with_capacity(t.pos + 1);
                let mut m = f32::NEG_INFINITY;
                for p in 0..=t.pos {
                    let slot = t.blocks[p / bs] as usize * bs + p % bs;
                    let s = ops::dot(qh, kvr.key(li, slot, kh)) * scale;
                    m = m.max(s);
                    scores.push(s);
                }
                let mut sum = 0f32;
                for s in scores.iter_mut() {
                    *s = (*s - m).exp();
                    sum += *s;
                }
                out.fill(0.0);
                for (p, &w) in scores.iter().enumerate() {
                    let slot = t.blocks[p / bs] as usize * bs + p % bs;
                    let vh = kvr.value(li, slot, kh);
                    let w = w / sum;
                    for (o, &vv) in out.iter_mut().zip(vh) {
                        *o = w.mul_add(vv, *o);
                    }
                }
            });
            l.wo.matmul(&att, n, &mut o);
            for (xi, oi) in x.iter_mut().zip(&o) {
                *xi += oi;
            }
            for i in 0..n {
                ops::rmsnorm(&x[i * d..(i + 1) * d], &l.mlp_norm, c.eps, &mut a[i * d..(i + 1) * d]);
            }
            l.w1.matmul(&a, n, &mut g);
            l.w3.matmul(&a, n, &mut u);
            for (gi, ui) in g.iter_mut().zip(&u) {
                *gi = ops::silu(*gi) * ui;
            }
            l.w2.matmul(&g, n, &mut o);
            for (xi, oi) in x.iter_mut().zip(&o) {
                *xi += oi;
            }
        }
        let want: Vec<usize> = (0..n).filter(|&i| toks[i].want_logits).collect();
        if want.is_empty() {
            return vec![];
        }
        let mut hsel = vec![0f32; want.len() * d];
        for (j, &i) in want.iter().enumerate() {
            ops::rmsnorm(&x[i * d..(i + 1) * d], &self.norm, c.eps, &mut hsel[j * d..(j + 1) * d]);
        }
        let mut logits = vec![0f32; want.len() * c.vocab];
        self.lm_head.matmul(&hsel, want.len(), &mut logits);
        logits.chunks_exact(c.vocab).map(|r| r.to_vec()).collect()
    }
}
