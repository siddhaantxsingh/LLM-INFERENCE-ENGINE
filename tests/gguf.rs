//! GGUF loading, the GPT-2 style tokenizer and the Llama / Qwen2 forward
//! pass, against expectations computed by independent references
//! (`validate/make_tiny_fixture.py`): Hugging Face `tokenizers`, llama.cpp's
//! tokenizer, jinja2 for the chat template and a float64 NumPy forward pass.
//! The tiny GGUF files were written by the `gguf` Python package.
//!
//! Tests against the real SmolLM2 download are opt-in (`#[ignore]`, and
//! `LLM_GGUF_MODEL` must point at the file); see docs/testing.md.

use llm_engine::engine::{Engine, EngineConfig};
use llm_engine::gguf::{self, DType, Gguf, Value};
use llm_engine::kv::KvCache;
use llm_engine::model::{Model, StepToken};
use llm_engine::sampler::SamplingParams;
use llm_engine::tokenizer::Tokenizer;
use serde::Deserialize;
use std::path::PathBuf;
use std::sync::Arc;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)
}

#[derive(Deserialize)]
struct Prompt {
    text: String,
    ids: Vec<u32>,
    argmax: Vec<u32>,
    last_logits: Vec<f64>,
    greedy16: Vec<u32>,
}

#[derive(Deserialize)]
struct Chat {
    messages: Vec<(String, String)>,
    rendered: String,
}

#[derive(Deserialize)]
struct Expected {
    corpus: Vec<String>,
    hf_ids: Vec<Vec<u32>>,
    llama_cpp_ids: Vec<Vec<u32>>,
    chat: Vec<Chat>,
    prompts: Vec<Prompt>,
}

fn load(arch: &str) -> (Arc<Model>, Arc<Tokenizer>, Expected) {
    let (m, t) = Model::load_any(&fixture(&format!("tiny_{arch}.gguf")), None).unwrap();
    let e: Expected = serde_json::from_str(&std::fs::read_to_string(fixture(&format!("tiny_{arch}.json"))).unwrap()).unwrap();
    (Arc::new(m), Arc::new(t.expect("GGUF carries a tokenizer")), e)
}

/// All-position logits for one prompt, with deliberately scattered KV blocks.
fn logits(m: &Model, ids: &[u32]) -> Vec<Vec<f32>> {
    let bs = 4;
    let nb = ids.len().div_ceil(bs);
    let mut kv = KvCache::new(m.cfg.layers, m.cfg.kv_heads, m.cfg.head_dim(), 2 * nb + 1, bs);
    let blocks: Vec<u32> = (0..nb as u32).map(|i| 2 * nb as u32 - 2 * i).collect();
    let toks: Vec<StepToken> = ids
        .iter()
        .enumerate()
        .map(|(pos, &token)| StepToken {
            token,
            pos,
            blocks: &blocks,
            want_logits: true,
        })
        .collect();
    m.forward(&toks, &mut kv)
}

fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |b, (i, &x)| if x > b.1 { (i, x) } else { b })
        .0 as u32
}

fn check_model(arch: &str) {
    let (m, t, e) = load(arch);
    assert_eq!(m.cfg.rope_neox, arch == "qwen2");
    assert_eq!(m.layers[0].bq.is_some(), arch == "qwen2");
    for p in &e.prompts {
        assert_eq!(t.encode(p.text.as_bytes()), p.ids, "{arch}: token ids of {:?}", p.text);
        let l = logits(&m, &p.ids);
        let am: Vec<u32> = l.iter().map(|r| argmax(r)).collect();
        assert_eq!(am, p.argmax, "{arch}: argmax at every position");
        let last = l.last().unwrap();
        let worst = last
            .iter()
            .zip(&p.last_logits)
            .map(|(a, b)| (*a as f64 - b).abs())
            .fold(0.0, f64::max);
        let scale = p.last_logits.iter().fold(0.0f64, |a, b| a.max(b.abs()));
        assert!(worst < 2e-4 * scale.max(1.0), "{arch}: max |logit diff| {worst} (scale {scale})");
    }
    // greedy generation through the engine (batched, paged, prefix-cached) equals the reference
    let mut eng = Engine::new(
        m.clone(),
        t.clone(),
        EngineConfig {
            block_size: 4,
            num_blocks: 64,
            poison_freed: true,
            ..Default::default()
        },
    );
    let reqs = e
        .prompts
        .iter()
        .map(|p| {
            (
                p.ids.clone(),
                SamplingParams {
                    max_tokens: 16,
                    ignore_eos: true,
                    ..Default::default()
                },
            )
        })
        .collect();
    for (out, p) in eng.generate(reqs).iter().zip(&e.prompts) {
        assert_eq!(out.0, p.greedy16, "{arch}: greedy tokens for {:?}", p.text);
    }
}

#[test]
fn tiny_llama_matches_numpy_reference() {
    check_model("llama");
}

#[test]
fn tiny_qwen2_with_biases_and_neox_rope_matches_numpy_reference() {
    check_model("qwen2");
}

#[test]
fn tokenizer_matches_hf_tokenizers_and_llama_cpp() {
    for arch in ["llama", "qwen2"] {
        let (_, t, e) = load(arch);
        let mut agree_cpp = 0;
        for (i, text) in e.corpus.iter().enumerate() {
            let ids = t.encode(text.as_bytes());
            assert_eq!(ids, e.hf_ids[i], "{arch}: {text:?} vs Hugging Face tokenizers");
            agree_cpp += (ids == e.llama_cpp_ids[i]) as usize;
            assert_eq!(t.decode(&ids), text.as_bytes(), "{arch}: round trip of {text:?}");
        }
        assert_eq!(agree_cpp, e.corpus.len(), "{arch}: llama.cpp ids");
        assert!(e.corpus.len() >= 20);
    }
}

#[test]
fn chat_template_matches_jinja() {
    for arch in ["llama", "qwen2"] {
        let (_, t, e) = load(arch);
        for c in &e.chat {
            assert_eq!(t.apply_chat_template(&c.messages).unwrap(), c.rendered, "{arch}");
        }
    }
}

#[test]
fn end_of_sequence_token_stops_generation() {
    let (m, _, e) = load("llama");
    let (_, t2) = Model::load_any(&fixture("tiny_llama.gguf"), None).unwrap();
    let mut t2 = t2.unwrap();
    let p = &e.prompts[1];
    // make the token greedy decoding picks second the end-of-sequence token
    t2.stop_ids = vec![p.greedy16[1]];
    let mut eng = Engine::new(m, Arc::new(t2), EngineConfig::default());
    let sp = |ignore_eos| SamplingParams {
        max_tokens: 16,
        ignore_eos,
        ..Default::default()
    };
    let out = eng.generate(vec![(p.ids.clone(), sp(false)), (p.ids.clone(), sp(true))]);
    assert_eq!(out[0].0, p.greedy16[..2]);
    assert_eq!(out[0].1.as_bytes(), eng.tok.decode(&p.greedy16[..1]), "the EOS token adds no text");
    assert_eq!(out[1].0, p.greedy16);
}

// ---- the reader on files written here -----------------------------------

struct W(Vec<u8>);
impl W {
    fn u32(&mut self, v: u32) {
        self.0.extend(v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.0.extend(v.to_le_bytes());
    }
    fn s(&mut self, v: &str) {
        self.u64(v.len() as u64);
        self.0.extend(v.as_bytes());
    }
}

/// A GGUF with every metadata type, 64-byte alignment and one tensor of each
/// supported type.
fn synthetic() -> (Vec<u8>, Vec<Vec<f32>>) {
    let mut w = W(b"GGUF".to_vec());
    w.u32(3);
    w.u64(4); // tensors
    w.u64(15); // metadata
    let kv: Vec<(&str, u32, Vec<u8>)> = vec![
        ("u8", 0, vec![200]),
        ("i8", 1, vec![0xfe]),
        ("u16", 2, 60000u16.to_le_bytes().to_vec()),
        ("i16", 3, (-3i16).to_le_bytes().to_vec()),
        ("u32", 4, 4_000_000_000u32.to_le_bytes().to_vec()),
        ("i32", 5, (-5i32).to_le_bytes().to_vec()),
        ("f32", 6, 1.5f32.to_le_bytes().to_vec()),
        ("bool", 7, vec![1]),
        ("u64", 10, u64::MAX.to_le_bytes().to_vec()),
        ("i64", 11, (-7i64).to_le_bytes().to_vec()),
        ("f64", 12, 2.25f64.to_le_bytes().to_vec()),
        ("general.alignment", 4, 64u32.to_le_bytes().to_vec()),
    ];
    for (k, t, v) in &kv {
        w.s(k);
        w.u32(*t);
        w.0.extend(v);
    }
    w.s("str");
    w.u32(8);
    w.s("héllo");
    w.s("arr");
    w.u32(9);
    w.u32(8);
    w.u64(2);
    w.s("a");
    w.s("");
    w.s("nested");
    w.u32(9);
    w.u32(9);
    w.u64(1);
    w.u32(5);
    w.u64(2);
    w.0.extend(1i32.to_le_bytes());
    w.0.extend((-1i32).to_le_bytes());
    // tensors: values chosen to be exact in every type
    let vals: Vec<f32> = (0..64).map(|i| (i as f32 - 20.0) * 0.25).collect();
    let mut data: Vec<u8> = vec![];
    let mut infos = vec![];
    let mut expect = vec![];
    for (name, ty) in [("f32", DType::F32), ("f16", DType::F16), ("bf16", DType::BF16), ("q8", DType::Q8_0)] {
        while !data.len().is_multiple_of(64) {
            data.push(0);
        }
        infos.push((name, ty, data.len() as u64));
        match ty {
            DType::F32 => vals.iter().for_each(|v| data.extend(v.to_le_bytes())),
            DType::F16 => vals.iter().for_each(|v| data.extend(gguf::f32_to_f16(*v).to_le_bytes())),
            DType::BF16 => vals.iter().for_each(|v| data.extend(((v.to_bits() >> 16) as u16).to_le_bytes())),
            _ => {
                for blk in vals.chunks(32) {
                    let amax = blk.iter().fold(0f32, |a, b| a.max(b.abs()));
                    let d = gguf::f16_to_f32(gguf::f32_to_f16(amax / 127.0));
                    data.extend(gguf::f32_to_f16(d).to_le_bytes());
                    blk.iter().for_each(|v| data.push((v / d).round() as i8 as u8));
                }
            }
        }
        expect.push(gguf::dequantize(ty, &data[infos.last().unwrap().2 as usize..]));
    }
    for (name, ty, off) in &infos {
        w.s(name);
        w.u32(2);
        w.u64(16);
        w.u64(4);
        w.u32(ty.id());
        w.u64(*off);
    }
    while !w.0.len().is_multiple_of(64) {
        w.0.push(0);
    }
    w.0.extend(data);
    (w.0, expect)
}

#[test]
fn reader_parses_every_metadata_and_tensor_type() {
    let (bytes, expect) = synthetic();
    let g = Gguf::parse(bytes).unwrap();
    assert_eq!(g.version, 3);
    assert_eq!(g.alignment, 64);
    assert_eq!(g.get("u8"), Some(&Value::U8(200)));
    assert_eq!(g.get("i8"), Some(&Value::I8(-2)));
    assert_eq!(g.get("u16"), Some(&Value::U16(60000)));
    assert_eq!(g.get("i16"), Some(&Value::I16(-3)));
    assert_eq!(g.get_u64("u32").unwrap(), 4_000_000_000);
    assert_eq!(g.get("i32"), Some(&Value::I32(-5)));
    assert_eq!(g.get_f64("f32").unwrap(), 1.5);
    assert_eq!(g.get("bool"), Some(&Value::Bool(true)));
    assert_eq!(g.get_u64("u64").unwrap(), u64::MAX);
    assert_eq!(g.get("i64"), Some(&Value::I64(-7)));
    assert_eq!(g.get_f64("f64").unwrap(), 2.25);
    assert_eq!(g.get_str("str").unwrap(), "héllo");
    assert_eq!(g.get("arr"), Some(&Value::Arr(vec![Value::Str("a".into()), Value::Str("".into())])));
    assert_eq!(
        g.get("nested"),
        Some(&Value::Arr(vec![Value::Arr(vec![Value::I32(1), Value::I32(-1)])]))
    );
    let vals: Vec<f32> = (0..64).map(|i| (i as f32 - 20.0) * 0.25).collect();
    for (i, name) in ["f32", "f16", "bf16", "q8"].iter().enumerate() {
        let t = g.tensor_f32(name, &[16, 4]).unwrap();
        assert_eq!(t, expect[i][..64]);
        let tol = if *name == "q8" { 0.05 } else { 0.0 };
        for (a, b) in t.iter().zip(&vals) {
            assert!((a - b).abs() <= tol, "{name}: {a} vs {b}");
        }
    }
    assert!(g.tensor_f32("f32", &[4, 16]).unwrap_err().contains("dims"));
    assert!(g.tensor_f32("missing", &[1]).is_err());
}

#[test]
fn reader_rejects_corrupt_files_without_panicking() {
    let (bytes, _) = synthetic();
    // every truncation is an error (the tensor data is checked against the file size)
    for n in 0..bytes.len() {
        assert!(Gguf::parse(bytes[..n].to_vec()).is_err(), "prefix of {n} bytes accepted");
    }
    let mut b = bytes.clone();
    b[0] = b'X';
    assert!(Gguf::parse(b).err().unwrap().contains("magic"));
    let mut b = bytes.clone();
    b[4] = 1;
    assert!(Gguf::parse(b).err().unwrap().contains("version"));
    // absurd counts and lengths fail fast instead of allocating
    let mut b = bytes.clone();
    b[16..24].copy_from_slice(&u64::MAX.to_le_bytes());
    assert!(Gguf::parse(b).is_err());
    let mut b = bytes.clone();
    b[24..32].copy_from_slice(&(u64::MAX / 2).to_le_bytes()); // first key length
    assert!(Gguf::parse(b).is_err());
    // flipping any single byte of the header region never panics
    for i in 0..bytes.len().min(700) {
        let mut b = bytes.clone();
        b[i] ^= 0xff;
        let _ = Gguf::parse(b);
    }
}

#[test]
fn unsupported_files_give_clear_errors() {
    let g = Gguf::parse(synthetic().0).unwrap();
    let e = Model::from_gguf(&g, None).err().unwrap();
    assert!(e.contains("general.architecture"), "{e}");
    assert!(Tokenizer::from_gguf(&g).is_err());
}

// ---- opt-in: the real SmolLM2-135M-Instruct --------------------------------

fn real_model() -> Option<(Arc<Model>, Arc<Tokenizer>)> {
    let p = std::env::var("LLM_GGUF_MODEL").ok()?;
    let (m, t) = Model::load_any(std::path::Path::new(&p), Some(2048)).unwrap();
    Some((Arc::new(m), Arc::new(t.unwrap())))
}

#[test]
#[ignore = "needs LLM_GGUF_MODEL=/home/user/models/smollm2-135m-instruct-f16.gguf (tools/fetch_model.sh)"]
fn smollm2_tokenizer_and_chat() {
    let (m, t) = real_model().expect("set LLM_GGUF_MODEL");
    assert_eq!(
        (m.cfg.layers, m.cfg.dim, m.cfg.heads, m.cfg.kv_heads, m.cfg.vocab),
        (30, 576, 9, 3, 49152)
    );
    // ids from llama.cpp (llama-cpp-python 0.3.36, vocab only)
    assert_eq!(t.encode(b"Hello, world!"), [19556, 28, 905, 17]);
    assert_eq!(
        t.encode(b"<|im_start|>user\n12345<|im_end|>"),
        [1, 4093, 198, 33, 34, 35, 36, 37, 2]
    );
    let prompt = t
        .apply_chat_template(&[("user".into(), "What is the capital of France?".into())])
        .unwrap();
    // jinja2 rendering of the GGUF's template (validate/make_tiny_fixture.py uses the same template)
    assert_eq!(
        prompt,
        "<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n\
         <|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n"
    );
    let mut e = Engine::new(
        m,
        t.clone(),
        EngineConfig {
            num_blocks: 64,
            ..Default::default()
        },
    );
    let (toks, text, _) = e
        .generate(vec![(
            t.encode_prompt(&prompt),
            SamplingParams {
                max_tokens: 32,
                ..Default::default()
            },
        )])
        .remove(0);
    assert!(text.contains("Paris"), "{text:?}");
    assert_eq!(toks.last(), Some(&2), "stops at <|im_end|>");
}
