//! Byte-level BPE tokenizers.
//!
//! Two flavours share one merge loop:
//! * **Legacy**: the tokenizer of the model trained in `train/`, identical
//!   to `train/tokenizer.py` (checked in `tests/engine.rs`). Token i >= 256
//!   is merge i - 256.
//! * **GPT-2 style**, loaded from GGUF metadata (`tokenizer.ggml.model =
//!   "gpt2"`): vocabulary strings in the GPT-2 byte-to-unicode alphabet,
//!   ranked merges ("a b"), a regex pre-tokenizer chosen by
//!   `tokenizer.ggml.pre`, and special tokens (control / user-defined) that
//!   are matched as whole strings before BPE, as Hugging Face `tokenizers`
//!   and llama.cpp do.

use crate::gguf::Gguf;
use std::collections::HashMap;

/// A pre-tokenizer: text -> chunks that merges never cross.
enum Pre {
    Legacy,
    /// regexes applied in sequence; every piece (match or gap) is kept
    Regex(Vec<fancy_regex::Regex>),
}

pub struct Tokenizer {
    /// (left, right) -> (rank, merged token)
    merges: HashMap<(u32, u32), (u32, u32)>,
    bytes: Vec<Vec<u8>>,
    /// the single-byte token for each byte value (None: not in the vocabulary)
    byte_tok: Vec<Option<u32>>,
    /// replaces a byte with no token (Hugging Face BPE's `unk_token`)
    unk: Option<u32>,
    pre: Pre,
    /// special tokens matched before BPE, longest first
    special: Vec<(Vec<u8>, u32)>,
    pub bos: Option<u32>,
    pub eos: Option<u32>,
    /// tokens that end generation (eos, end-of-turn)
    pub stop_ids: Vec<u32>,
    pub add_bos: bool,
    pub chat_template: Option<String>,
}

fn is_word(b: u8) -> bool {
    b >= 0x80 || b == b'_' || b.is_ascii_alphanumeric()
}

/// Legacy pre-tokenizer: merges never cross these chunk boundaries.
pub fn chunks(data: &[u8]) -> Vec<&[u8]> {
    let (mut out, mut i, n) = (vec![], 0, data.len());
    while i < n {
        let c = data[i];
        let mut j;
        if c == b' ' && i + 1 < n && is_word(data[i + 1]) {
            j = i + 1;
            while j < n && is_word(data[j]) {
                j += 1;
            }
        } else if is_word(c) {
            j = i;
            while j < n && is_word(data[j]) {
                j += 1;
            }
        } else if c == b' ' {
            j = i;
            while j < n && data[j] == b' ' {
                j += 1;
            }
            if j < n && is_word(data[j]) && j - i > 1 {
                j -= 1;
            }
        } else if c == b'\n' {
            j = i;
            while j < n && data[j] == b'\n' {
                j += 1;
            }
        } else {
            j = i + 1;
        }
        out.push(&data[i..j]);
        i = j;
    }
    out
}

/// GPT-2's reversible map from bytes to printable unicode characters.
pub fn byte_to_unicode() -> [char; 256] {
    let mut m = ['\0'; 256];
    let mut n = 0u32;
    for b in 0..256u32 {
        let printable = (0x21..=0x7e).contains(&b) || (0xa1..=0xac).contains(&b) || (0xae..=0xff).contains(&b);
        m[b as usize] = if printable {
            char::from_u32(b).unwrap()
        } else {
            n += 1;
            char::from_u32(255 + n).unwrap()
        };
    }
    m
}

/// Pre-tokenizer regexes, as in llama.cpp's `llama-vocab.cpp` and the
/// models' Hugging Face `tokenizer.json`.
const GPT2_RE: &str = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";
const QWEN2_RE: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

fn pre_regexes(pre: &str) -> Result<Vec<&'static str>, String> {
    Ok(match pre {
        "default" | "gpt2" | "gpt-2" => vec![GPT2_RE],
        // SmolLM / StarCoder: every digit is its own piece, then GPT-2
        "smollm" | "starcoder" | "refact" => vec![r"\p{N}", GPT2_RE],
        "qwen2" => vec![QWEN2_RE],
        p => return Err(format!("unsupported pre-tokenizer {p:?}")),
    })
}

fn split_all(re: &fancy_regex::Regex, s: &str) -> Vec<(usize, usize)> {
    let mut out = vec![];
    let mut last = 0;
    for m in re.find_iter(s) {
        let m = m.expect("regex backtrack limit");
        if m.start() > last {
            out.push((last, m.start()));
        }
        if m.end() > m.start() {
            out.push((m.start(), m.end()));
        }
        last = m.end();
    }
    if last < s.len() {
        out.push((last, s.len()));
    }
    out
}

impl Tokenizer {
    /// The tokenizer of the model trained in `train/` (`models/tokenizer.json`).
    pub fn from_json(s: &str) -> Result<Tokenizer, String> {
        let v: serde_json::Value = serde_json::from_str(s).map_err(|e| e.to_string())?;
        let merges: Vec<(u32, u32)> = serde_json::from_value(v["merges"].clone()).map_err(|e| e.to_string())?;
        let mut bytes: Vec<Vec<u8>> = (0..=255u8).map(|b| vec![b]).collect();
        let mut rank = HashMap::new();
        for (i, &(a, b)) in merges.iter().enumerate() {
            let mut t = bytes[a as usize].clone();
            t.extend_from_slice(&bytes[b as usize]);
            bytes.push(t);
            rank.insert((a, b), (i as u32, 256 + i as u32));
        }
        Ok(Tokenizer {
            merges: rank,
            bytes,
            byte_tok: (0..256).map(Some).collect(),
            unk: None,
            pre: Pre::Legacy,
            special: vec![],
            bos: None,
            eos: None,
            stop_ids: vec![],
            add_bos: false,
            chat_template: None,
        })
    }

    /// A GPT-2 style byte-level BPE from GGUF metadata.
    pub fn from_gguf(g: &Gguf) -> Result<Tokenizer, String> {
        let model = g.get_str("tokenizer.ggml.model")?;
        if model != "gpt2" {
            return Err(format!(
                "tokenizer model {model:?} is not supported (only byte-level BPE, \"gpt2\")"
            ));
        }
        let pre = g.get("tokenizer.ggml.pre").and_then(|v| v.as_str()).unwrap_or("default");
        let regexes = pre_regexes(pre)?
            .into_iter()
            .map(|r| fancy_regex::Regex::new(r).map_err(|e| e.to_string()))
            .collect::<Result<Vec<_>, _>>()?;
        let toks = g
            .get("tokenizer.ggml.tokens")
            .and_then(|v| v.as_arr())
            .ok_or("missing tokenizer.ggml.tokens")?;
        let types: Vec<u64> = match g.get("tokenizer.ggml.token_type").and_then(|v| v.as_arr()) {
            Some(a) => a.iter().map(|v| v.as_u64().unwrap_or(1)).collect(),
            None => vec![1; toks.len()],
        };
        if types.len() != toks.len() {
            return Err("token_type and tokens differ in length".into());
        }
        let b2u = byte_to_unicode();
        let mut u2b: HashMap<char, u8> = HashMap::new();
        for (b, &c) in b2u.iter().enumerate() {
            u2b.insert(c, b as u8);
        }
        let mut bytes = Vec::with_capacity(toks.len());
        let mut id_of: HashMap<&str, u32> = HashMap::new();
        let mut special = vec![];
        for (i, t) in toks.iter().enumerate() {
            let s = t.as_str().ok_or("non-string token")?;
            id_of.entry(s).or_insert(i as u32);
            // 3 = control, 4 = user-defined: literal text, matched before BPE
            if types[i] == 3 || types[i] == 4 {
                bytes.push(s.as_bytes().to_vec());
                special.push((s.as_bytes().to_vec(), i as u32));
            } else {
                // a normal token's string is in the byte-to-unicode alphabet
                bytes.push(
                    s.chars()
                        .map(|c| {
                            u2b.get(&c)
                                .copied()
                                .ok_or(format!("token {i} {s:?}: char outside the byte alphabet"))
                        })
                        .collect::<Result<Vec<u8>, String>>()?,
                );
            }
        }
        special.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then(a.1.cmp(&b.1)));
        // some vocabularies lack a few control bytes; those become `unk`
        let byte_tok: Vec<Option<u32>> = (0..256).map(|b| id_of.get(b2u[b].to_string().as_str()).copied()).collect();
        let ms = g
            .get("tokenizer.ggml.merges")
            .and_then(|v| v.as_arr())
            .ok_or("missing tokenizer.ggml.merges")?;
        let mut merges = HashMap::with_capacity(ms.len());
        for (r, m) in ms.iter().enumerate() {
            let m = m.as_str().ok_or("non-string merge")?;
            let (a, b) = m.split_once(' ').ok_or(format!("bad merge {m:?}"))?;
            let (Some(&ia), Some(&ib)) = (id_of.get(a), id_of.get(b)) else {
                return Err(format!("merge {m:?}: unknown token"));
            };
            let merged = format!("{a}{b}");
            let &im = id_of
                .get(merged.as_str())
                .ok_or(format!("merge {m:?}: result not in the vocabulary"))?;
            merges.entry((ia, ib)).or_insert((r as u32, im));
        }
        let opt = |k: &str| {
            g.get(k)
                .and_then(|v| v.as_u64())
                .map(|v| v as u32)
                .filter(|&v| (v as usize) < toks.len())
        };
        let (bos, eos) = (opt("tokenizer.ggml.bos_token_id"), opt("tokenizer.ggml.eos_token_id"));
        let unk = opt("tokenizer.ggml.unknown_token_id");
        let mut stop_ids: Vec<u32> = [eos, opt("tokenizer.ggml.eot_token_id")].into_iter().flatten().collect();
        stop_ids.dedup();
        Ok(Tokenizer {
            merges,
            bytes,
            byte_tok,
            unk,
            pre: Pre::Regex(regexes),
            special,
            bos,
            eos,
            stop_ids,
            add_bos: g.get("tokenizer.ggml.add_bos_token").and_then(|v| v.as_bool()).unwrap_or(false),
            chat_template: g.get("tokenizer.chat_template").and_then(|v| v.as_str()).map(str::to_string),
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.bytes.len()
    }

    fn encode_chunk(&self, w: &[u8], out: &mut Vec<u32>) {
        let mut t: Vec<u32> = w.iter().filter_map(|&b| self.byte_tok[b as usize].or(self.unk)).collect();
        while t.len() > 1 {
            // lowest-rank adjacent pair; ties go to the leftmost
            let mut best: Option<(u32, usize, u32)> = None;
            for i in 0..t.len() - 1 {
                if let Some(&(r, id)) = self.merges.get(&(t[i], t[i + 1]))
                    && best.is_none_or(|(br, _, _)| r < br)
                {
                    best = Some((r, i, id));
                }
            }
            let Some((_, i, id)) = best else { break };
            t[i] = id;
            t.remove(i + 1);
        }
        out.extend_from_slice(&t);
    }

    /// Encode text without special-token matching (special-token text is
    /// tokenized as ordinary text).
    pub fn encode_ordinary(&self, data: &[u8]) -> Vec<u32> {
        let mut out = vec![];
        match &self.pre {
            Pre::Legacy => {
                for c in chunks(data) {
                    self.encode_chunk(c, &mut out);
                }
            }
            Pre::Regex(res) => {
                // the regexes need text; invalid UTF-8 is replaced (U+FFFD)
                let s = String::from_utf8_lossy(data);
                let mut pieces = vec![(0, s.len())];
                for re in res {
                    pieces = pieces
                        .into_iter()
                        .flat_map(|(a, b)| split_all(re, &s[a..b]).into_iter().map(move |(x, y)| (a + x, a + y)))
                        .collect();
                }
                for (a, b) in pieces {
                    self.encode_chunk(&s.as_bytes()[a..b], &mut out);
                }
            }
        }
        out
    }

    /// Encode text. Special tokens (e.g. `<|im_start|>`) written in the text
    /// are recognised, as Hugging Face tokenizers do by default. No BOS is
    /// added; see `encode_prompt`.
    pub fn encode(&self, data: &[u8]) -> Vec<u32> {
        if self.special.is_empty() {
            return self.encode_ordinary(data);
        }
        let mut out = vec![];
        let (mut i, mut start) = (0, 0);
        while i < data.len() {
            if let Some((s, id)) = self.special.iter().find(|(s, _)| data[i..].starts_with(s)) {
                out.extend(self.encode_ordinary(&data[start..i]));
                out.push(*id);
                i += s.len();
                start = i;
            } else {
                i += 1;
            }
        }
        out.extend(self.encode_ordinary(&data[start..]));
        out
    }

    /// `encode`, plus the BOS token when the model asks for one.
    pub fn encode_prompt(&self, text: &str) -> Vec<u32> {
        let mut v = vec![];
        if self.add_bos
            && let Some(b) = self.bos
        {
            v.push(b);
        }
        v.extend(self.encode(text.as_bytes()));
        v
    }

    pub fn token_bytes(&self, id: u32) -> &[u8] {
        &self.bytes[id as usize]
    }

    pub fn decode(&self, ids: &[u32]) -> Vec<u8> {
        ids.iter().flat_map(|&i| self.bytes[i as usize].iter().copied()).collect()
    }

    /// Render a conversation with the model's chat template and the
    /// assistant prompt appended. Only ChatML templates (`<|im_start|>`,
    /// used by SmolLM2 and Qwen2.5) are supported; the default system
    /// message is taken from the template, as its Jinja code does when the
    /// conversation has none. Tool calls are not supported.
    pub fn apply_chat_template(&self, messages: &[(String, String)]) -> Result<String, String> {
        let tpl = self.chat_template.as_deref().ok_or("the model has no chat template")?;
        if !tpl.contains("<|im_start|>") {
            return Err("only ChatML chat templates are supported".into());
        }
        let mut out = String::new();
        if messages.first().is_none_or(|m| m.0 != "system")
            && let Some(sys) = default_system(tpl)
        {
            out += &format!("<|im_start|>system\n{sys}<|im_end|>\n");
        }
        for (role, content) in messages {
            out += &format!("<|im_start|>{role}\n{content}<|im_end|>\n");
        }
        out += "<|im_start|>assistant\n";
        Ok(out)
    }
}

/// The default system message in a ChatML template: the text of a string
/// literal `'<|im_start|>system\n...<|im_end|>` that holds the whole message
/// (not one that continues with `' + messages[0]['content']`), with escape
/// sequences decoded.
fn default_system(tpl: &str) -> Option<String> {
    // the newline after "system" is either an escape (`\n`) or a real newline
    for (q, lit) in [
        ('\'', "'<|im_start|>system\\n"),
        ('\'', "'<|im_start|>system\n"),
        ('"', "\"<|im_start|>system\\n"),
        ('"', "\"<|im_start|>system\n"),
    ] {
        let mut from = 0;
        while let Some(p) = tpl[from..].find(lit) {
            let rest = &tpl[from + p + lit.len()..];
            from += p + lit.len();
            let Some(end) = rest.find("<|im_end|>") else { break };
            let body = &rest[..end];
            if !body.replace(&format!("\\{q}"), "").contains(q) {
                return Some(body.replace("\\n", "\n").replace(&format!("\\{q}"), &q.to_string()));
            }
        }
    }
    None
}
