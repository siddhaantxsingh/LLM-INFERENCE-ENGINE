//! Greedy speculative decoding for one sequence.
//!
//! A small draft model proposes `k` tokens one at a time; the target model
//! scores all of them in a single batched forward pass (k + 1 positions).
//! The longest prefix on which the target's own greedy choice agrees with
//! the draft is accepted, followed by the target's token at the first
//! disagreement (or a bonus token if all k were accepted). Because the
//! kernels are batch-invariant, the target's logits for a position are bit
//! for bit the same whether it decodes one token or verifies k at once, so
//! the output is exactly the target's greedy output; only the number of
//! target passes changes.

use crate::kv::KvCache;
use crate::model::{Model, StepToken};
use crate::sampler::argmax;

pub struct SpecStats {
    pub target_passes: usize,
    pub draft_passes: usize,
    pub proposed: usize,
    pub accepted: usize,
}

struct Runner<'a> {
    m: &'a Model,
    kv: KvCache,
    blocks: Vec<u32>,
    /// tokens whose K/V are in the cache
    len: usize,
}

impl<'a> Runner<'a> {
    fn new(m: &'a Model) -> Runner<'a> {
        let bs = 16;
        let nb = m.cfg.max_seq.div_ceil(bs);
        Runner {
            m,
            kv: KvCache::new(m.cfg.layers, m.cfg.kv_heads, m.cfg.head_dim(), nb, bs),
            blocks: (0..nb as u32).collect(),
            len: 0,
        }
    }

    /// Feed tokens[len..] (positions len..), return logits for each fed token.
    fn feed(&mut self, tokens: &[u32]) -> Vec<Vec<f32>> {
        let toks: Vec<StepToken> = (self.len..tokens.len())
            .map(|p| StepToken {
                token: tokens[p],
                pos: p,
                blocks: &self.blocks,
                want_logits: true,
            })
            .collect();
        let out = self.m.forward(&toks, &mut self.kv);
        self.len = tokens.len();
        out
    }
}

/// Greedy decode `max_tokens` tokens after `prompt`.
pub fn generate(target: &Model, draft: &Model, prompt: &[u32], max_tokens: usize, k: usize) -> (Vec<u32>, SpecStats) {
    assert_eq!(target.cfg.vocab, draft.cfg.vocab, "draft and target must share the vocabulary");
    let limit = (prompt.len() + max_tokens).min(target.cfg.max_seq).min(draft.cfg.max_seq);
    let mut st = SpecStats {
        target_passes: 0,
        draft_passes: 0,
        proposed: 0,
        accepted: 0,
    };
    let (mut t, mut d) = (Runner::new(target), Runner::new(draft));
    let mut seq = prompt.to_vec();
    let first = argmax(t.feed(&seq).last().unwrap());
    st.target_passes += 1;
    seq.push(first);
    while seq.len() < limit {
        // draft: catch up on accepted tokens, then propose up to k
        let room = (limit - seq.len()).min(k);
        let mut prop = seq.clone();
        for _ in 0..room {
            let l = d.feed(&prop);
            st.draft_passes += 1;
            prop.push(argmax(l.last().unwrap()));
        }
        let proposals = &prop[seq.len()..];
        st.proposed += proposals.len();
        // target: verify the last accepted token plus all proposals in one pass
        t.len = seq.len() - 1;
        let logits = t.feed(&prop[..seq.len() + proposals.len()]);
        st.target_passes += 1;
        let mut accepted = 0;
        for (i, &p) in proposals.iter().enumerate() {
            if argmax(&logits[i]) == p {
                accepted += 1;
            } else {
                break;
            }
        }
        st.accepted += accepted;
        seq.extend_from_slice(&proposals[..accepted]);
        if seq.len() < limit {
            // the target's own next token: at the first mismatch, or a bonus after all accepted
            seq.push(argmax(&logits[accepted]));
        }
        // caches past the accepted tokens hold stale entries; they are overwritten before being read
        t.len = t.len.min(seq.len() - 1);
        d.len = d.len.min(seq.len() - 1);
    }
    (seq[prompt.len()..].to_vec(), st)
}
