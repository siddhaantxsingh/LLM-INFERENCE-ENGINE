//! Token sampling: greedy, temperature, top-k, top-p (nucleus), with a
//! per-request seeded generator so results are reproducible.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SamplingParams {
    /// 0 = greedy
    #[serde(default)]
    pub temperature: f32,
    /// 0 = disabled
    #[serde(default)]
    pub top_k: usize,
    /// 1.0 = disabled
    #[serde(default = "one")]
    pub top_p: f32,
    #[serde(default)]
    pub seed: u64,
    #[serde(default = "default_max")]
    pub max_tokens: usize,
    /// generation stops when the text ends with any of these
    #[serde(default)]
    pub stop: Vec<String>,
    /// keep generating after the model's end-of-sequence token (benchmarks)
    #[serde(default)]
    pub ignore_eos: bool,
}

fn one() -> f32 {
    1.0
}
fn default_max() -> usize {
    64
}

impl Default for SamplingParams {
    fn default() -> Self {
        SamplingParams {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            seed: 0,
            max_tokens: 64,
            stop: vec![],
            ignore_eos: false,
        }
    }
}

/// SplitMix64.
#[derive(Clone, Debug)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// uniform in [0, 1)
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

pub fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &v) in logits.iter().enumerate() {
        if v > logits[best] {
            best = i;
        }
    }
    best as u32
}

/// The candidate set and probabilities after temperature, top-k and top-p.
pub fn distribution(logits: &[f32], p: &SamplingParams) -> Vec<(u32, f64)> {
    let t = p.temperature.max(1e-6) as f64;
    let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
    // sort by logit desc, index asc: deterministic
    idx.sort_by(|&a, &b| logits[b as usize].total_cmp(&logits[a as usize]).then(a.cmp(&b)));
    if p.top_k > 0 {
        idx.truncate(p.top_k.max(1));
    }
    let m = logits[idx[0] as usize] as f64;
    let mut probs: Vec<(u32, f64)> = idx.iter().map(|&i| (i, ((logits[i as usize] as f64 - m) / t).exp())).collect();
    let z: f64 = probs.iter().map(|x| x.1).sum();
    for x in &mut probs {
        x.1 /= z;
    }
    if p.top_p < 1.0 {
        let mut cum = 0.0;
        let mut keep = 0;
        for x in &probs {
            keep += 1;
            cum += x.1;
            if cum >= p.top_p as f64 {
                break;
            }
        }
        probs.truncate(keep.max(1));
        let z: f64 = probs.iter().map(|x| x.1).sum();
        for x in &mut probs {
            x.1 /= z;
        }
    }
    probs
}

pub fn sample(logits: &[f32], p: &SamplingParams, rng: &mut Rng) -> u32 {
    if p.temperature <= 0.0 {
        return argmax(logits);
    }
    let probs = distribution(logits, p);
    let u = rng.uniform();
    let mut cum = 0.0;
    for &(i, pr) in &probs {
        cum += pr;
        if u < cum {
            return i;
        }
    }
    probs.last().unwrap().0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_k_and_top_p_restrict_the_support() {
        let logits = [1.0, 3.0, 2.0, 0.0, 2.5];
        let p = SamplingParams {
            temperature: 1.0,
            top_k: 2,
            ..Default::default()
        };
        let d = distribution(&logits, &p);
        assert_eq!(d.iter().map(|x| x.0).collect::<Vec<_>>(), vec![1, 4]);
        let p = SamplingParams {
            temperature: 1.0,
            top_p: 0.5,
            ..Default::default()
        };
        let d = distribution(&logits, &p);
        assert!(d.len() < 5 && d[0].0 == 1);
        assert!((d.iter().map(|x| x.1).sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn sampling_matches_the_distribution() {
        let logits = [0.0f32, 1.0, 2.0, -1.0];
        let p = SamplingParams {
            temperature: 0.7,
            ..Default::default()
        };
        let d = distribution(&logits, &p);
        let mut rng = Rng::new(42);
        let mut counts = [0usize; 4];
        let n = 200_000;
        for _ in 0..n {
            counts[sample(&logits, &p, &mut rng) as usize] += 1;
        }
        for (i, pr) in d {
            let got = counts[i as usize] as f64 / n as f64;
            assert!((got - pr).abs() < 0.005, "token {i}: {got} vs {pr}");
        }
    }

    #[test]
    fn greedy_ignores_rng() {
        let logits = [0.1f32, 0.5, 0.5, 0.2];
        assert_eq!(sample(&logits, &SamplingParams::default(), &mut Rng::new(1)), 1);
    }
}
