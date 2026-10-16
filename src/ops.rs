//! Numeric kernels. Every dot product uses one fixed summation order
//! (4 x 8-lane accumulators, combined in a fixed tree, then the tail), in
//! both the AVX2 and the portable path. A result therefore depends only on
//! its two input vectors, not on batch size, thread count or which other
//! requests share the batch: batched and unbatched generation are
//! bit-identical, which the tests check.

use rayon::prelude::*;

/// A weight matrix [rows, cols] (y = W x), f32 or int8 with one scale per row.
pub enum Mat {
    F32 {
        rows: usize,
        cols: usize,
        w: Vec<f32>,
    },
    Q8 {
        rows: usize,
        cols: usize,
        q: Vec<i8>,
        scale: Vec<f32>,
    },
}

impl Mat {
    pub fn rows(&self) -> usize {
        match self {
            Mat::F32 { rows, .. } | Mat::Q8 { rows, .. } => *rows,
        }
    }
    pub fn cols(&self) -> usize {
        match self {
            Mat::F32 { cols, .. } | Mat::Q8 { cols, .. } => *cols,
        }
    }
    pub fn bytes(&self) -> usize {
        match self {
            Mat::F32 { w, .. } => w.len() * 4,
            Mat::Q8 { q, scale, .. } => q.len() + scale.len() * 4,
        }
    }

    /// Symmetric per-row int8: q = round(w / s), s = max|w| / 127.
    pub fn quantize(&self) -> Mat {
        match self {
            Mat::Q8 { .. } => panic!("already quantized"),
            Mat::F32 { rows, cols, w } => {
                let mut q = vec![0i8; rows * cols];
                let mut scale = vec![0f32; *rows];
                for r in 0..*rows {
                    let row = &w[r * cols..(r + 1) * cols];
                    let m = row.iter().fold(0f32, |a, &x| a.max(x.abs()));
                    let s = if m > 0.0 { m / 127.0 } else { 1.0 };
                    scale[r] = s;
                    for (d, &x) in q[r * cols..(r + 1) * cols].iter_mut().zip(row) {
                        *d = (x / s).round().clamp(-127.0, 127.0) as i8;
                    }
                }
                Mat::Q8 {
                    rows: *rows,
                    cols: *cols,
                    q,
                    scale,
                }
            }
        }
    }

    fn row_dot(&self, r: usize, x: &[f32]) -> f32 {
        match self {
            Mat::F32 { cols, w, .. } => dot(&w[r * cols..(r + 1) * cols], x),
            Mat::Q8 { cols, q, scale, .. } => dot_q8(&q[r * cols..(r + 1) * cols], x) * scale[r],
        }
    }

    /// out[b, rows] = x[b, cols] W^T. Parallel over row tiles; each weight
    /// row is loaded once per call and used for every batch row.
    pub fn matmul(&self, x: &[f32], b: usize, out: &mut [f32]) {
        let (rows, cols) = (self.rows(), self.cols());
        assert_eq!(x.len(), b * cols);
        assert_eq!(out.len(), b * rows);
        const TILE: usize = 16;
        let mut t = vec![0f32; rows * b]; // [rows, b]
        t.par_chunks_mut(TILE * b).enumerate().for_each(|(ti, chunk)| {
            for (i, r) in (ti * TILE..(ti * TILE + TILE).min(rows)).enumerate() {
                for bi in 0..b {
                    chunk[i * b + bi] = self.row_dot(r, &x[bi * cols..(bi + 1) * cols]);
                }
            }
        });
        for r in 0..rows {
            for bi in 0..b {
                out[bi * rows + r] = t[r * b + bi];
            }
        }
    }
}

#[inline]
fn combine(acc: [[f32; 8]; 4]) -> f32 {
    let mut lanes = [0f32; 8];
    for (l, lane) in lanes.iter_mut().enumerate() {
        *lane = (acc[0][l] + acc[1][l]) + (acc[2][l] + acc[3][l]);
    }
    ((lanes[0] + lanes[4]) + (lanes[2] + lanes[6])) + ((lanes[1] + lanes[5]) + (lanes[3] + lanes[7]))
}

/// Portable reference with the same summation order and fused multiply-adds.
pub fn dot_portable(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let mut acc = [[0f32; 8]; 4];
    let main = n / 32 * 32;
    let mut i = 0;
    while i < main {
        for (k, accum) in acc.iter_mut().enumerate() {
            for l in 0..8 {
                accum[l] = a[i + 8 * k + l].mul_add(b[i + 8 * k + l], accum[l]);
            }
        }
        i += 32;
    }
    let mut s = combine(acc);
    for j in main..n {
        s = a[j].mul_add(b[j], s);
    }
    s
}

pub fn dot_q8_portable(q: &[i8], x: &[f32]) -> f32 {
    let n = q.len();
    let mut acc = [[0f32; 8]; 4];
    let main = n / 32 * 32;
    let mut i = 0;
    while i < main {
        for (k, accum) in acc.iter_mut().enumerate() {
            for l in 0..8 {
                accum[l] = (q[i + 8 * k + l] as f32).mul_add(x[i + 8 * k + l], accum[l]);
            }
        }
        i += 32;
    }
    let mut s = combine(acc);
    for j in main..n {
        s = (q[j] as f32).mul_add(x[j], s);
    }
    s
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx2,fma")]
    unsafe fn to_arr(v: __m256) -> [f32; 8] {
        let mut o = [0f32; 8];
        unsafe { _mm256_storeu_ps(o.as_mut_ptr(), v) };
        o
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len();
        let main = n / 32 * 32;
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        unsafe {
            let mut acc = [_mm256_setzero_ps(); 4];
            let mut i = 0;
            while i < main {
                for (k, accum) in acc.iter_mut().enumerate() {
                    *accum = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i + 8 * k)), _mm256_loadu_ps(pb.add(i + 8 * k)), *accum);
                }
                i += 32;
            }
            let mut s = super::combine([to_arr(acc[0]), to_arr(acc[1]), to_arr(acc[2]), to_arr(acc[3])]);
            for j in main..n {
                s = a[j].mul_add(b[j], s);
            }
            s
        }
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_q8(q: &[i8], x: &[f32]) -> f32 {
        let n = q.len();
        let main = n / 32 * 32;
        let (pq, px) = (q.as_ptr(), x.as_ptr());
        unsafe {
            let mut acc = [_mm256_setzero_ps(); 4];
            let mut i = 0;
            while i < main {
                for (k, accum) in acc.iter_mut().enumerate() {
                    let w8 = _mm_loadl_epi64(pq.add(i + 8 * k) as *const __m128i);
                    let w = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w8));
                    *accum = _mm256_fmadd_ps(w, _mm256_loadu_ps(px.add(i + 8 * k)), *accum);
                }
                i += 32;
            }
            let mut s = super::combine([to_arr(acc[0]), to_arr(acc[1]), to_arr(acc[2]), to_arr(acc[3])]);
            for j in main..n {
                s = (q[j] as f32).mul_add(x[j], s);
            }
            s
        }
    }
}

static SIMD: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// AVX2+FMA if the CPU has them and LLM_NO_SIMD is unset.
pub fn simd_enabled() -> bool {
    *SIMD.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            std::env::var_os("LLM_NO_SIMD").is_none() && is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    })
}

#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    if simd_enabled() {
        return unsafe { x86::dot(a, b) };
    }
    dot_portable(a, b)
}

#[inline]
pub fn dot_q8(q: &[i8], x: &[f32]) -> f32 {
    debug_assert_eq!(q.len(), x.len());
    #[cfg(target_arch = "x86_64")]
    if simd_enabled() {
        return unsafe { x86::dot_q8(q, x) };
    }
    dot_q8_portable(q, x)
}

pub fn rmsnorm(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    let ss = dot(x, x) / x.len() as f32;
    let inv = 1.0 / (ss + eps).sqrt();
    for ((o, &v), &g) in out.iter_mut().zip(x).zip(w) {
        *o = v * inv * g;
    }
}

pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simd_and_portable_are_bit_identical() {
        let mut s = 1u64;
        let mut r = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        };
        for n in [1, 7, 31, 32, 33, 256, 688, 1000] {
            let a: Vec<f32> = (0..n).map(|_| r()).collect();
            let b: Vec<f32> = (0..n).map(|_| r()).collect();
            let q: Vec<i8> = (0..n).map(|i| (i * 37 % 255 - 127) as i8).collect();
            assert_eq!(dot(&a, &b).to_bits(), dot_portable(&a, &b).to_bits(), "n={n}");
            assert_eq!(dot_q8(&q, &b).to_bits(), dot_q8_portable(&q, &b).to_bits(), "n={n}");
            let exact: f64 = a.iter().zip(&b).map(|(x, y)| *x as f64 * *y as f64).sum();
            assert!((dot(&a, &b) as f64 - exact).abs() < 1e-5 * n as f64);
        }
    }

    #[test]
    fn matmul_is_batch_invariant() {
        let (rows, cols) = (37, 100);
        let w: Vec<f32> = (0..rows * cols).map(|i| ((i * 7919) % 1000) as f32 / 1000.0 - 0.5).collect();
        let m = Mat::F32 { rows, cols, w };
        let x: Vec<f32> = (0..5 * cols).map(|i| ((i * 104729) % 997) as f32 / 997.0 - 0.5).collect();
        let mut all = vec![0f32; 5 * rows];
        m.matmul(&x, 5, &mut all);
        for b in 0..5 {
            let mut one = vec![0f32; rows];
            m.matmul(&x[b * cols..(b + 1) * cols], 1, &mut one);
            assert_eq!(one, all[b * rows..(b + 1) * rows]);
        }
        let q = m.quantize();
        let mut qa = vec![0f32; 5 * rows];
        q.matmul(&x, 5, &mut qa);
        for (a, b) in qa.iter().zip(&all) {
            assert!((a - b).abs() < 0.05, "{a} vs {b}");
        }
    }
}
