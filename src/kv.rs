//! Paged KV cache and its block allocator with prefix caching.
//!
//! Memory is a pool of fixed-size blocks (`block_size` positions each, all
//! layers). A sequence owns a block table. Full blocks can be registered
//! under a chain hash of every token up to the block's end; a later prompt
//! with the same prefix takes a reference on those blocks instead of
//! recomputing them. Blocks whose reference count drops to zero stay
//! cached (evictable, least recently released first) until the space is
//! needed.

use std::collections::{BTreeMap, HashMap, VecDeque};

pub struct KvCache {
    block_size: usize,
    head_dim: usize,
    kv_dim: usize,
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
}

impl KvCache {
    pub fn new(layers: usize, kv_heads: usize, head_dim: usize, blocks: usize, block_size: usize) -> KvCache {
        let kv_dim = kv_heads * head_dim;
        let n = blocks * block_size * kv_dim;
        KvCache {
            block_size,
            head_dim,
            kv_dim,
            k: vec![vec![0.0; n]; layers],
            v: vec![vec![0.0; n]; layers],
        }
    }
    pub fn block_size(&self) -> usize {
        self.block_size
    }
    pub fn bytes(&self) -> usize {
        2 * self.k.len() * self.k[0].len() * 4
    }
    pub fn write(&mut self, layer: usize, slot: usize, k: &[f32], v: &[f32]) {
        let o = slot * self.kv_dim;
        self.k[layer][o..o + self.kv_dim].copy_from_slice(k);
        self.v[layer][o..o + self.kv_dim].copy_from_slice(v);
    }
    /// Fill a block with NaN (debug aid: any read of freed KV data then
    /// corrupts the output instead of silently reusing stale values).
    pub fn poison(&mut self, block: usize) {
        let n = self.block_size * self.kv_dim;
        for l in 0..self.k.len() {
            self.k[l][block * n..(block + 1) * n].fill(f32::NAN);
            self.v[l][block * n..(block + 1) * n].fill(f32::NAN);
        }
    }
    #[inline]
    pub fn key(&self, layer: usize, slot: usize, kv_head: usize) -> &[f32] {
        let o = slot * self.kv_dim + kv_head * self.head_dim;
        &self.k[layer][o..o + self.head_dim]
    }
    #[inline]
    pub fn value(&self, layer: usize, slot: usize, kv_head: usize) -> &[f32] {
        let o = slot * self.kv_dim + kv_head * self.head_dim;
        &self.v[layer][o..o + self.head_dim]
    }
}

/// Chain hash of a full block: depends on every token from position 0.
pub fn block_hash(parent: u64, tokens: &[u32]) -> u64 {
    // FNV-1a over the parent hash and the tokens: deterministic across runs
    let mut h: u64 = 0xcbf2_9ce4_8422_2325 ^ parent.rotate_left(17);
    for &t in tokens {
        for b in t.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01B3);
        }
    }
    h
}

#[derive(Default, Debug, Clone, Copy, serde::Serialize)]
pub struct PoolStats {
    pub total: usize,
    pub in_use: usize,
    pub cached_free: usize,
    pub prefix_hits: u64,
    pub prefix_lookups: u64,
    pub evictions: u64,
}

pub struct BlockPool {
    refs: Vec<u32>,
    hash: Vec<Option<u64>>,
    /// blocks never registered (or evicted) with no references
    free: VecDeque<u32>,
    /// hash -> block for registered blocks
    cached: HashMap<u64, u32>,
    /// registered blocks with no references, by release order
    evictable: BTreeMap<u64, u32>,
    evict_key: Vec<u64>,
    tick: u64,
    pub stats: PoolStats,
}

impl BlockPool {
    pub fn new(blocks: usize) -> BlockPool {
        BlockPool {
            refs: vec![0; blocks],
            hash: vec![None; blocks],
            free: (0..blocks as u32).collect(),
            cached: HashMap::new(),
            evictable: BTreeMap::new(),
            evict_key: vec![0; blocks],
            tick: 0,
            stats: PoolStats {
                total: blocks,
                ..Default::default()
            },
        }
    }

    pub fn available(&self) -> usize {
        self.free.len() + self.evictable.len()
    }

    /// A fresh, unregistered block (evicting the oldest cached one if needed).
    pub fn alloc(&mut self) -> Option<u32> {
        let b = if let Some(b) = self.free.pop_front() {
            b
        } else {
            let (&k, &b) = self.evictable.iter().next()?;
            self.evictable.remove(&k);
            let h = self.hash[b as usize].take().expect("evictable blocks are registered");
            self.cached.remove(&h);
            self.stats.evictions += 1;
            b
        };
        self.refs[b as usize] = 1;
        Some(b)
    }

    /// Take a reference on the cached block for `hash`, if any.
    pub fn lookup(&mut self, hash: u64) -> Option<u32> {
        self.stats.prefix_lookups += 1;
        let &b = self.cached.get(&hash)?;
        if self.refs[b as usize] == 0 {
            self.evictable.remove(&self.evict_key[b as usize]);
        }
        self.refs[b as usize] += 1;
        self.stats.prefix_hits += 1;
        Some(b)
    }

    /// Record that block `b` (full, owned by the caller) holds the prefix
    /// with this hash. A duplicate of an already cached prefix stays private.
    pub fn register(&mut self, b: u32, hash: u64) {
        if self.hash[b as usize].is_none() && !self.cached.contains_key(&hash) {
            self.hash[b as usize] = Some(hash);
            self.cached.insert(hash, b);
        }
    }

    /// Drop a reference. True if the block is now free and unregistered
    /// (its contents are garbage from now on).
    pub fn release(&mut self, b: u32) -> bool {
        let r = &mut self.refs[b as usize];
        assert!(*r > 0, "double free of block {b}");
        *r -= 1;
        if *r == 0 {
            if self.hash[b as usize].is_some() {
                self.tick += 1;
                self.evict_key[b as usize] = self.tick;
                self.evictable.insert(self.tick, b);
            } else {
                self.free.push_back(b);
                return true;
            }
        }
        false
    }

    pub fn refcount(&self, b: u32) -> u32 {
        self.refs[b as usize]
    }

    pub fn snapshot(&self) -> PoolStats {
        let mut s = self.stats;
        s.in_use = self.refs.iter().filter(|&&r| r > 0).count();
        s.cached_free = self.evictable.len();
        s
    }

    /// Internal consistency (used by tests).
    pub fn check(&self) -> Result<(), String> {
        let n = self.refs.len();
        let zero = self.refs.iter().filter(|&&r| r == 0).count();
        if zero != self.free.len() + self.evictable.len() {
            return Err(format!(
                "{zero} unreferenced blocks but {} free + {} evictable",
                self.free.len(),
                self.evictable.len()
            ));
        }
        for (&h, &b) in &self.cached {
            if self.hash[b as usize] != Some(h) {
                return Err(format!("cache map and block {b} disagree"));
            }
        }
        if self
            .free
            .iter()
            .any(|&b| self.hash[b as usize].is_some() || self.refs[b as usize] != 0)
        {
            return Err("free list holds a registered or referenced block".into());
        }
        let _ = n;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_blocks_are_shared_then_evicted_lru() {
        let mut p = BlockPool::new(3);
        let a = p.alloc().unwrap();
        p.register(a, 11);
        let b = p.alloc().unwrap();
        p.register(b, 22);
        assert_eq!(p.lookup(11), Some(a));
        assert_eq!(p.refcount(a), 2);
        p.release(a);
        p.release(a);
        p.release(b);
        p.check().unwrap();
        // a was released before b: a is evicted first, but free blocks go first
        let c = p.alloc().unwrap();
        assert!(c != a && c != b);
        let d = p.alloc().unwrap();
        assert_eq!(d, a);
        assert_eq!(p.lookup(11), None);
        assert_eq!(p.lookup(22), Some(b));
        assert_eq!(p.alloc(), None);
        p.check().unwrap();
    }
}
