//! GGUF reader (format versions 2 and 3): metadata, tensor directory and
//! tensor data, with dequantisation of the types this engine supports
//! (F32, F16, BF16, Q8_0) to f32.
//!
//! Layout (little-endian): magic "GGUF", u32 version, u64 tensor count,
//! u64 metadata count, the metadata key/values, the tensor infos (name,
//! dims, type, offset), padding to `general.alignment` (default 32), then
//! the tensor data. Tensor dims are listed fastest-varying first, so a
//! matrix [rows, cols] in row-major order has dims [cols, rows].
//!
//! Every read is bounds-checked: a truncated or corrupt file gives an
//! error, never a panic or an oversized allocation.

use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    Arr(Vec<Value>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        Some(match *self {
            Value::U8(v) => v as u64,
            Value::U16(v) => v as u64,
            Value::U32(v) => v as u64,
            Value::U64(v) => v,
            Value::I8(v) if v >= 0 => v as u64,
            Value::I16(v) if v >= 0 => v as u64,
            Value::I32(v) if v >= 0 => v as u64,
            Value::I64(v) if v >= 0 => v as u64,
            _ => return None,
        })
    }
    pub fn as_f64(&self) -> Option<f64> {
        match *self {
            Value::F32(v) => Some(v as f64),
            Value::F64(v) => Some(v),
            _ => self.as_u64().map(|v| v as f64),
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_bool(&self) -> Option<bool> {
        match *self {
            Value::Bool(b) => Some(b),
            _ => None,
        }
    }
    pub fn as_arr(&self) -> Option<&[Value]> {
        match self {
            Value::Arr(a) => Some(a),
            _ => None,
        }
    }
}

/// ggml tensor types (the ids are fixed by the format).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    BF16,
    Q8_0,
    /// any other ggml type id (k-quants, i-quants, ...): listed, not loadable
    Other(u32),
}

impl DType {
    pub fn from_id(id: u32) -> DType {
        match id {
            0 => DType::F32,
            1 => DType::F16,
            8 => DType::Q8_0,
            30 => DType::BF16,
            x => DType::Other(x),
        }
    }
    pub fn id(self) -> u32 {
        match self {
            DType::F32 => 0,
            DType::F16 => 1,
            DType::Q8_0 => 8,
            DType::BF16 => 30,
            DType::Other(x) => x,
        }
    }
    /// (elements per block, bytes per block)
    fn block(self) -> Option<(usize, usize)> {
        match self {
            DType::F32 => Some((1, 4)),
            DType::F16 | DType::BF16 => Some((1, 2)),
            DType::Q8_0 => Some((32, 34)),
            DType::Other(_) => None,
        }
    }
    /// Bytes for `n` elements (None if unsupported or not a whole number of blocks).
    pub fn bytes_for(self, n: usize) -> Option<usize> {
        let (be, bb) = self.block()?;
        if !n.is_multiple_of(be) {
            return None;
        }
        (n / be).checked_mul(bb)
    }
}

#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub name: String,
    /// fastest-varying first (ggml `ne`)
    pub dims: Vec<u64>,
    pub dtype: DType,
    /// relative to the start of the data section
    pub offset: u64,
}

impl TensorInfo {
    pub fn n_elements(&self) -> usize {
        self.dims.iter().product::<u64>() as usize
    }
}

pub struct Gguf {
    pub version: u32,
    pub meta: BTreeMap<String, Value>,
    pub tensors: Vec<TensorInfo>,
    pub alignment: usize,
    data_start: usize,
    buf: Vec<u8>,
}

struct Cursor<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self
            .p
            .checked_add(n)
            .filter(|&e| e <= self.b.len())
            .ok_or_else(|| format!("truncated file: need {n} bytes at offset {}, file has {}", self.p, self.b.len()))?;
        let s = &self.b[self.p..end];
        self.p = end;
        Ok(s)
    }
    fn arr<const N: usize>(&mut self) -> Result<[u8; N], String> {
        Ok(self.take(N)?.try_into().unwrap())
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.arr()?))
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.arr()?))
    }
    fn remaining(&self) -> usize {
        self.b.len() - self.p
    }
    fn string(&mut self) -> Result<String, String> {
        let n = self.u64()?;
        if n > self.remaining() as u64 {
            return Err(format!("string length {n} exceeds the file"));
        }
        let s = self.take(n as usize)?;
        String::from_utf8(s.to_vec()).map_err(|_| format!("invalid UTF-8 in string at offset {}", self.p - s.len()))
    }
    fn value(&mut self, ty: u32, depth: usize) -> Result<Value, String> {
        Ok(match ty {
            0 => Value::U8(self.arr::<1>()?[0]),
            1 => Value::I8(self.arr::<1>()?[0] as i8),
            2 => Value::U16(u16::from_le_bytes(self.arr()?)),
            3 => Value::I16(i16::from_le_bytes(self.arr()?)),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(i32::from_le_bytes(self.arr()?)),
            6 => Value::F32(f32::from_le_bytes(self.arr()?)),
            7 => match self.arr::<1>()?[0] {
                0 => Value::Bool(false),
                1 => Value::Bool(true),
                b => return Err(format!("invalid bool byte {b}")),
            },
            8 => Value::Str(self.string()?),
            9 => {
                if depth > 4 {
                    return Err("arrays nested too deeply".into());
                }
                let ety = self.u32()?;
                let n = self.u64()?;
                // every element takes at least one byte: reject impossible counts before allocating
                if n > self.remaining() as u64 {
                    return Err(format!("array of {n} elements exceeds the file"));
                }
                let mut v = Vec::with_capacity(n as usize);
                for _ in 0..n {
                    v.push(self.value(ety, depth + 1)?);
                }
                Value::Arr(v)
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(i64::from_le_bytes(self.arr()?)),
            12 => Value::F64(f64::from_le_bytes(self.arr()?)),
            t => return Err(format!("unknown metadata value type {t}")),
        })
    }
}

pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if man == 0 {
            sign
        } else {
            // subnormal: value = man * 2^-24
            let v = man as f32 * (1.0 / 16_777_216.0);
            return if sign != 0 { -v } else { v };
        }
    } else if exp == 31 {
        sign | 0x7f80_0000 | (man << 13)
    } else {
        sign | ((exp + 112) << 23) | (man << 13)
    };
    f32::from_bits(bits)
}

/// Round-to-nearest-even f32 -> f16 (used by tests and the fixture writer).
pub fn f32_to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x7f_ffff;
    if exp == 255 {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 31 {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let rem = m & ((1 << shift) - 1);
        let mut r = m >> shift;
        if rem > half || (rem == half && r & 1 == 1) {
            r += 1;
        }
        return sign | r as u16;
    }
    let mut r = ((e as u32) << 10) | (man >> 13);
    let rem = man & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && r & 1 == 1) {
        r += 1; // may carry into the exponent, which is correct
    }
    sign | r as u16
}

impl Gguf {
    pub fn open(path: &std::path::Path) -> Result<Gguf, String> {
        let buf = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Gguf::parse(buf)
    }

    pub fn parse(buf: Vec<u8>) -> Result<Gguf, String> {
        let mut c = Cursor { b: &buf, p: 0 };
        if c.take(4)? != b"GGUF" {
            return Err("not a GGUF file (bad magic)".into());
        }
        let version = c.u32()?;
        if !(2..=3).contains(&version) {
            return Err(format!("unsupported GGUF version {version}"));
        }
        let n_tensors = c.u64()?;
        let n_kv = c.u64()?;
        if n_tensors > c.remaining() as u64 || n_kv > c.remaining() as u64 {
            return Err("tensor or metadata count exceeds the file".into());
        }
        let mut meta = BTreeMap::new();
        for _ in 0..n_kv {
            let k = c.string()?;
            let ty = c.u32()?;
            let v = c.value(ty, 0)?;
            if meta.insert(k.clone(), v).is_some() {
                return Err(format!("duplicate metadata key {k}"));
            }
        }
        let mut tensors = Vec::with_capacity(n_tensors as usize);
        for _ in 0..n_tensors {
            let name = c.string()?;
            let nd = c.u32()?;
            if nd == 0 || nd > 4 {
                return Err(format!("{name}: {nd} dimensions"));
            }
            let dims = (0..nd).map(|_| c.u64()).collect::<Result<Vec<_>, _>>()?;
            let dtype = DType::from_id(c.u32()?);
            let offset = c.u64()?;
            tensors.push(TensorInfo { name, dims, dtype, offset });
        }
        let alignment = match meta.get("general.alignment") {
            None => 32,
            Some(v) => v
                .as_u64()
                .filter(|a| *a > 0 && a.is_power_of_two())
                .ok_or("invalid general.alignment")? as usize,
        };
        let data_start = c.p.div_ceil(alignment) * alignment;
        let g = Gguf {
            version,
            meta,
            tensors,
            alignment,
            data_start,
            buf,
        };
        // check every tensor of a supported type lies inside the file
        for t in &g.tensors {
            let n = t
                .dims
                .iter()
                .try_fold(1u64, |a, &d| a.checked_mul(d))
                .ok_or(format!("{}: size overflows", t.name))?;
            if !t.offset.is_multiple_of(alignment as u64) {
                return Err(format!("{}: offset {} not aligned to {alignment}", t.name, t.offset));
            }
            if let Some(nb) = t.dtype.bytes_for(n as usize) {
                let end = (g.data_start as u64).checked_add(t.offset).and_then(|s| s.checked_add(nb as u64));
                if end.is_none_or(|e| e > g.buf.len() as u64) {
                    return Err(format!("{}: data extends past the end of the file", t.name));
                }
            } else if matches!(t.dtype, DType::F32 | DType::F16 | DType::BF16 | DType::Q8_0) {
                return Err(format!("{}: {n} elements is not a whole number of {:?} blocks", t.name, t.dtype));
            }
        }
        Ok(g)
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.meta.get(key)
    }
    pub fn get_u64(&self, key: &str) -> Result<u64, String> {
        self.get(key)
            .and_then(Value::as_u64)
            .ok_or(format!("missing or non-integer metadata {key}"))
    }
    pub fn get_f64(&self, key: &str) -> Result<f64, String> {
        self.get(key)
            .and_then(Value::as_f64)
            .ok_or(format!("missing or non-numeric metadata {key}"))
    }
    pub fn get_str(&self, key: &str) -> Result<&str, String> {
        self.get(key)
            .and_then(Value::as_str)
            .ok_or(format!("missing or non-string metadata {key}"))
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Raw bytes of a tensor (supported types only).
    pub fn tensor_bytes(&self, t: &TensorInfo) -> Result<&[u8], String> {
        let nb = t
            .dtype
            .bytes_for(t.n_elements())
            .ok_or(format!("{}: tensor type {:?} is not supported", t.name, t.dtype))?;
        let s = self.data_start + t.offset as usize;
        Ok(&self.buf[s..s + nb])
    }

    /// Tensor `name` as f32, checking its shape (dims fastest first).
    pub fn tensor_f32(&self, name: &str, dims: &[usize]) -> Result<Vec<f32>, String> {
        let t = self.tensor(name).ok_or(format!("missing tensor {name}"))?;
        if t.dims.len() != dims.len() || t.dims.iter().zip(dims).any(|(&a, &b)| a as usize != b) {
            return Err(format!("{name}: dims {:?}, expected {dims:?}", t.dims));
        }
        let raw = self.tensor_bytes(t)?;
        Ok(dequantize(t.dtype, raw))
    }
}

/// Convert raw tensor bytes of a supported type to f32.
pub fn dequantize(dtype: DType, raw: &[u8]) -> Vec<f32> {
    match dtype {
        DType::F32 => raw.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect(),
        DType::F16 => raw.as_chunks::<2>().0.iter().map(|b| f16_to_f32(u16::from_le_bytes(*b))).collect(),
        DType::BF16 => raw
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| f32::from_bits((u16::from_le_bytes(*b) as u32) << 16))
            .collect(),
        DType::Q8_0 => {
            let mut out = Vec::with_capacity(raw.len() / 34 * 32);
            for blk in raw.as_chunks::<34>().0 {
                let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                out.extend(blk[2..].iter().map(|&q| q as i8 as f32 * d));
            }
            out
        }
        DType::Other(t) => panic!("unsupported tensor type {t}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_conversion_round_trips() {
        for h in 0..=u16::MAX {
            let f = f16_to_f32(h);
            if f.is_nan() {
                assert!(f16_to_f32(f32_to_f16(f)).is_nan());
            } else {
                assert_eq!(f32_to_f16(f), h, "{h:#x} -> {f}");
            }
        }
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x0001), 5.960_464_5e-8);
        assert_eq!(f32_to_f16(65520.0), 0x7c00); // rounds to infinity
    }
}
