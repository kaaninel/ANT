/// CPU-resident AddrNet — pure Rust MLP, no Candle, no GPU overhead.
///
/// Mirrors the GPU AddrNet exactly (inference/argmax path only).
/// Weights are synced from the GPU model after each Phase B optimizer step.
///
/// Layout:
///   proj_in:  (ADDR_HIDDEN_DIM × D_MODEL)   — flat row-major f32
///   bin_embed:(ADDR_N_BINS × ADDR_HIDDEN_DIM) — flat row-major f32
///   mlp:      (ADDR_HIDDEN_DIM × ADDR_HIDDEN_DIM) — flat row-major f32
///   out_w:    (ADDR_N_BINS × ADDR_HIDDEN_DIM) — flat row-major f32
///
/// Forward (single vector, argmax mode):
///   h = proj_in * hidden              (D_MODEL → ADDR_HIDDEN_DIM)
///   for 8 steps:
///     logits = out_w * h              (ADDR_HIDDEN_DIM → ADDR_N_BINS), clamp [-30,30]
///     idx    = argmax(logits)
///     embed  = bin_embed[idx]         (row lookup)
///     h      = h + embed
///     h_mlp  = mlp * h
///     h      = h_mlp * sigmoid(h_mlp) (SiLU)
///   return 8-byte address

use crate::config::{D_MODEL, ADDR_HIDDEN_DIM, ADDR_N_BINS, ADDR_DEPTH, N_ADDR_NETS};

// ---------------------------------------------------------------------------
// Helper: dense matrix-vector multiply  y = W * x  (no bias)
// W is flat row-major: (rows × cols)
// ---------------------------------------------------------------------------
#[inline]
fn matvec(w: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; rows];
    for r in 0..rows {
        let row = &w[r * cols..(r + 1) * cols];
        let mut acc = 0.0f32;
        for c in 0..cols {
            acc += row[c] * x[c];
        }
        y[r] = acc;
    }
    y
}

#[inline]
fn argmax_f32(v: &[f32]) -> usize {
    v.iter().enumerate()
        .fold((0, f32::NEG_INFINITY), |(best_i, best_v), (i, &val)| {
            if val > best_v { (i, val) } else { (best_i, best_v) }
        })
        .0
}

#[inline]
fn silu(v: &[f32]) -> Vec<f32> {
    v.iter().map(|&x| x / (1.0 + (-x).exp())).collect()
}

// ---------------------------------------------------------------------------
// CpuAddrNet
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct CpuAddrNet {
    /// proj_in weight: (ADDR_HIDDEN_DIM × D_MODEL) flat row-major
    proj_in: Vec<f32>,
    /// bin_embed: (ADDR_N_BINS × ADDR_HIDDEN_DIM) flat row-major
    bin_embed: Vec<f32>,
    /// mlp weight: (ADDR_HIDDEN_DIM × ADDR_HIDDEN_DIM) flat row-major
    mlp: Vec<f32>,
    /// out weight: (ADDR_N_BINS × ADDR_HIDDEN_DIM) flat row-major
    out_w: Vec<f32>,
}

impl CpuAddrNet {
    /// Construct from flat row-major weight matrices extracted from the GPU model.
    /// Each inner Vec is a row; we flatten them.
    pub fn from_weights(
        proj_in: Vec<Vec<f32>>,
        bin_embed: Vec<Vec<f32>>,
        mlp: Vec<Vec<f32>>,
        out_w: Vec<Vec<f32>>,
    ) -> Self {
        Self {
            proj_in:   proj_in.into_iter().flatten().collect(),
            bin_embed: bin_embed.into_iter().flatten().collect(),
            mlp:       mlp.into_iter().flatten().collect(),
            out_w:     out_w.into_iter().flatten().collect(),
        }
    }

    /// Run one argmax forward pass: hidden (D_MODEL,) → 8-byte address.
    pub fn forward(&self, hidden: &[f32]) -> Vec<u8> {
        debug_assert_eq!(hidden.len(), D_MODEL);
        let mut h = matvec(&self.proj_in, hidden, ADDR_HIDDEN_DIM, D_MODEL);
        let mut addr = Vec::with_capacity(ADDR_DEPTH);

        for _ in 0..ADDR_DEPTH {
            let raw = matvec(&self.out_w, &h, ADDR_N_BINS, ADDR_HIDDEN_DIM);
            let logits: Vec<f32> = raw.iter().map(|v| v.clamp(-30.0, 30.0)).collect();
            let idx = argmax_f32(&logits);
            addr.push(idx as u8);

            // embed = row idx of bin_embed
            let embed = &self.bin_embed[idx * ADDR_HIDDEN_DIM..(idx + 1) * ADDR_HIDDEN_DIM];
            for (hi, &ei) in h.iter_mut().zip(embed.iter()) { *hi += ei; }

            let h_mlp = matvec(&self.mlp, &h, ADDR_HIDDEN_DIM, ADDR_HIDDEN_DIM);
            h = silu(&h_mlp);
        }
        addr
    }

    /// Batch forward: hidden_batch: B × D_MODEL → B addresses each of length ADDR_DEPTH.
    pub fn forward_batch(&self, hidden_batch: &[Vec<f32>]) -> Vec<Vec<u8>> {
        hidden_batch.iter().map(|h| self.forward(h)).collect()
    }
}

// ---------------------------------------------------------------------------
// CpuAddrBank — all N_ADDR_NETS nets in one struct, thread-safe
// ---------------------------------------------------------------------------

use std::sync::{Arc, RwLock};

/// Thread-safe bank of CPU AddrNets, kept in sync with the GPU model.
///
/// Write lock: only during weight sync (brief).
/// Read lock: many concurrent forward calls.
pub struct CpuAddrBank {
    nets: Arc<RwLock<Vec<CpuAddrNet>>>,
}

impl CpuAddrBank {
    /// Create a zeroed bank (weights all zero until first sync).
    pub fn new_zeroed() -> Self {
        let blank_net = CpuAddrNet {
            proj_in:   vec![0.0; ADDR_HIDDEN_DIM * D_MODEL],
            bin_embed: vec![0.0; ADDR_N_BINS * ADDR_HIDDEN_DIM],
            mlp:       vec![0.0; ADDR_HIDDEN_DIM * ADDR_HIDDEN_DIM],
            out_w:     vec![0.0; ADDR_N_BINS * ADDR_HIDDEN_DIM],
        };
        Self { nets: Arc::new(RwLock::new(vec![blank_net; N_ADDR_NETS])) }
    }

    /// Replace all nets with freshly synced weights.
    /// `new_nets` must have exactly N_ADDR_NETS entries.
    pub fn update(&self, new_nets: Vec<CpuAddrNet>) {
        assert_eq!(new_nets.len(), N_ADDR_NETS);
        let mut guard = self.nets.write().expect("CpuAddrBank poisoned");
        *guard = new_nets;
    }

    /// Run all N_ADDR_NETS forward passes on a single hidden vector.
    /// Returns N_ADDR_NETS addresses, each of ADDR_DEPTH bytes.
    pub fn forward_all(&self, hidden: &[f32]) -> Vec<Vec<u8>> {
        let guard = self.nets.read().expect("CpuAddrBank poisoned");
        guard.iter().map(|net| net.forward(hidden)).collect()
    }

    /// Batch version: returns (N, B, ADDR_DEPTH) — outer index = net.
    pub fn forward_all_batch(&self, hidden_batch: &[Vec<f32>]) -> Vec<Vec<Vec<u8>>> {
        let guard = self.nets.read().expect("CpuAddrBank poisoned");
        guard.iter().map(|net| net.forward_batch(hidden_batch)).collect()
    }

    pub fn arc_clone(&self) -> Arc<RwLock<Vec<CpuAddrNet>>> {
        Arc::clone(&self.nets)
    }
}
