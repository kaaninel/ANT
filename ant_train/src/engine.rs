/// ANT Engine — Per-token READ→PROCESS→WRITE cycle.
///
/// Training: two-pass encode() for correct address distribution.
/// Inference: true per-token generate() with live trie updates.

use std::sync::Arc;
use candle_core::{Device, Result, Tensor, DType, Module, IndexOp};
use rand::Rng;

use crate::config::*;
use crate::model::{ANT, KVCache};
use crate::trie::MemorySystem;
use crate::cpu_addr::CpuAddrBank;

/// Handle for an in-flight background trie read.
/// Call `resolve()` to block until done and materialise GPU tensors.
pub struct PrefetchHandle {
    rx: std::sync::mpsc::Receiver<(Vec<f32>, Vec<bool>)>,
    pub b: usize,
}

impl PrefetchHandle {
    pub fn resolve(self, device: &Device) -> Result<(Tensor, Tensor, Tensor)> {
        let (vecs_flat, mask_flat) = self.rx.recv()
            .map_err(|_| candle_core::Error::Msg("prefetch trie-read thread panicked".to_string()))?;
        let s = N_MEM_SLOTS;
        let d = D_MODEL;
        let vecs = Tensor::from_vec(vecs_flat, (self.b, s, d), device)?;
        let mask_f32: Vec<f32> = mask_flat.iter()
            .map(|&m| if m { 0.0 } else { f32::NEG_INFINITY })
            .collect();
        let mask = Tensor::from_vec(mask_f32, (self.b, s), device)?;
        Ok((vecs.clone(), vecs, mask))
    }
}

pub struct ANTEngine {
    pub model: ANT,
    pub memory: Arc<MemorySystem>,
    pub device: Device,
    /// Persistent tag register: (B, d_model)
    pub tag_register: Option<Tensor>,
    /// CPU-resident AddrNet bank — populated once at end of Phase B,
    /// drives ALL address computation in Phase C and inference.
    pub cpu_addr_bank: Arc<CpuAddrBank>,
}

impl ANTEngine {
    pub fn new(model: ANT, mem_path: &str, device: Device) -> Self {
        let memory = Arc::new(MemorySystem::new(
            mem_path, D_MODEL, MEM_DEPTH_CAP,
            MEM_EMA_ALPHA_BASE, MEM_EMA_ALPHA_MIN, MEM_FLUSH_INTERVAL));
        let cpu_addr_bank = Arc::new(CpuAddrBank::new_zeroed());
        Self { model, memory, device, tag_register: None, cpu_addr_bank }
    }

    pub fn reset_state(&mut self, batch_size: usize) -> Result<()> {
        self.tag_register = Some(Tensor::zeros(
            (batch_size, D_MODEL), DType::F32, &self.device)?);
        Ok(())
    }

    /// Finalize Phase B: sync GPU AddrNet weights → CpuAddrBank, then drop GPU addr_nets.
    ///
    /// Must be called before Phase C or inference. After this:
    ///   - model.addr_nets is None (GPU tensors freed)
    ///   - cpu_addr_bank holds the final trained weights
    ///   - All address computation uses the CPU path
    pub fn finalize_phase_b(&mut self) -> Result<()> {
        self.model.sync_cpu_addr_bank(&self.cpu_addr_bank)?;
        self.model.drop_addr_nets();
        println!("  [Engine] Phase B finalized: AddrNet synced to CpuAddrBank, GPU tensors freed.");
        Ok(())
    }

    // ---------------------------------------------------------------------------
    // Memory I/O helpers
    // ---------------------------------------------------------------------------

    /// Convert AddrNet outputs to Vec<Vec<Vec<u8>>> for trie ops.
    fn addrs_to_bytes(addr_tensors: &[Tensor]) -> Result<Vec<Vec<Vec<u8>>>> {
        ANT::addrs_to_bytes(addr_tensors)
    }

    /// Compute batch addresses using CPU AddrNet bank.
    /// hidden: (B, d) tensor — transferred to CPU, addresses computed via CpuAddrBank.
    /// Returns [B][N_ADDR_NETS][ADDR_DEPTH] — same format as addrs_to_bytes output.
    fn compute_batch_addrs_cpu(&self, hidden: &Tensor) -> Result<Vec<Vec<Vec<u8>>>> {
        let b = hidden.dim(0)?;
        let h_cpu = hidden.to_vec2::<f32>()?;
        // per_net: [N_ADDR_NETS][B][ADDR_DEPTH]
        let per_net = self.cpu_addr_bank.forward_all_batch(&h_cpu);
        // Transpose to [B][N_ADDR_NETS][ADDR_DEPTH]
        let batch = (0..b)
            .map(|bi| per_net.iter().map(|net| net[bi].clone()).collect())
            .collect();
        Ok(batch)
    }

    /// Read from trie, return (mem_keys, mem_values, mem_mask_f32).
    /// mem_keys == mem_values (trie stores single vector per node).
    /// mem_mask_f32: 0.0 = valid, -inf = invalid (for attn masking).
    ///
    /// Address computation: GPU AddrNet during Phase B (addr_nets Some),
    /// CpuAddrBank during Phase C and inference (addr_nets None).
    pub fn read_memory(&self, hidden: &Tensor, temperature: f32, rng: &mut impl Rng)
        -> Result<(Tensor, Tensor, Tensor)>
    {
        let b = hidden.dim(0)?;
        let s = N_MEM_SLOTS;
        let d = D_MODEL;

        let batch_addrs = if self.model.addr_nets.is_some() {
            let addr_tensors = self.model.compute_addresses(hidden, temperature, false, rng)?;
            Self::addrs_to_bytes(&addr_tensors)?
        } else {
            self.compute_batch_addrs_cpu(hidden)?
        };

        let (vecs_flat, mask_flat) = self.memory.read_batch(&batch_addrs, s);

        let vecs = Tensor::from_vec(vecs_flat, (b, s, d), &self.device)?;
        let mask_f32_flat: Vec<f32> = mask_flat.iter()
            .map(|&m| if m { 0.0 } else { f32::NEG_INFINITY })
            .collect();
        let mask_f32 = Tensor::from_vec(mask_f32_flat, (b, s), &self.device)?;

        Ok((vecs.clone(), vecs, mask_f32))
    }

    /// Write hidden states (B, d) to trie at AddrNet addresses.
    ///
    /// `next_tokens`: optional (B,) u32 tensor of next-token ids.
    /// When provided, the address input is blended:
    ///   addr_input = hidden + MEM_WRITE_NEXT_TOK_ALPHA * embed(next_tok)
    /// This makes write addresses forward-looking so future reads hit more
    /// relevant trie slots. Use argmax(logits) as proxy during inference.
    pub fn write_memory(&self, hidden: &Tensor, temperature: f32,
                        next_tokens: Option<&Tensor>, rng: &mut impl Rng)
        -> Result<()>
    {
        let values = self.model.compute_value(hidden)?; // (B, d)
        let values_np = values.to_vec2::<f32>()?; // B x d

        // Skip any NaN/inf rows
        let valid: Vec<bool> = values_np.iter()
            .map(|row| row.iter().all(|v| v.is_finite()))
            .collect();
        if !valid.iter().any(|&v| v) { return Ok(()); }

        // Blend hidden with next-token embedding for forward-looking addresses
        let addr_hidden = if MEM_WRITE_NEXT_TOK_ALPHA > 0.0 {
            if let Some(next_tok) = next_tokens {
                let next_embed = self.model.embed.forward(next_tok)?; // (B, d)
                let blended = hidden.add(&(next_embed * MEM_WRITE_NEXT_TOK_ALPHA as f64)?)?;
                blended.detach()
            } else {
                hidden.detach()
            }
        } else {
            hidden.detach()
        };

        let filtered_addr_hidden = if valid.iter().all(|&v| v) {
            addr_hidden
        } else {
            let indices: Vec<usize> = valid.iter().enumerate()
                .filter_map(|(i, &v)| if v { Some(i) } else { None })
                .collect();
            let rows: Vec<Tensor> = indices.iter()
                .map(|&i| addr_hidden.get(i))
                .collect::<Result<Vec<_>>>()?;
            Tensor::stack(&rows, 0)?
        };

        let batch_addrs = if self.model.addr_nets.is_some() {
            let addr_tensors = self.model.compute_addresses(
                &filtered_addr_hidden, temperature, false, rng)?;
            Self::addrs_to_bytes(&addr_tensors)?
        } else {
            self.compute_batch_addrs_cpu(&filtered_addr_hidden)?
        };

        let val_refs: Vec<&[f32]> = if valid.iter().all(|&v| v) {
            values_np.iter().map(|row| row.as_slice()).collect()
        } else {
            valid.iter().enumerate()
                .filter_map(|(i, &v)| if v { Some(values_np[i].as_slice()) } else { None })
                .collect()
        };
        self.memory.write_batch(&batch_addrs, &val_refs);
        Ok(())
    }

    // ---------------------------------------------------------------------------
    // Async trie I/O — overlap CPU trie ops with GPU inner steps
    // ---------------------------------------------------------------------------

    /// Run pass-1 forward on `token_ids` and compute AddrNet addresses.
    ///
    /// Phase B (addr_nets Some): GPU AddrNet computes addresses.
    /// Phase C / inference (addr_nets None): CpuAddrBank computes addresses —
    ///   the hidden→CPU transfer is a tiny 128×B×4 bytes, addr MLP is 10.5K ops.
    pub fn compute_seed_addrs(&self, token_ids: &Tensor, temperature: f32,
                               rng: &mut impl Rng)
        -> Result<(Vec<Vec<Vec<u8>>>, usize)>
    {
        let b = token_ids.dim(0)?;
        let tag = self.tag_register.as_ref().map(|t| t.clone());
        let (_, _, hidden_p1) = self.model.forward(
            &token_ids.detach(), None, None, None, tag.as_ref(), None)?;
        let h_mean = hidden_p1.mean(1)?.detach();

        let batch_addrs = if self.model.addr_nets.is_some() {
            let addr_tensors = self.model.compute_addresses(&h_mean, temperature, false, rng)?;
            Self::addrs_to_bytes(&addr_tensors)?
        } else {
            self.compute_batch_addrs_cpu(&h_mean)?
        };
        Ok((batch_addrs, b))
    }

    /// Spawn a background thread to read the trie from pre-computed addresses.
    /// The thread holds an Arc clone of the MemorySystem.
    /// With RwLock on the trie, multiple concurrent reads are possible.
    pub fn spawn_prefetch_read(&self, batch_addrs: Vec<Vec<Vec<u8>>>, b: usize) -> PrefetchHandle {
        let memory = Arc::clone(&self.memory);
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let result = memory.read_batch(&batch_addrs, N_MEM_SLOTS);
            let _ = tx.send(result);
        });
        PrefetchHandle { rx, b }
    }

    /// Proactive prefetch for Phase C: transfers h_mean to CPU then dispatches
    /// BOTH address computation AND trie read to a background thread.
    ///
    /// This fully decouples address computation + trie I/O from the GPU timeline.
    /// Use instead of compute_seed_addrs + spawn_prefetch_read in Phase C.
    pub fn spawn_proactive_prefetch(&self, h_mean: &Tensor) -> Result<PrefetchHandle> {
        let b = h_mean.dim(0)?;
        let h_cpu = h_mean.to_vec2::<f32>()?;
        let bank = Arc::clone(&self.cpu_addr_bank);
        let memory = Arc::clone(&self.memory);
        let (tx, rx) = std::sync::mpsc::channel();

        std::thread::spawn(move || {
            // Address computation: pure CPU, ~1ms for 10.5K params
            let per_net = bank.forward_all_batch(&h_cpu);
            let batch_addrs: Vec<Vec<Vec<u8>>> = (0..b)
                .map(|bi| per_net.iter().map(|net| net[bi].clone()).collect())
                .collect();
            // Trie read: RwLock — concurrent with other reads
            let result = memory.read_batch(&batch_addrs, N_MEM_SLOTS);
            let _ = tx.send(result);
        });

        Ok(PrefetchHandle { rx, b })
    }

    /// Extract write data from a (B, T, d) hidden tensor.
    ///
    /// Phase B (addr_nets Some): GPU AddrNet computes addresses.
    /// Phase C / inference (addr_nets None): CpuAddrBank computes addresses.
    pub fn extract_write_data(&self, hidden: &Tensor, temperature: f32,
                               next_token_ids: Option<&Tensor>,
                               rng: &mut impl Rng)
        -> Result<Vec<(Vec<Vec<Vec<u8>>>, Vec<Vec<f32>>)>>
    {
        let (b, t, _) = hidden.dims3()?;
        let mut token_data = Vec::with_capacity(t);
        for ti in 0..t {
            let mut h_rows: Vec<Tensor> = Vec::with_capacity(b);
            for bi in 0..b {
                h_rows.push(hidden.get(bi)?.get(ti)?.unsqueeze(0)?);
            }
            let h_batch = Tensor::cat(&h_rows, 0)?.detach();

            let values = self.model.compute_value(&h_batch)?;
            let values_cpu = values.to_vec2::<f32>()?;

            // Blend with next-token embedding for forward-looking addresses
            let addr_input = if MEM_WRITE_NEXT_TOK_ALPHA > 0.0 {
                if let Some(ntids) = next_token_ids {
                    let mut row: Vec<u32> = Vec::with_capacity(b);
                    for bi in 0..b {
                        let tok = ntids.get(bi)?.get(ti.min(t - 1))?.to_scalar::<u32>()?;
                        row.push(tok.min(VOCAB_SIZE as u32 - 1));
                    }
                    let t_tensor = Tensor::from_vec(row, (b,), &self.device)?;
                    let next_embed = self.model.embed.forward(&t_tensor)?;
                    (&h_batch).add(&(next_embed * MEM_WRITE_NEXT_TOK_ALPHA as f64)?)?.detach()
                } else {
                    h_batch
                }
            } else {
                h_batch
            };

            let batch_addrs = if self.model.addr_nets.is_some() {
                let addr_tensors = self.model.compute_addresses(
                    &addr_input, temperature, false, rng)?;
                Self::addrs_to_bytes(&addr_tensors)?
            } else {
                self.compute_batch_addrs_cpu(&addr_input)?
            };

            token_data.push((batch_addrs, values_cpu));
        }
        Ok(token_data)
    }

    /// Fire-and-forget: spawn background thread to write pre-extracted token data to trie.
    pub fn spawn_async_write(&self, write_data: Vec<(Vec<Vec<Vec<u8>>>, Vec<Vec<f32>>)>) {
        let memory = Arc::clone(&self.memory);
        std::thread::spawn(move || {
            for (batch_addrs, values_cpu) in write_data {
                let val_refs: Vec<&[f32]> = values_cpu.iter().map(|v| v.as_slice()).collect();
                memory.write_batch(&batch_addrs, &val_refs);
            }
        });
    }

    // ---------------------------------------------------------------------------
    // Training encode — two-pass (full, single-step)
    // ---------------------------------------------------------------------------

    /// Pass-1 only: run a no-memory forward pass to compute h_mean, then read
    /// the trie and return cached memory vectors.
    ///
    /// Use this once per trie cycle before calling `forward_with_mem` in a loop.
    /// The returned tensors are detached — no gradients flow through them.
    pub fn read_for_encode(&self, token_ids: &Tensor, temperature: f32,
                           rng: &mut impl Rng)
        -> Result<(Tensor, Tensor, Tensor)>
    {
        let tag = self.tag_register.as_ref().map(|t| t.clone());
        let (_, _, hidden_p1) = self.model.forward(
            &token_ids.detach(), None, None, None, tag.as_ref(), None)?;
        let h_mean = hidden_p1.mean(1)?.detach();
        self.read_memory(&h_mean, temperature, rng)
    }

    /// Pass-2 only: forward with pre-loaded (detached) memory vectors.
    ///
    /// No trie I/O — use the tensors returned by `read_for_encode`.
    /// Updates `tag_register` with the last-token hidden state of this batch.
    pub fn forward_with_mem(&mut self, token_ids: &Tensor,
                            mem_k: &Tensor, mem_v: &Tensor, mem_mask: &Tensor)
        -> Result<(Tensor, Tensor, Tensor)>
    {
        let b = token_ids.dim(0)?;
        if self.tag_register.is_none() || self.tag_register.as_ref().unwrap().dim(0)? != b {
            self.reset_state(b)?;
        }
        let tag = self.tag_register.as_ref().map(|t| t.clone());
        let (logits, halt_logits, hidden) = self.model.forward(
            token_ids, Some(mem_k), Some(mem_v), Some(mem_mask),
            tag.as_ref(), None)?;

        // Update tag register: last-token hidden per batch item
        let mut last_rows: Vec<Tensor> = Vec::with_capacity(b);
        for bi in 0..b {
            let seq = hidden.get(bi)?;
            let last = seq.get(seq.dim(0)? - 1)?.unsqueeze(0)?;
            last_rows.push(last);
        }
        let last_hidden = Tensor::cat(&last_rows, 0)?;
        self.tag_register = Some(last_hidden.detach());

        Ok((logits, halt_logits, hidden))
    }

    /// Write a full (B, T, d) hidden tensor to the trie — one write per token.
    ///
    /// `next_token_ids`: optional (B, T) target ids. When provided, each token's write
    /// address is conditioned on the corresponding next token:
    ///   addr_input[t] = hidden[t] + alpha * embed(next_tok[t])
    /// Pass `target_ids` from the supervised batch here. For the final token of the
    /// sequence (no known next), the last entry of next_token_ids wraps or is zero.
    ///
    /// Use this once per trie cycle after the inner gradient loop completes.
    pub fn write_hidden(&self, hidden: &Tensor, temperature: f32,
                        next_token_ids: Option<&Tensor>,
                        rng: &mut impl Rng) -> Result<()>
    {
        let (b, t, _) = hidden.dims3()?;
        for ti in 0..t {
            let mut h_rows: Vec<Tensor> = Vec::with_capacity(b);
            for bi in 0..b {
                h_rows.push(hidden.get(bi)?.get(ti)?.unsqueeze(0)?);
            }
            let h_batch = Tensor::cat(&h_rows, 0)?.detach();

            // Extract next-token ids for this timestep if available
            let next_tok_t = if let Some(ntids) = next_token_ids {
                // ntids: (B, T) — take column ti, clamp to vocab
                let mut row: Vec<u32> = Vec::with_capacity(b);
                for bi in 0..b {
                    let tok = ntids.get(bi)?.get(ti.min(t - 1))?.to_scalar::<u32>()?;
                    row.push(tok.min(VOCAB_SIZE as u32 - 1));
                }
                let t_tensor = Tensor::from_vec(row, (b,), &self.device)?;
                Some(t_tensor)
            } else {
                None
            };

            self.write_memory(&h_batch, temperature,
                              next_tok_t.as_ref(), rng)?;
        }
        Ok(())
    }

    /// Two-pass training encode (single-step convenience wrapper).
    ///
    /// Pass 1: forward WITHOUT memory → get processed hidden states → addresses
    /// Pass 2: forward WITH memory (cross-attn) → logits, hidden for loss
    ///
    /// Returns (logits: B×T×V, halt_logits: B×T×2, hidden: B×T×d)
    pub fn encode(&mut self, token_ids: &Tensor, temperature: f32,
                  write_to_trie: bool, rng: &mut impl Rng)
        -> Result<(Tensor, Tensor, Tensor)>
    {
        let b = token_ids.dim(0)?;

        if self.tag_register.is_none() || self.tag_register.as_ref().unwrap().dim(0)? != b {
            self.reset_state(b)?;
        }

        let tag = self.tag_register.as_ref().map(|t| t.clone());

        // Pass 1: no memory
        let (_, _, hidden_p1) = {
            let no_grad_ids = token_ids.detach();
            self.model.forward(&no_grad_ids, None, None, None, tag.as_ref(), None)?
        };

        // Mean pool over sequence for address generation
        let h_mean = hidden_p1.mean(1)?; // (B, d)

        // Read memory using pass-1 hidden
        let (mem_k, mem_v, mem_mask) = self.read_memory(&h_mean.detach(), temperature, rng)?;

        // Pass 2: with memory
        let (logits, halt_logits, hidden) = self.model.forward(
            token_ids,
            Some(&mem_k), Some(&mem_v), Some(&mem_mask),
            tag.as_ref(), None)?;

        // Update tag register: last token hidden for each batch item
        let mut last_rows: Vec<Tensor> = Vec::with_capacity(b);
        for bi in 0..b {
            let seq = hidden.get(bi)?; // (T, d)
            let last = seq.get(seq.dim(0)? - 1)?; // (d,)
            last_rows.push(last.unsqueeze(0)?); // (1, d)
        }
        let last_hidden = Tensor::cat(&last_rows, 0)?; // (B, d)
        self.tag_register = Some(last_hidden.detach());

        // Write to trie — use argmax(logits[ti]) as next-token proxy for forward-looking addresses
        if write_to_trie {
            let t = hidden.dim(1)?;
            for ti in 0..t {
                let mut h_rows: Vec<Tensor> = Vec::with_capacity(b);
                for bi in 0..b {
                    let h_t = hidden.get(bi)?.get(ti)?.unsqueeze(0)?; // (1, d)
                    h_rows.push(h_t);
                }
                let h_batch = Tensor::cat(&h_rows, 0)?; // (B, d)
                // Proxy next token: argmax of logits at this timestep
                let proxy_next = logits.i((.., ti, ..))?.argmax(1)?.to_dtype(DType::U32)?;
                self.write_memory(&h_batch.detach(), temperature,
                                  Some(&proxy_next), rng)?;
            }
        }

        Ok((logits, halt_logits, hidden))
    }

    // ---------------------------------------------------------------------------
    // Inference generate
    // ---------------------------------------------------------------------------

    pub fn generate(&mut self, prompt_ids: &[u32], max_tokens: usize,
                    temperature: f32, top_k: usize, top_p: f32,
                    rng: &mut impl Rng) -> Result<Vec<u32>>
    {
        self.reset_state(1)?;
        let device = &self.device.clone();

        let prompt = Tensor::from_vec(prompt_ids.to_vec(), (1, prompt_ids.len()), device)?;

        // Prefill: two-pass
        let (_, _, h_pre) = self.model.forward(&prompt, None, None, None,
                                               self.tag_register.as_ref(), None)?;
        let h_mean = h_pre.mean(1)?;
        let (mk, mv, mm) = self.read_memory(&h_mean, temperature, rng)?;

        let mut cache = KVCache::new();
        let (logits, _, hidden) = self.model.forward(
            &prompt, Some(&mk), Some(&mv), Some(&mm),
            self.tag_register.as_ref(), Some(&mut cache))?;

        // Write prompt tokens to trie — use logits argmax as next-token proxy
        let t = hidden.dim(1)?;
        for ti in 0..t {
            let h_t = hidden.get(0)?.get(ti)?.unsqueeze(0)?;
            let proxy_next = logits.i((0, ti, ..))?.unsqueeze(0)?.argmax(1)?.to_dtype(DType::U32)?;
            self.write_memory(&h_t, temperature, Some(&proxy_next), rng)?;
        }

        let tag_last = hidden.get(0)?.get(t - 1)?.unsqueeze(0)?.detach();
        self.tag_register = Some(tag_last);

        // Sample first token
        let next_logits = logits.get(0)?.get(logits.dim(1)? - 1)?.unsqueeze(0)?;
        let mut next_token = self.sample(&next_logits, temperature, top_k, top_p, rng)?;
        let mut generated = vec![next_token];
        let mut prev_hidden = hidden.get(0)?.get(t - 1)?.unsqueeze(0)?;

        for _ in 1..max_tokens {
            if next_token == EOS_ID { break; }

            let tok = Tensor::from_vec(vec![next_token], (1, 1), device)?;
            let (mk, mv, mm) = self.read_memory(&prev_hidden, temperature, rng)?;

            let (logits, halt_logits, hidden) = self.model.forward(
                &tok, Some(&mk), Some(&mv), Some(&mm),
                self.tag_register.as_ref(), Some(&mut cache))?;

            // Write to trie — use sampled next token as proxy
            let h0 = hidden.get(0)?.get(0)?.unsqueeze(0)?;
            let next_tok_proxy = Tensor::from_vec(vec![next_token], (1,), device)?;
            self.write_memory(&h0.detach(), temperature, Some(&next_tok_proxy), rng)?;

            // Update tag register
            self.tag_register = Some(h0.detach());

            // Halt head: check if model wants more memory cycles
            let halt_prob = candle_nn::ops::softmax(&halt_logits.get(0)?.get(0)?, 0)?;
            let halt = halt_prob.get(1)?.to_scalar::<f32>()? > 0.5;

            if !halt {
                // Up to 3 extra memory fetch cycles
                for _ in 0..3 {
                    let (mk2, mv2, mm2) = self.read_memory(
                        &hidden.get(0)?.get(0)?.unsqueeze(0)?, temperature, rng)?;
                    let _ = self.model.forward(
                        &tok, Some(&mk2), Some(&mv2), Some(&mm2),
                        self.tag_register.as_ref(), None)?;
                }
            }

            prev_hidden = hidden.get(0)?.get(0)?.unsqueeze(0)?;
            let nl = logits.get(0)?.get(0)?.unsqueeze(0)?;
            next_token = self.sample(&nl, temperature, top_k, top_p, rng)?;

            if next_token == NOOP_ID { continue; }
            generated.push(next_token);
        }

        Ok(generated)
    }

    fn sample(&self, logits: &Tensor, temperature: f32, top_k: usize, top_p: f32,
              rng: &mut impl Rng) -> Result<u32>
    {
        let mut logits = logits.squeeze(0)?.to_vec1::<f32>()?;

        // Check for NaN/inf
        if logits.iter().any(|v| !v.is_finite()) {
            return Ok(UNK_ID);
        }

        if temperature <= 0.0 {
            return Ok(logits.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap_or(UNK_ID));
        }

        // Temperature scaling
        for v in logits.iter_mut() { *v /= temperature; }

        // Top-k
        if top_k > 0 && top_k < logits.len() {
            let mut sorted = logits.clone();
            sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
            let threshold = sorted[top_k - 1];
            for v in logits.iter_mut() {
                if *v < threshold { *v = f32::NEG_INFINITY; }
            }
        }

        // Softmax
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = logits.iter().map(|&v| (v - max).exp()).collect();
        let sum: f32 = exp.iter().sum();
        let mut probs: Vec<f32> = exp.iter().map(|&v| v / sum).collect();

        // Top-p nucleus
        if top_p < 1.0 {
            let mut pairs: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
            pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let mut cum = 0.0;
            for (_, p) in &pairs {
                cum += p;
                if cum >= top_p { break; }
            }
            let threshold = pairs.iter().scan(0.0f32, |acc, (_, p)| {
                *acc += p; Some(*acc)
            }).zip(pairs.iter())
              .find(|(c, _)| *c >= top_p)
              .map(|(_, (_, p))| *p)
              .unwrap_or(0.0);
            for (i, p) in probs.iter_mut().enumerate() {
                if pairs.iter().position(|(pi, _)| *pi == i)
                       .map(|pos| {
                           let cum: f32 = pairs[..pos].iter().map(|(_, p)| p).sum();
                           cum >= top_p
                       })
                       .unwrap_or(false)
                {
                    *p = 0.0;
                }
            }
            let _ = threshold;
            let sum: f32 = probs.iter().sum();
            if sum > 0.0 { for p in probs.iter_mut() { *p /= sum; } }
        }

        // Sample
        let r: f32 = rng.gen();
        let mut cum = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            cum += p;
            if r <= cum { return Ok(i as u32); }
        }
        Ok((probs.len() - 1) as u32)
    }

    pub fn flush(&self) { self.memory.flush(); }

    pub fn reset_memory(&self) { self.memory.reset(); }

    pub fn memory_stats(&self) -> (usize, u64) {
        (self.memory.total_nodes(), self.memory.total_entries())
    }
}
