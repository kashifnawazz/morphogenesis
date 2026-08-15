//! The model: token ids in, a score for every possible next word out.
//!
//! `weights.rs` says where numbers come from, `kernels.rs` has the five basic
//! operations, and this file calls them in the right order, 28 times.
//!
//! Since Phase 5 it carries a KV cache, so generating the 50th word costs the
//! same as generating the 2nd instead of redoing the whole sentence.

use std::io;
use std::sync::Arc;

use crate::config::Config;
use crate::kernels::{matvec, rmsnorm, rope, silu, softmax};
use crate::weights::Weights;

/// Remembers the Key and Value vectors of every word seen so far.
///
/// Without this, writing 50 words means running the model over the whole
/// sentence 50 times -- work grows with the square of the length. With it,
/// each new word only computes its own Key and Value and reuses the rest.
pub struct KvCache {
    k: Vec<Vec<f32>>, // one entry per layer: max_len * kv_dim
    v: Vec<Vec<f32>>,
    pub len: usize,
    pub max_len: usize,
}

impl KvCache {
    pub fn new(cfg: &Config, max_len: usize) -> Self {
        let kv_dim = cfg.kv_dim();
        KvCache {
            k: vec![vec![0.0; max_len * kv_dim]; cfg.num_hidden_layers],
            v: vec![vec![0.0; max_len * kv_dim]; cfg.num_hidden_layers],
            len: 0,
            max_len,
        }
    }

    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Roughly how much RAM this is using.
    pub fn bytes(&self) -> u64 {
        (self.k.len() * self.k[0].len() * 4 * 2) as u64
    }
}

pub struct Model {
    pub cfg: Config,
    weights: Arc<dyn Weights>,
    /// Whether to hint the next layer's read before computing this one.
    /// Only exists so the benchmark can measure whether it actually helps.
    prefetch: bool,
}

impl Model {
    pub fn new(cfg: Config, weights: Arc<dyn Weights>) -> Self {
        Model { cfg, weights, prefetch: true }
    }

    pub fn with_prefetch(mut self, on: bool) -> Self {
        self.prefetch = on;
        self
    }

    pub fn weights(&self) -> &Arc<dyn Weights> {
        &self.weights
    }

    fn w(&self, layer: usize, part: &str) -> io::Result<Arc<[f32]>> {
        self.weights
            .get(&format!("model.layers.{layer}.{part}.weight"))
    }

    /// Push `tokens` through the model, appending to the cache.
    ///
    /// Works for both cases: a whole prompt at once (the cache starts empty),
    /// or a single new word (the cache already holds everything before it).
    ///
    /// Returns the scores for the position after the last token.
    pub fn forward(&self, tokens: &[u32], cache: &mut KvCache) -> io::Result<Vec<f32>> {
        let cfg = &self.cfg;
        let n = tokens.len();
        let start = cache.len;
        assert!(start + n <= cache.max_len, "KV cache is full");

        let hidden = cfg.hidden_size;
        let head_dim = cfg.head_dim;
        let n_heads = cfg.num_attention_heads;
        let n_kv = cfg.num_key_value_heads;
        let q_dim = cfg.q_dim();
        let kv_dim = cfg.kv_dim();
        let group = cfg.heads_per_kv();

        // ---- 1. token id -> 1024 numbers (a lookup, not maths) -------------
        let mut x = vec![0f32; n * hidden];
        for (t, &tok) in tokens.iter().enumerate() {
            let row = self
                .weights
                .get_row("model.embed_tokens.weight", tok as usize)?;
            x[t * hidden..(t + 1) * hidden].copy_from_slice(&row);
        }

        // ---- 2. twenty-eight layers ----------------------------------------
        for layer in 0..cfg.num_hidden_layers {
            // Start the NEXT layer's read now, so the disk works while we do.
            // Without this the two are strictly serial: read, compute, read,
            // compute. With it they overlap.
            if self.prefetch {
                self.weights.prefetch_layer(layer + 1);
            }

            // ===== attention: let the words look at each other ==============
            let in_norm = self.w(layer, "input_layernorm")?;
            let wq = self.w(layer, "self_attn.q_proj")?;
            let wk = self.w(layer, "self_attn.k_proj")?;
            let wv = self.w(layer, "self_attn.v_proj")?;
            let q_norm = self.w(layer, "self_attn.q_norm")?;
            let k_norm = self.w(layer, "self_attn.k_norm")?;

            let mut q_all = vec![0f32; n * q_dim];

            for t in 0..n {
                let pos = start + t;
                let h = rmsnorm(
                    &x[t * hidden..(t + 1) * hidden],
                    &in_norm,
                    cfg.rms_norm_eps,
                );

                let mut q = matvec(&wq, &h, q_dim, hidden);
                let mut k = matvec(&wk, &h, kv_dim, hidden);
                let v = matvec(&wv, &h, kv_dim, hidden);

                // Qwen3 extra: rescale each head on its own. Llama has no
                // equivalent; leave it out and the model quietly degrades.
                for head in 0..n_heads {
                    let s = &mut q[head * head_dim..(head + 1) * head_dim];
                    s.copy_from_slice(&rmsnorm(s, &q_norm, cfg.rms_norm_eps));
                }
                for head in 0..n_kv {
                    let s = &mut k[head * head_dim..(head + 1) * head_dim];
                    s.copy_from_slice(&rmsnorm(s, &k_norm, cfg.rms_norm_eps));
                }

                // Stamp the position on. Q and K only, never V.
                for head in 0..n_heads {
                    rope(&mut q[head * head_dim..][..head_dim], pos, cfg.rope_theta);
                }
                for head in 0..n_kv {
                    rope(&mut k[head * head_dim..][..head_dim], pos, cfg.rope_theta);
                }

                q_all[t * q_dim..(t + 1) * q_dim].copy_from_slice(&q);
                cache.k[layer][pos * kv_dim..(pos + 1) * kv_dim].copy_from_slice(&k);
                cache.v[layer][pos * kv_dim..(pos + 1) * kv_dim].copy_from_slice(&v);
            }

            let scale = 1.0 / (head_dim as f32).sqrt();
            let mut attn = vec![0f32; n * q_dim];

            for t in 0..n {
                let pos = start + t;
                for head in 0..n_heads {
                    let kv_head = head / group; // 2 query heads share each kv head
                    let qh = &q_all[t * q_dim + head * head_dim..][..head_dim];

                    // Score against every word up to and including this one --
                    // never later ones, because at writing time the future
                    // doesn't exist.
                    let mut scores = vec![0f32; pos + 1];
                    for (s, score) in scores.iter_mut().enumerate() {
                        let kh = &cache.k[layer][s * kv_dim + kv_head * head_dim..][..head_dim];
                        *score = qh.iter().zip(kh).map(|(a, b)| a * b).sum::<f32>() * scale;
                    }
                    softmax(&mut scores);

                    let out = &mut attn[t * q_dim + head * head_dim..][..head_dim];
                    for (s, &weight) in scores.iter().enumerate() {
                        let vh = &cache.v[layer][s * kv_dim + kv_head * head_dim..][..head_dim];
                        for (o, &vi) in out.iter_mut().zip(vh) {
                            *o += weight * vi;
                        }
                    }
                }
            }

            let wo = self.w(layer, "self_attn.o_proj")?;
            for t in 0..n {
                let mixed = matvec(&wo, &attn[t * q_dim..(t + 1) * q_dim], hidden, q_dim);
                for (xi, m) in x[t * hidden..(t + 1) * hidden].iter_mut().zip(&mixed) {
                    *xi += m; // residual: add, never replace
                }
            }

            // ===== MLP: each word thinks on its own =========================
            let post_norm = self.w(layer, "post_attention_layernorm")?;
            let w_gate = self.w(layer, "mlp.gate_proj")?;
            let w_up = self.w(layer, "mlp.up_proj")?;
            let w_down = self.w(layer, "mlp.down_proj")?;
            let inter = cfg.intermediate_size;

            for t in 0..n {
                let h = rmsnorm(
                    &x[t * hidden..(t + 1) * hidden],
                    &post_norm,
                    cfg.rms_norm_eps,
                );
                let gate = matvec(&w_gate, &h, inter, hidden);
                let up = matvec(&w_up, &h, inter, hidden);
                let mixed: Vec<f32> = gate.iter().zip(&up).map(|(g, u)| silu(*g) * u).collect();
                let out = matvec(&w_down, &mixed, hidden, inter);
                for (xi, o) in x[t * hidden..(t + 1) * hidden].iter_mut().zip(&out) {
                    *xi += o;
                }
            }
        }

        cache.len += n;

        // ---- 3. score every possible next word -----------------------------
        // Only the last position matters -- that's what we predict from.
        let final_norm = self.weights.get("model.norm.weight")?;
        let last = rmsnorm(
            &x[(n - 1) * hidden..n * hidden],
            &final_norm,
            cfg.rms_norm_eps,
        );
        let head = self.weights.get("lm_head.weight")?;
        Ok(matvec(&head, &last, cfg.vocab_size, hidden))
    }

    /// Keep predicting words until we hit `max_new` or an end-of-text token.
    ///
    /// `on_token` is called as each word is produced, so a caller can print
    /// them as they arrive rather than waiting for the whole answer.
    pub fn generate(
        &self,
        prompt: &[u32],
        max_new: usize,
        eos: &[u32],
        mut on_token: impl FnMut(u32),
    ) -> io::Result<Vec<u32>> {
        let mut cache = KvCache::new(&self.cfg, prompt.len() + max_new + 1);

        // Prefill: the whole prompt in one pass.
        let mut logits = self.forward(prompt, &mut cache)?;

        let mut out = Vec::new();
        for _ in 0..max_new {
            let next = argmax(&logits) as u32;
            if eos.contains(&next) {
                break;
            }
            out.push(next);
            on_token(next);
            // Decode: just ONE token now, reusing everything in the cache.
            logits = self.forward(&[next], &mut cache)?;
        }
        Ok(out)
    }
}

/// The id of the highest-scoring word.
pub fn argmax(logits: &[f32]) -> usize {
    let mut best = 0;
    for (i, v) in logits.iter().enumerate() {
        if v > &logits[best] {
            best = i;
        }
    }
    best
}

/// The top `k` word ids, best first.
pub fn top_k(logits: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut pairs: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    pairs.sort_by(|a, b| b.1.total_cmp(&a.1));
    pairs.truncate(k);
    pairs
}
