//! Keeping some layers in RAM instead of going to disk for them.
//!
//! # Why pinning, and not LRU
//!
//! The obvious cache policy is "least recently used". Here it would be the
//! WORST possible choice.
//!
//! Our access pattern is a strict sequential sweep: layer 0, 1, 2, ... 27,
//! then back to 0 for the next token. If the cache is smaller than the whole
//! model, LRU evicts layer 0 to make room for layer 27 -- and layer 0 is
//! precisely what we need next. Every single lookup misses. That's the classic
//! sequential-scan pathology.
//!
//! Pinning a fixed prefix instead gives a guaranteed hit rate: pin the first N
//! layers and those N always hit, no matter how big the model is.
//!
//! (The kimi-k3 reference uses an LRU, but its access pattern is different --
//! it picks 16 experts out of 896 per layer, so there's genuine reuse to
//! exploit. Ours is a dense model with none.)

use std::collections::HashMap;
use std::io;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::config::Config;
use crate::morph::LAYER_PARTS;
use crate::weights::Weights;

pub struct Pinned<W: Weights> {
    inner: W,
    resident: HashMap<String, Arc<[f32]>>,
    resident_bytes: u64,
    layers_pinned: usize,
    hits: AtomicU64,
    misses: AtomicU64,
}

impl<W: Weights> Pinned<W> {
    /// Preload whole layers, starting at layer 0, until `budget_bytes` is
    /// used up. A budget of 0 pins nothing.
    ///
    /// Note the budget is counted in f32 (4 bytes per number), because that's
    /// what we hold in RAM -- twice the bf16 size on disk.
    pub fn new(inner: W, cfg: &Config, budget_bytes: u64) -> io::Result<Self> {
        let mut resident = HashMap::new();
        let mut used = 0u64;
        let mut layers_pinned = 0;

        'outer: for layer in 0..cfg.num_hidden_layers {
            // Work out the cost of this whole layer before committing, so we
            // never pin half of one.
            let mut batch = Vec::new();
            let mut layer_bytes = 0u64;
            for part in LAYER_PARTS {
                let name = format!("model.layers.{layer}.{part}.weight");
                let data = inner.get(&name)?;
                layer_bytes += (data.len() * 4) as u64;
                batch.push((name, data));
            }
            if used + layer_bytes > budget_bytes {
                break 'outer;
            }
            for (name, data) in batch {
                resident.insert(name, data);
            }
            used += layer_bytes;
            layers_pinned += 1;
        }

        inner.reset_bytes(); // preloading isn't part of any measurement

        Ok(Pinned {
            inner,
            resident,
            resident_bytes: used,
            layers_pinned,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        })
    }

    pub fn layers_pinned(&self) -> usize {
        self.layers_pinned
    }

    pub fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    pub fn hit_rate(&self) -> f64 {
        let h = self.hits.load(Ordering::Relaxed) as f64;
        let m = self.misses.load(Ordering::Relaxed) as f64;
        if h + m == 0.0 { 0.0 } else { h / (h + m) }
    }

    pub fn inner(&self) -> &W {
        &self.inner
    }
}

impl<W: Weights> Weights for Pinned<W> {
    fn get(&self, name: &str) -> io::Result<Arc<[f32]>> {
        if let Some(hit) = self.resident.get(name) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            return Ok(Arc::clone(hit));
        }
        self.misses.fetch_add(1, Ordering::Relaxed);
        self.inner.get(name)
    }

    fn get_row(&self, name: &str, row: usize) -> io::Result<Vec<f32>> {
        self.inner.get_row(name, row)
    }

    fn bytes_read(&self) -> u64 {
        self.inner.bytes_read()
    }

    fn reset_bytes(&self) {
        self.inner.reset_bytes();
    }

    fn prefetch_layer(&self, layer: usize) {
        // Pinned layers are already in RAM; only hint for the ones that aren't.
        if layer >= self.layers_pinned {
            self.inner.prefetch_layer(layer);
        }
    }

    fn describe(&self) -> String {
        format!(
            "{} + {} layers pinned ({} MB)",
            self.inner.describe(),
            self.layers_pinned,
            self.resident_bytes / (1 << 20)
        )
    }
}
