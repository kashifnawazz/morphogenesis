//! Where the model gets its numbers from.
//!
//! The model doesn't care whether weights come from the original safetensors
//! file, from our repacked `.morph` file, from a memory map, or from RAM. It
//! just asks for them by name. That's what this trait is for -- and it's what
//! lets Phase 6 swap the storage underneath and measure the difference.

use std::io;
use std::sync::Arc;

/// Anything that can hand the model weights by name.
///
/// `Arc<[f32]>` rather than `Vec<f32>` so a cache can hand out the same
/// numbers to many callers without copying them.
pub trait Weights: Send + Sync {
    /// A whole tensor.
    fn get(&self, name: &str) -> io::Result<Arc<[f32]>>;

    /// One row of a 2-D tensor. Used for the embedding lookup, where we want
    /// 2 KB out of a 622 MB table.
    fn get_row(&self, name: &str, row: usize) -> io::Result<Vec<f32>>;

    /// How many bytes have actually been pulled off the disk so far.
    ///
    /// This is THE number this project is about. Everything else is detail.
    fn bytes_read(&self) -> u64;

    /// Zero the counter, so you can measure one forward pass at a time.
    fn reset_bytes(&self);

    /// Short human-readable description, for benchmark output.
    fn describe(&self) -> String;

    /// Hint that this layer will be wanted shortly, so the read can start now
    /// instead of when we block on it.
    ///
    /// The default does nothing -- sources that are already in RAM have
    /// nothing to prefetch.
    fn prefetch_layer(&self, _layer: usize) {}
}
