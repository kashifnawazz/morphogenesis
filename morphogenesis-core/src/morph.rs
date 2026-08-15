//! Our own weight file: the same numbers, laid out in the order we read them.
//!
//! # Why bother
//!
//! In `model.safetensors` the tensors are ordered ALPHABETICALLY, because
//! that's what the exporter happened to do. So the layers appear as
//!
//! ```text
//!     0, 1, 10, 11, 12, ... 19, 2, 20, 21, ... 27, 3, 4, ...
//! ```
//!
//! and inside a layer, `down_proj` comes before `gate_proj` before `q_proj`.
//! But we USE them in a completely different order: layer 0, then 1, then 2,
//! and within a layer q, k, v, o, gate, up, down.
//!
//! On an SSD that mismatch costs almost nothing. On a spinning disk every
//! mismatch is a ~10 ms head movement, and there are roughly 300 of them per
//! token -- about 3 seconds of pure waiting, on top of the transfer.
//!
//! So this module rewrites the file with everything in the order we actually
//! want it, layers contiguous and 4 KB aligned. Reading then becomes one long
//! sweep instead of a scatter.
//!
//! It also drops the duplicated embedding table, which safetensors stores
//! twice (311 MB, 21% of the file) despite the config saying they're tied.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Error, ErrorKind, Read, Write};
use std::os::unix::fs::FileExt;
use std::os::unix::io::AsRawFd;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use memmap2::Mmap;
use serde::{Deserialize, Serialize};

use crate::config::Config;
use crate::safetensors::SafeTensors;
use crate::weights::Weights;

const MAGIC: &[u8; 8] = b"MORPHGEN";
const VERSION: u32 = 1;
const ALIGN: u64 = 4096; // page size, so O_DIRECT stays possible later
/// Room reserved for the JSON table of contents. 311 entries need ~40 KB.
const HEADER_REGION: u64 = 64 * 1024;

/// The order the model actually touches a layer's weights.
/// Compare with the alphabetical order safetensors uses.
pub const LAYER_PARTS: [&str; 11] = [
    "input_layernorm",
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.q_norm",
    "self_attn.k_norm",
    "self_attn.o_proj",
    "post_attention_layernorm",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
];

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Entry {
    pub offset: u64,
    pub nbytes: u64,
    pub shape: Vec<usize>,
    pub dtype: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MorphHeader {
    pub n_layers: usize,
    /// Distance in bytes from one layer's start to the next. Constant, so
    /// layer n lives at `layer0_offset + n * layer_stride` with no lookup.
    pub layer_stride: u64,
    pub layer0_offset: u64,
    pub tensors: HashMap<String, Entry>,
    /// True when lm_head and embed_tokens are the same matrix and we stored
    /// it only once.
    pub embeddings_tied: bool,
}

// ---------------------------------------------------------------------------
// Writing
// ---------------------------------------------------------------------------

/// Rewrite a safetensors file into execution order.
pub fn repack(src: &SafeTensors, cfg: &Config, out_path: &Path) -> std::io::Result<MorphHeader> {
    // Are lm_head and embed_tokens really identical? The config claims they're
    // tied, but claims aren't evidence -- check the bytes.
    let tied = if cfg.tie_word_embeddings {
        let a = src.get_raw("lm_head.weight")?;
        let b = src.get_raw("model.embed_tokens.weight")?;
        a == b
    } else {
        false
    };

    // ---- work out where everything goes, before writing anything ----------
    let mut tensors: HashMap<String, Entry> = HashMap::new();
    let mut cursor = HEADER_REGION; // leave room for the table of contents

    let layer0_offset = cursor;
    let mut layer_stride = 0u64;

    for layer in 0..cfg.num_hidden_layers {
        let layer_start = cursor;
        for part in LAYER_PARTS {
            let name = format!("model.layers.{layer}.{part}.weight");
            let info = src
                .info(&name)
                .ok_or_else(|| Error::new(ErrorKind::NotFound, name.clone()))?;
            tensors.insert(
                name,
                Entry {
                    offset: cursor,
                    nbytes: info.nbytes(),
                    shape: info.shape.clone(),
                    dtype: info.dtype.clone(),
                },
            );
            cursor += info.nbytes();
        }
        cursor = cursor.div_ceil(ALIGN) * ALIGN; // pad to a page boundary
        if layer == 0 {
            layer_stride = cursor - layer_start;
        }
    }

    // Tail tensors: the final norm, then the embedding table.
    for name in ["model.norm.weight", "model.embed_tokens.weight"] {
        let info = src
            .info(name)
            .ok_or_else(|| Error::new(ErrorKind::NotFound, name.to_string()))?;
        tensors.insert(
            name.to_string(),
            Entry {
                offset: cursor,
                nbytes: info.nbytes(),
                shape: info.shape.clone(),
                dtype: info.dtype.clone(),
            },
        );
        cursor += info.nbytes();
        cursor = cursor.div_ceil(ALIGN) * ALIGN;
    }

    if tied {
        // Point lm_head at the very same bytes. One matrix, two names.
        let e = tensors["model.embed_tokens.weight"].clone();
        tensors.insert("lm_head.weight".to_string(), e);
    } else {
        let info = src
            .info("lm_head.weight")
            .ok_or_else(|| Error::new(ErrorKind::NotFound, "lm_head.weight".to_string()))?;
        tensors.insert(
            "lm_head.weight".to_string(),
            Entry {
                offset: cursor,
                nbytes: info.nbytes(),
                shape: info.shape.clone(),
                dtype: info.dtype.clone(),
            },
        );
    }

    let header = MorphHeader {
        n_layers: cfg.num_hidden_layers,
        layer_stride,
        layer0_offset,
        tensors,
        embeddings_tied: tied,
    };

    // ---- now write it out --------------------------------------------------
    let file = File::create(out_path)?;
    let mut w = BufWriter::with_capacity(1 << 22, file);

    let header_json = serde_json::to_vec(&header).map_err(std::io::Error::other)?;
    if header_json.len() as u64 + 16 > HEADER_REGION {
        return Err(Error::new(
            ErrorKind::InvalidData,
            format!("header is {} bytes, region is {HEADER_REGION}", header_json.len()),
        ));
    }
    w.write_all(MAGIC)?;
    w.write_all(&VERSION.to_le_bytes())?;
    w.write_all(&(header_json.len() as u32).to_le_bytes())?;
    w.write_all(&header_json)?;
    w.write_all(&vec![0u8; (HEADER_REGION as usize) - 16 - header_json.len()])?;

    let mut written = HEADER_REGION;
    let write_tensor = |w: &mut BufWriter<File>, name: &str, written: &mut u64| -> std::io::Result<()> {
        let entry = &header.tensors[name];
        if *written < entry.offset {
            w.write_all(&vec![0u8; (entry.offset - *written) as usize])?;
            *written = entry.offset;
        }
        let raw = src.get_raw(name)?;
        w.write_all(&raw)?;
        *written += raw.len() as u64;
        Ok(())
    };

    for layer in 0..cfg.num_hidden_layers {
        for part in LAYER_PARTS {
            let name = format!("model.layers.{layer}.{part}.weight");
            write_tensor(&mut w, &name, &mut written)?;
        }
    }
    write_tensor(&mut w, "model.norm.weight", &mut written)?;
    write_tensor(&mut w, "model.embed_tokens.weight", &mut written)?;
    if !tied {
        write_tensor(&mut w, "lm_head.weight", &mut written)?;
    }

    w.flush()?;
    Ok(header)
}

// ---------------------------------------------------------------------------
// Reading
// ---------------------------------------------------------------------------

/// How the reader gets at the bytes.
pub enum Backend {
    /// Explicit reads at an offset. Every byte crosses a syscall, and we can
    /// count them exactly.
    Pread(File),
    /// The file is mapped into memory; the kernel pages it in on demand.
    /// Faster when the data is already cached, and the byte counter becomes
    /// an estimate rather than a measurement.
    Mmap(Mmap, File),
    /// Reads that bypass the kernel's page cache entirely.
    ///
    /// Every read must be 4 KB aligned in offset, length AND buffer address,
    /// so we widen each request to page boundaries and slice the result. The
    /// point isn't raw speed -- it's that nothing we read pollutes the page
    /// cache, leaving that RAM for the layers we deliberately pinned.
    Direct(File),
}

pub struct MorphFile {
    backend: Backend,
    header: MorphHeader,
    bytes_read: AtomicU64,
}

/// How to get bytes off the disk.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    Pread,
    Mmap,
    Direct,
}

impl MorphFile {
    pub fn open(path: &Path, mode: Mode) -> std::io::Result<Self> {
        let mut file = File::open(path)?;

        let mut magic = [0u8; 8];
        file.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(Error::new(ErrorKind::InvalidData, "not a .morph file"));
        }
        let mut buf4 = [0u8; 4];
        file.read_exact(&mut buf4)?;
        if u32::from_le_bytes(buf4) != VERSION {
            return Err(Error::new(ErrorKind::InvalidData, "unsupported version"));
        }
        file.read_exact(&mut buf4)?;
        let header_len = u32::from_le_bytes(buf4) as usize;

        let mut header_json = vec![0u8; header_len];
        file.read_exact(&mut header_json)?;
        let header: MorphHeader =
            serde_json::from_slice(&header_json).map_err(std::io::Error::other)?;

        let backend = match mode {
            Mode::Mmap => {
                // SAFETY: we only read, and the file isn't modified while open.
                let mmap = unsafe { Mmap::map(&file)? };
                Backend::Mmap(mmap, file)
            }
            Mode::Pread => Backend::Pread(file),
            Mode::Direct => {
                use std::os::unix::fs::OpenOptionsExt;
                let direct = std::fs::OpenOptions::new()
                    .read(true)
                    .custom_flags(libc::O_DIRECT)
                    .open(path)?;
                Backend::Direct(direct)
            }
        };

        Ok(MorphFile {
            backend,
            header,
            bytes_read: AtomicU64::new(0),
        })
    }

    pub fn header(&self) -> &MorphHeader {
        &self.header
    }

    fn read_at(&self, offset: u64, len: usize) -> std::io::Result<Vec<u8>> {
        self.bytes_read.fetch_add(len as u64, Ordering::Relaxed);
        match &self.backend {
            Backend::Pread(f) => {
                let mut buf = vec![0u8; len];
                f.read_exact_at(&mut buf, offset)?;
                Ok(buf)
            }
            Backend::Mmap(m, _) => {
                let start = offset as usize;
                Ok(m[start..start + len].to_vec())
            }
            Backend::Direct(f) => {
                // Widen the request out to page boundaries in both directions.
                let lo = offset & !(ALIGN - 1);
                let hi = (offset + len as u64).div_ceil(ALIGN) * ALIGN;
                let mut buf = AlignedBuf::new((hi - lo) as usize);
                f.read_exact_at(buf.as_mut_slice(), lo)?;
                let skip = (offset - lo) as usize;
                Ok(buf.as_mut_slice()[skip..skip + len].to_vec())
            }
        }
    }

    /// Tell the kernel we'll want this layer soon, so it can start fetching
    /// while we're still busy with the previous one.
    ///
    /// This is prefetching without a background thread: `POSIX_FADV_WILLNEED`
    /// queues an asynchronous readahead and returns immediately. By the time
    /// we actually ask for those bytes the transfer is already in flight, so
    /// disk time overlaps compute time instead of following it.
    ///
    /// Does nothing under O_DIRECT, which deliberately bypasses the cache
    /// this would populate.
    pub fn prefetch_layer(&self, layer: usize) {
        if layer >= self.header.n_layers {
            return;
        }
        let fd = match &self.backend {
            Backend::Pread(f) => f.as_raw_fd(),
            Backend::Mmap(_, f) => f.as_raw_fd(),
            Backend::Direct(_) => return,
        };
        let offset = self.header.layer0_offset + layer as u64 * self.header.layer_stride;
        unsafe {
            libc::posix_fadvise(
                fd,
                offset as libc::off_t,
                self.header.layer_stride as libc::off_t,
                libc::POSIX_FADV_WILLNEED,
            );
        }
    }

    fn entry(&self, name: &str) -> std::io::Result<&Entry> {
        self.header
            .tensors
            .get(name)
            .ok_or_else(|| Error::new(ErrorKind::NotFound, format!("no tensor named {name}")))
    }

    /// Ask the kernel to forget this file's cached pages.
    ///
    /// Without this, a second benchmark run reads from RAM and reports a
    /// number that has nothing to do with the disk. Normally you'd need root
    /// to drop the whole page cache; `posix_fadvise(DONTNEED)` drops just this
    /// file's pages and needs no privileges.
    pub fn drop_page_cache(&self) -> std::io::Result<()> {
        let fd = match &self.backend {
            Backend::Pread(f) => f.as_raw_fd(),
            Backend::Mmap(_, f) => f.as_raw_fd(),
            Backend::Direct(f) => f.as_raw_fd(),
        };
        let rc = unsafe { libc::posix_fadvise(fd, 0, 0, libc::POSIX_FADV_DONTNEED) };
        if rc != 0 {
            return Err(Error::from_raw_os_error(rc));
        }
        Ok(())
    }
}

impl Weights for MorphFile {
    fn get(&self, name: &str) -> std::io::Result<Arc<[f32]>> {
        let e = self.entry(name)?;
        let raw = self.read_at(e.offset, e.nbytes as usize)?;
        Ok(decode(&raw, &e.dtype, name)?.into())
    }

    fn get_row(&self, name: &str, row: usize) -> std::io::Result<Vec<f32>> {
        let e = self.entry(name)?;
        let row_len = *e.shape.last().unwrap();
        let item = if e.dtype == "F32" { 4u64 } else { 2u64 };
        let row_bytes = row_len as u64 * item;
        let raw = self.read_at(e.offset + row as u64 * row_bytes, row_bytes as usize)?;
        decode(&raw, &e.dtype, name)
    }

    fn bytes_read(&self) -> u64 {
        self.bytes_read.load(Ordering::Relaxed)
    }

    fn reset_bytes(&self) {
        self.bytes_read.store(0, Ordering::Relaxed);
    }

    fn prefetch_layer(&self, layer: usize) {
        MorphFile::prefetch_layer(self, layer);
    }

    fn describe(&self) -> String {
        let how = match self.backend {
            Backend::Pread(_) => "pread",
            Backend::Mmap(..) => "mmap",
            Backend::Direct(_) => "O_DIRECT",
        };
        format!(
            "morph/{how} (execution order, {} layers, stride {} MB, tied={})",
            self.header.n_layers,
            self.header.layer_stride / (1 << 20),
            self.header.embeddings_tied
        )
    }
}

fn decode(raw: &[u8], dtype: &str, name: &str) -> std::io::Result<Vec<f32>> {
    match dtype {
        "BF16" => Ok(raw
            .chunks_exact(2)
            .map(|p| f32::from_bits((u16::from_le_bytes([p[0], p[1]]) as u32) << 16))
            .collect()),
        "F32" => Ok(raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()),
        other => Err(Error::new(
            ErrorKind::InvalidData,
            format!("{name}: unsupported dtype {other}"),
        )),
    }
}


/// A byte buffer whose start address is page aligned.
///
/// O_DIRECT hands bytes straight from the device to your memory, with no
/// kernel copy in between -- so the destination has to satisfy the hardware's
/// alignment rules, not just the offset and length.
struct AlignedBuf {
    ptr: *mut u8,
    len: usize,
    layout: std::alloc::Layout,
}

impl AlignedBuf {
    fn new(len: usize) -> Self {
        let layout = std::alloc::Layout::from_size_align(len, ALIGN as usize).unwrap();
        // SAFETY: len > 0 and ALIGN is a valid power-of-two alignment.
        let ptr = unsafe { std::alloc::alloc(layout) };
        assert!(!ptr.is_null(), "aligned allocation failed");
        AlignedBuf { ptr, len, layout }
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: ptr is a valid allocation of exactly len bytes.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        // SAFETY: allocated by alloc() with this exact layout.
        unsafe { std::alloc::dealloc(self.ptr, self.layout) }
    }
}
