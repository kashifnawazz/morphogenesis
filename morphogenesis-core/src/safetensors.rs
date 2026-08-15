use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Error, ErrorKind, Read};
use std::os::unix::fs::FileExt;
use std::path::Path;

/// One entry from the table of contents at the front of the file.
#[derive(Debug, Deserialize)]
pub struct TensorInfo {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: (u64, u64),
}

impl TensorInfo {
    /// How many numbers are in this tensor: 2048 x 1024 = 2,097,152.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// How many bytes it occupies on disk.
    pub fn nbytes(&self) -> u64 {
        self.data_offsets.1 - self.data_offsets.0
    }
}

/// An open safetensors file you can pull tensors out of by name.
pub struct SafeTensors {
    file: File,
    tensors: HashMap<String, TensorInfo>,
    data_start: u64,
}

impl SafeTensors {
    pub fn open(path: &Path) -> std::io::Result<Self> {
        let mut file = File::open(path)?;

        // 1. First 8 bytes = how long the table of contents is.
        let mut len_bytes = [0u8; 8];
        file.read_exact(&mut len_bytes)?;
        let header_len = u64::from_le_bytes(len_bytes);

        // 2. Read that many bytes. The cursor is already at byte 8.
        let mut header = vec![0u8; header_len as usize];
        file.read_exact(&mut header)?;

        // 3. Turn the JSON text into a searchable map.
        let tensors = parse_header(&header).map_err(|e| Error::new(ErrorKind::InvalidData, e))?;

        // 4. Numbers start right after the header.
        Ok(SafeTensors {
            file,
            tensors,
            data_start: 8 + header_len,
        })
    }

    pub fn data_start(&self) -> u64 {
        self.data_start
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    pub fn info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    /// Every tensor name in the file, sorted.
    pub fn names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.tensors.keys().map(|s| s.as_str()).collect();
        names.sort_unstable();
        names
    }

    /// Read one tensor off the disk and hand back its numbers as f32.
    ///
    /// Takes `&self`, not `&mut self`, because `read_exact_at` reads at an
    /// absolute offset without moving the file cursor. That means several
    /// threads can pull different tensors at once, which is what Phase 6 needs.
    pub fn get(&self, name: &str) -> std::io::Result<Vec<f32>> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::new(ErrorKind::NotFound, format!("no tensor named {name}")))?;

        // Offsets in the header are relative to where the DATA starts,
        // not to the start of the file. Hence the + self.data_start.
        let mut raw = vec![0u8; info.nbytes() as usize];
        self.file
            .read_exact_at(&mut raw, self.data_start + info.data_offsets.0)?;

        match info.dtype.as_str() {
            "BF16" => Ok(bf16_to_f32(&raw)),
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
}

/// Turn a run of bf16 bytes into f32 numbers.
///
/// A bf16 IS the top half of an f32 -- same sign bit, same 8 exponent bits,
/// same bias. So widening it is just sticking 16 zero bits on the bottom.
/// No maths, just moving bits.
///
/// ```text
///     bf16:                    0011111000001011
///     f32:     0011111000001011 0000000000000000
/// ```
fn bf16_to_f32(raw: &[u8]) -> Vec<f32> {
    raw.chunks_exact(2)
        .map(|pair| {
            let bits = u16::from_le_bytes([pair[0], pair[1]]);
            // from_bits REINTERPRETS the bits as a float.
            // (`as f32` would CONVERT the number instead -- totally different.)
            f32::from_bits((bits as u32) << 16)
        })
        .collect()
}

pub fn parse_header(bytes: &[u8]) -> Result<HashMap<String, TensorInfo>, serde_json::Error> {
    let raw: HashMap<String, serde_json::Value> = serde_json::from_slice(bytes)?;

    let mut tensors = HashMap::new();
    for (name, value) in raw {
        // Metadata isn't a tensor.
        if name == "__metadata__" {
            continue;
        }
        let info: TensorInfo = serde_json::from_value(value)?;
        tensors.insert(name, info);
    }

    Ok(tensors)
}
