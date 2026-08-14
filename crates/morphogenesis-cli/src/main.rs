use std::path::Path;

use morphogenesis_core::safetensors;

fn main() -> std::io::Result<()> {
    let path = Path::new("models/qwen3-0.6b/model.safetensors");

    let header_len = safetensors::read_header_len(path)?;
    println!("header length = {header_len}");

    let header = safetensors::read_header_bytes(path)?;
    println!("header bytes read = {}", header.len());
    println!("first 80 bytes:\n{}", String::from_utf8_lossy(&header[..80]));

    Ok(())
}
