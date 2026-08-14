use std::fs::File;
use std::io::Read;
use std::path::Path;

/// Reads the first 8 bytes of a safetensors file, which hold the length of
/// the table of contents.
pub fn read_header_len(path: &Path) -> std::io::Result<u64> {
    // Open the file. The `?` means "if this fails, stop and return the error".
    let mut file = File::open(path)?;

    // A box to hold exactly 8 bytes, all zero for now.
    let mut buf = [0u8; 8];

    // Fill the box with the first 8 bytes of the file.
    file.read_exact(&mut buf)?;

    // Turn those 8 raw bytes into a number.
    // `le` = little-endian: the first byte is the smallest part of the number.
    Ok(u64::from_le_bytes(buf))
}

/// Read the 
pub fn read_header_bytes(path: &Path) -> std::io::Result<Vec<u8>> {
    let mut file = File::open(path)?;

    // First 8 bytes contain the header length.
    let mut header_len_bytes = [0u8; 8];
    file.read_exact(&mut header_len_bytes)?;

    let header_len = u64::from_le_bytes(header_len_bytes) as usize;

    // File is already positioned at byte 8.
    let mut header = vec![0u8; header_len];
    file.read_exact(&mut header)?;

    Ok(header)
}
