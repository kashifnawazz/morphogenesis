//! Turning text into token ids and back.
//!
//! This is byte-level BPE. Four stages, in order:
//!
//!   1. pull out special tokens like <|im_end|> and keep them whole
//!   2. split the rest with a regex -- merges may never cross these splits
//!   3. swap every byte for a stand-in printable character
//!   4. glue pairs back together using learned merge rules
//!
//! Stage 3 is why the vocabulary is full of things like "Ġcapital".
//! `Ġ` is just a space. See docs/guide/04-the-tokenizer.md.

use std::collections::HashMap;
use std::io;
use std::path::Path;

use fancy_regex::Regex;
use serde::Deserialize;

/// Qwen's splitting rule, copied out of tokenizer.json.
///
/// Reading it piece by piece:
///   contractions ('s, 't, ...) | letters with an optional leading symbol |
///   ONE digit | punctuation | newlines | trailing spaces | any whitespace
///
/// Note `\p{N}` on its own: digits are always split one at a time, so "12345"
/// is five tokens. That is deliberate -- it makes arithmetic far more reliable.
const SPLIT_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

#[derive(Deserialize)]
struct TokenizerJson {
    added_tokens: Vec<AddedToken>,
}

#[derive(Deserialize)]
struct AddedToken {
    id: u32,
    content: String,
}

pub struct Tokenizer {
    /// piece text -> id
    vocab: HashMap<String, u32>,
    /// id -> piece text (a Vec because ids are dense from 0)
    id_to_piece: Vec<String>,
    /// "a b" -> how early this merge was learned. Lower wins.
    merges: HashMap<String, u32>,
    /// byte value -> stand-in character
    byte_enc: [char; 256],
    /// stand-in character -> byte value
    byte_dec: HashMap<char, u8>,
    /// tokens matched literally, before the regex ever sees them
    specials: Vec<(String, u32)>,
    pattern: Regex,
}

impl Tokenizer {
    /// `dir` holds vocab.json, merges.txt and tokenizer.json.
    pub fn load(dir: &Path) -> io::Result<Self> {
        // --- the vocabulary -------------------------------------------------
        let vocab_text = std::fs::read_to_string(dir.join("vocab.json"))?;
        let vocab: HashMap<String, u32> =
            serde_json::from_str(&vocab_text).map_err(io::Error::other)?;

        // --- the merge rules, in the order they were learned -----------------
        let merges_text = std::fs::read_to_string(dir.join("merges.txt"))?;
        let mut merges = HashMap::new();
        for line in merges_text.lines() {
            if line.starts_with("#version") || line.is_empty() {
                continue;
            }
            // The file is already "a b" per line, which is exactly the key
            // we want -- pieces can never contain a space, because a space
            // byte becomes 'Ġ' in stage 3.
            merges.insert(line.to_string(), merges.len() as u32);
        }

        // --- the special tokens ---------------------------------------------
        let tj_text = std::fs::read_to_string(dir.join("tokenizer.json"))?;
        let tj: TokenizerJson = serde_json::from_str(&tj_text).map_err(io::Error::other)?;
        let specials: Vec<(String, u32)> =
            tj.added_tokens.iter().map(|a| (a.content.clone(), a.id)).collect();

        // --- id -> piece, covering both ordinary and special tokens ---------
        let max_id = vocab
            .values()
            .copied()
            .chain(specials.iter().map(|(_, id)| *id))
            .max()
            .unwrap_or(0);
        let mut id_to_piece = vec![String::new(); max_id as usize + 1];
        for (piece, &id) in &vocab {
            id_to_piece[id as usize] = piece.clone();
        }
        for (piece, id) in &specials {
            id_to_piece[*id as usize] = piece.clone();
        }

        let byte_enc = byte_encoder();
        let byte_dec = byte_enc
            .iter()
            .enumerate()
            .map(|(b, &c)| (c, b as u8))
            .collect();

        Ok(Tokenizer {
            vocab,
            id_to_piece,
            merges,
            byte_enc,
            byte_dec,
            specials,
            pattern: Regex::new(SPLIT_PATTERN).map_err(io::Error::other)?,
        })
    }

    pub fn vocab_len(&self) -> usize {
        self.id_to_piece.len()
    }

    /// The raw piece text for an id, e.g. 12095 -> "ĠParis".
    pub fn piece(&self, id: u32) -> &str {
        self.id_to_piece.get(id as usize).map_or("", |s| s.as_str())
    }

    /// Text in, token ids out.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        self.encode_with_specials(text, &mut out);
        out
    }

    /// Pull special tokens out first so the regex never chops them up.
    ///
    /// Without this, "<|im_end|>" would be shredded into "<", "|", "im", "_",
    /// "end", "|", ">" and the model would see punctuation soup instead of a
    /// control token.
    fn encode_with_specials(&self, text: &str, out: &mut Vec<u32>) {
        let mut earliest: Option<(usize, usize, u32)> = None;
        for (s, id) in &self.specials {
            if let Some(pos) = text.find(s.as_str()) {
                if earliest.is_none_or(|(best, _, _)| pos < best) {
                    earliest = Some((pos, s.len(), *id));
                }
            }
        }

        match earliest {
            Some((pos, len, id)) => {
                self.encode_ordinary(&text[..pos], out);
                out.push(id);
                self.encode_with_specials(&text[pos + len..], out);
            }
            None => self.encode_ordinary(text, out),
        }
    }

    fn encode_ordinary(&self, text: &str, out: &mut Vec<u32>) {
        if text.is_empty() {
            return;
        }
        for chunk in self.pattern.find_iter(text) {
            let chunk = match chunk {
                Ok(m) => m.as_str(),
                Err(_) => continue,
            };

            // Stage 3: every byte becomes a stand-in character.
            let encoded: String = chunk.bytes().map(|b| self.byte_enc[b as usize]).collect();

            // Stage 4: glue pairs together, then look each piece up.
            for piece in self.bpe(&encoded) {
                if let Some(&id) = self.vocab.get(&piece) {
                    out.push(id);
                }
            }
        }
    }

    /// Repeatedly join the best-ranked adjacent pair until none is left.
    ///
    /// "Best" means lowest rank -- learned earliest, so most frequent.
    /// It is NOT left-to-right: for " morphogenesis" the pair 'e'+'n'
    /// (rank 12) fires before 'o'+'r' (rank 13), even though 'or' comes
    /// first in the word. Getting this backwards gives valid-looking but
    /// wrong tokens.
    fn bpe(&self, word: &str) -> Vec<String> {
        let mut parts: Vec<String> = word.chars().map(|c| c.to_string()).collect();

        loop {
            let mut best: Option<(u32, usize)> = None;
            for i in 0..parts.len().saturating_sub(1) {
                let key = format!("{} {}", parts[i], parts[i + 1]);
                if let Some(&rank) = self.merges.get(&key) {
                    if best.is_none_or(|(r, _)| rank < r) {
                        best = Some((rank, i));
                    }
                }
            }

            let Some((_, i)) = best else { break };
            let joined = format!("{}{}", parts[i], parts[i + 1]);
            parts[i] = joined;
            parts.remove(i + 1);
        }

        parts
    }

    /// Token ids back into text.
    ///
    /// Note this goes via BYTES, not characters. A single token can hold half
    /// a multi-byte character -- "日本語" is 9 bytes split across 2 tokens --
    /// so decoding tokens one at a time would produce broken output.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            for c in self.piece(id).chars() {
                match self.byte_dec.get(&c) {
                    Some(&b) => bytes.push(b),
                    // Special tokens aren't made of stand-in characters.
                    None => bytes.extend_from_slice(c.to_string().as_bytes()),
                }
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

/// Build the byte <-> character table.
///
/// Bytes that are already printable stand for themselves. The other 68
/// (control codes, space, and a few gaps) get pushed up into the U+0100 block
/// so that no whitespace or control character ever appears in a merge rule.
///
/// That is the whole trick: byte 32 (space) becomes U+0120 'Ġ'.
fn byte_encoder() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut next = 0u32;
    for b in 0..=255usize {
        let printable = (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
        table[b] = if printable {
            char::from_u32(b as u32).unwrap()
        } else {
            let c = char::from_u32(256 + next).unwrap();
            next += 1;
            c
        };
    }
    table
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tk() -> Tokenizer {
        Tokenizer::load(Path::new("../models/qwen3-0.6b")).expect("model files missing")
    }

    #[test]
    fn byte_table_matches_the_spec() {
        let t = byte_encoder();
        assert_eq!(t[0], '\u{0100}');
        assert_eq!(t[10], '\u{010A}'); // newline
        assert_eq!(t[32], '\u{0120}'); // space -> the famous Ġ
        assert_eq!(t[65], 'A');
        assert_eq!(t[97], 'a');
        assert_eq!(t[255], '\u{00FF}');
    }

    #[test]
    fn encodes_the_prompt_we_hardcoded() {
        assert_eq!(
            tk().encode("The capital of France is"),
            vec![785, 6722, 315, 9625, 374]
        );
    }

    #[test]
    fn known_encodings() {
        let t = tk();
        assert_eq!(t.encode(" Paris"), vec![12095]);
        assert_eq!(t.encode("Hello, world!"), vec![9707, 11, 1879, 0]);
        assert_eq!(t.encode("morphogenesis"), vec![89833, 51279]);
        assert_eq!(t.encode("日本語"), vec![101059, 102819]);
        assert_eq!(t.encode("🦀"), vec![147579]);
    }

    #[test]
    fn digits_are_always_split_one_at_a_time() {
        assert_eq!(tk().encode("12345"), vec![16, 17, 18, 19, 20]);
    }

    #[test]
    fn leading_space_changes_the_first_token() {
        // Tokenisation is not composable -- you can't encode two strings
        // separately and glue the ids together.
        let t = tk();
        assert_eq!(t.encode("morphogenesis"), vec![89833, 51279]);
        assert_eq!(t.encode(" morphogenesis"), vec![26351, 51279]);
    }

    #[test]
    fn round_trips() {
        let t = tk();
        for text in [
            "The capital of France is",
            "Hello, world!",
            "12345",
            "morphogenesis",
            "日本語",
            "🦀 crab",
            "def foo():\n    return 1",
            "don't  stop",
            "   leading and trailing   ",
        ] {
            assert_eq!(t.decode(&t.encode(text)), text, "failed on {text:?}");
        }
    }

    #[test]
    fn special_tokens_stay_whole() {
        let t = tk();
        let ids = t.encode("<|im_start|>user\nhi<|im_end|>");
        assert_eq!(ids[0], 151644); // <|im_start|>
        assert_eq!(*ids.last().unwrap(), 151645); // <|im_end|>
        assert_eq!(t.decode(&ids), "<|im_start|>user\nhi<|im_end|>");
    }
}
