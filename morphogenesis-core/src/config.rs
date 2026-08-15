use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Deserialize)]
#[derive(Clone)]
pub struct Config {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub tie_word_embeddings: bool,
}

impl Config {
    pub fn load(path: &Path) -> std::io::Result<Self> {
        let text = std::fs::read_to_string(path)?;

        let config: Self = serde_json::from_str(&text).map_err(std::io::Error::other)?;

        Ok(config)
    }

    /// 16 heads x 128 = 2048. Note this is NOT hiddes_size.
    pub fn q_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    /// 8 heads x 128 = 1024
    pub fn kv_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }
    /// How many query heads share each key/value head. Here: 2.
    pub fn heads_per_kv(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }
}
