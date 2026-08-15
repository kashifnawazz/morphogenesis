use std::path::Path;

use morphogenesis_core::config::Config;
use morphogenesis_core::safetensors::SafeTensors;

fn main() -> std::io::Result<()> {
    let path = Path::new("models/qwen3-0.6b/model.safetensors");
    let cfg = Config::load(Path::new("models/qwen3-0.6b/config.json"))?;
    let st = SafeTensors::open(path)?;

    println!("tensors    = {}", st.len());
    println!("data_start = {}", st.data_start());

    for name in [
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.self_attn.q_norm.weight",
        "model.norm.weight",
    ] {
        let info = st.info(name).unwrap();
        let values = st.get(name)?;
        println!("\n{name}");
        println!("  shape  {:?}   numel {}", info.shape, info.numel());
        println!("  first6 {:?}", &values[..6]);
    }

    let q = st.get("model.layers.0.self_attn.q_proj.weight")?;
    println!("\nmodel.layers.0.self_attn.q_proj.weight");
    println!("  numel  {}", q.len());
    println!("  first3 {:?}", &q[..3]);

    println!("{cfg:#?}");

    Ok(())
}
