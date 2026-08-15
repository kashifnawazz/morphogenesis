use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use morphogenesis_core::cache::Pinned;
use morphogenesis_core::config::Config;
use morphogenesis_core::model::{KvCache, Model};
use morphogenesis_core::morph::{self, Mode, MorphFile};
use morphogenesis_core::safetensors::SafeTensors;
use morphogenesis_core::tokenizer::Tokenizer;
use morphogenesis_core::weights::Weights;

const MODEL_DIR: &str = "models/qwen3-0.6b";
const MORPH_PATH: &str = "models/qwen3-0.6b/model.morph";

fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str).unwrap_or("run");

    match cmd {
        "repack" => cmd_repack(),
        "bench" => cmd_bench(),
        "run" => cmd_run(&args[1..]),
        _ => cmd_run(&args),
    }
}

fn mb(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

// ---------------------------------------------------------------------------

fn cmd_repack() -> std::io::Result<()> {
    let dir = Path::new(MODEL_DIR);
    let cfg = Config::load(&dir.join("config.json"))?;
    let src = SafeTensors::open(&dir.join("model.safetensors"))?;

    println!("repacking into execution order...");
    let start = Instant::now();
    let header = morph::repack(&src, &cfg, Path::new(MORPH_PATH))?;
    let elapsed = start.elapsed();

    let src_size = std::fs::metadata(dir.join("model.safetensors"))?.len();
    let out_size = std::fs::metadata(MORPH_PATH)?.len();

    println!("done in {elapsed:.1?}");
    println!("  source      {:>10.1} MB", mb(src_size));
    println!("  repacked    {:>10.1} MB", mb(out_size));
    println!("  saved       {:>10.1} MB", mb(src_size - out_size));
    println!("  embeddings tied (stored once): {}", header.embeddings_tied);
    println!("  layer stride {:.2} MB, 4 KB aligned", mb(header.layer_stride));
    Ok(())
}

// ---------------------------------------------------------------------------

fn cmd_run(args: &[String]) -> std::io::Result<()> {
    let dir = Path::new(MODEL_DIR);
    let prompt = if args.is_empty() {
        "The capital of France is".to_string()
    } else {
        args.join(" ")
    };

    let tk = Tokenizer::load(dir)?;
    let cfg = Config::load(&dir.join("config.json"))?;

    // Prefer the repacked file when it exists.
    let weights: Arc<dyn Weights> = if Path::new(MORPH_PATH).exists() {
        Arc::new(MorphFile::open(Path::new(MORPH_PATH), Mode::Mmap)?)
    } else {
        Arc::new(SafeTensors::open(&dir.join("model.safetensors"))?)
    };
    println!("weights: {}", weights.describe());

    let model = Model::new(cfg, weights);
    let tokens = tk.encode(&prompt);

    print!("{prompt}");
    std::io::stdout().flush()?;

    let start = Instant::now();
    let mut count = 0;
    model.generate(&tokens, 60, &[151645, 151643], |id| {
        print!("{}", tk.decode(&[id]));
        let _ = std::io::stdout().flush();
        count += 1;
    })?;
    let elapsed = start.elapsed();

    println!("\n");
    println!("{count} tokens in {elapsed:.2?}");
    if count > 0 {
        println!("{:.2} s/token", elapsed.as_secs_f64() / count as f64);
    }
    println!("{:.1} MB read", mb(model.weights().bytes_read()));
    Ok(())
}

// ---------------------------------------------------------------------------

fn cmd_bench() -> std::io::Result<()> {
    let dir = Path::new(MODEL_DIR);
    let cfg = Config::load(&dir.join("config.json"))?;
    let tk = Tokenizer::load(dir)?;
    let tokens = tk.encode("The capital of France is");

    if !Path::new(MORPH_PATH).exists() {
        println!("no {MORPH_PATH} -- run `morphogenesis-cli repack` first");
        return Ok(());
    }

    const REPEATS: usize = 3;
    println!("one forward pass over 5 tokens; median of {REPEATS} runs");
    println!("cold = this file's pages dropped from the kernel cache first\n");
    println!(
        "{:<40} {:>8} {:>8} {:>8} {:>9}",
        "configuration", "cold", "spread", "warm", "MB read"
    );
    println!("{}", "-".repeat(78));

    let st_path = dir.join("model.safetensors");
    bench_one("safetensors, pread", &cfg, &tokens, REPEATS, true, &st_path, || {
        Ok(Arc::new(SafeTensors::open(&st_path)?) as Arc<dyn Weights>)
    })?;

    let mp = PathBuf::from(MORPH_PATH);
    for (label, mode) in [
        ("morph, pread", Mode::Pread),
        ("morph, mmap", Mode::Mmap),
        ("morph, O_DIRECT", Mode::Direct),
    ] {
        bench_one(label, &cfg, &tokens, REPEATS, true, &mp, || {
            Ok(Arc::new(MorphFile::open(&mp, mode)?) as Arc<dyn Weights>)
        })?;
    }

    println!();
    // Isolate the prefetch hint: same config, only that flag differs.
    for (label, on) in [("morph, pread, prefetch OFF", false), ("morph, pread, prefetch ON", true)] {
        bench_one(label, &cfg, &tokens, REPEATS, on, &mp, || {
            Ok(Arc::new(MorphFile::open(&mp, Mode::Pread)?) as Arc<dyn Weights>)
        })?;
    }

    println!();
    for layers in [7usize, 14, 28] {
        let budget = layers as u64 * 63 * (1 << 20);
        let label = format!("morph, mmap + {layers} layers pinned");
        bench_one(&label, &cfg, &tokens, REPEATS, true, &mp, || {
            let f = MorphFile::open(&mp, Mode::Mmap)?;
            Ok(Arc::new(Pinned::new(f, &cfg, budget)?) as Arc<dyn Weights>)
        })?;
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn bench_one(
    label: &str,
    cfg: &Config,
    tokens: &[u32],
    repeats: usize,
    prefetch: bool,
    cache_file: &Path,
    build: impl Fn() -> std::io::Result<Arc<dyn Weights>>,
) -> std::io::Result<()> {
    let mut colds = Vec::new();
    let mut warms = Vec::new();
    let mut bytes = 0u64;

    for _ in 0..repeats {
        drop_cache(cache_file)?;
        let w = build()?;
        w.reset_bytes();
        let model = Model::new(cfg.clone(), Arc::clone(&w)).with_prefetch(prefetch);

        let mut kv = KvCache::new(cfg, tokens.len() + 1);
        let t0 = Instant::now();
        model.forward(tokens, &mut kv)?;
        colds.push(t0.elapsed().as_secs_f64());
        bytes = w.bytes_read();

        let mut kv = KvCache::new(cfg, tokens.len() + 1);
        let t1 = Instant::now();
        model.forward(tokens, &mut kv)?;
        warms.push(t1.elapsed().as_secs_f64());
    }

    colds.sort_by(f64::total_cmp);
    warms.sort_by(f64::total_cmp);
    let spread = colds[colds.len() - 1] - colds[0];

    println!(
        "{label:<40} {:>7.2}s {:>7.2}s {:>7.2}s {:>9.0}",
        colds[repeats / 2],
        spread,
        warms[repeats / 2],
        mb(bytes)
    );
    Ok(())
}

/// Ask the kernel to forget a file's cached pages. No root needed.
fn drop_cache(path: &Path) -> std::io::Result<()> {
    use std::os::unix::io::AsRawFd;
    let f = std::fs::File::open(path)?;
    unsafe {
        libc::posix_fadvise(f.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED);
    }
    Ok(())
}
