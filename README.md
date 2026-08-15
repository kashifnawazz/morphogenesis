# Morphogenesis

**An LLM inference engine that serves the model from disk — not RAM, not VRAM.**

Written in Rust, from scratch, with no ML framework. Target model: **Qwen3-0.6B**.

This is a learning project, and it is deliberately slow. The point isn't throughput — it's
that a disk-based engine *forces* you to account for every single byte that moves, and that
turns out to be the clearest possible way to understand how an LLM actually runs.

---

## The idea

Generating one token is **memory-bandwidth-bound, not compute-bound**. Every weight is read
exactly once per token and used for a single multiply-accumulate against a vector — about
**2 FLOPs per byte loaded**. The CPU spends nearly all its time waiting.

So the whole engine collapses to one equation:

```
time_per_token  ≈  bytes_of_weights / bandwidth_of_the_tier_they_live_in
```

Disk inference isn't a different algorithm. It's the *same* algorithm with the bottom of the
memory hierarchy as the weight store. Which means every optimisation reduces to one of three
things:

1. **Move fewer bytes** — quantisation, weight tying, sparsity
2. **Move them in an order the device likes** — sequential beats random by ~100× on a HDD
3. **Move them while doing something else** — prefetch, overlap I/O with compute

Peak memory should be **one layer (31.46 MB)**, not the whole model.

---

## Status

**Early. No inference yet.** Currently: the model is downloaded, the file formats are fully
reverse-engineered and documented, and the tokenizer has been verified by a from-scratch
reference implementation. The Rust engine itself is not started.

What's real right now is [the guides](docs/guide/) — and they're written from measurements
of the actual files, not from recollection.

---

## The guides

Written for someone who can program but knows nothing about AI. Concept first, then
technical detail, then why it matters for a disk-based engine.

| # | guide | covers |
|---|---|---|
| 01 | [Reading the Config](docs/guide/01-reading-the-config.md) | Every field in `config.json`. Tokens, the residual stream, what a *projection* is, attention, GQA, SwiGLU, RMSNorm, RoPE, weight tying. |
| 02 | [Model File Formats](docs/guide/02-model-file-formats.md) | pickle, safetensors, GGUF, ONNX — exact byte layouts. Block quantisation, K-quants, and why `Q4_K_M` isn't a type. |
| 03 | [bf16 and Numbers](docs/guide/03-bf16-and-numbers.md) | Float bit layouts. Why ML chose bf16 over f16, why bf16 → f32 is a shift, and the rule behind every kernel: store in bf16, compute in f32. |
| 04 | [The Tokenizer](docs/guide/04-the-tokenizer.md) | Byte-level BPE, the pre-tokenizer regex, why `Ġ` means space, ChatML, and the streaming-UTF-8 trap. |

Some things they turned up by measuring rather than assuming:

- The safetensors file stores the tied embedding matrix **twice** — 311 MB, 21% of the file,
  pure duplication.
- Tensors are laid out **lexicographically**, so layers appear as 0, 1, 10, 11, … 2, 20, …
  Running them in execution order means ~300 seeks per token.
- `head_dim` is **128**, not `hidden_size / num_heads` = 64. Assume the latter and every
  shape still fits — the model just emits fluent nonsense.
- 256 byte tokens + 151,387 merge rules = 151,643 = exactly the size of `vocab.json`.
- **267 embedding rows are unreachable** by the tokenizer, because 151,936 = 128 × 1,187.

---

## The model

[Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B), bf16 safetensors — 596,049,920
parameters, 1.40 GiB on disk.

| | | | |
|---|---|---|---|
| layers | 28 | hidden size | 1024 |
| attention heads | 16 | KV heads (GQA) | 8 |
| head dim | **128** | MLP intermediate | 3072 |
| vocab | 151,936 | RoPE θ | 1e6 |

Bytes read per decoded token: **~1.19 GB** — 880.9 MB of layers, 311.2 MB of `lm_head`,
and a single 2 KB embedding row.

---

## The lab

The development machine has **two storage tiers three orders of magnitude apart**, which
makes it an unusually good place to study this problem:

| tier | device | bandwidth | latency |
|---|---|---|---|
| RAM | 15 GB DDR4 | ~40 GB/s | ~80 ns |
| NVMe | Samsung 970 EVO | ~3 GB/s | ~80 µs |
| HDD | 7200rpm SATA | ~150 MB/s | **~10 ms seek** |

Predicted ceilings at bf16, streaming every weight every token:

| tier | s/token | tok/s |
|---|---:|---:|
| RAM | 0.03 | ~33 |
| NVMe | 0.40 | ~2.5 |
| HDD | 8.0 | ~0.125 |

Same code, same model, **266× spread**. Benchmarking identical code across both disk tiers
is the core experiment — and it's something the prior art never had.

---

## Roadmap

Each phase has a **gate**: an objective check that must pass before moving on. Skipping one
means debugging a transformer with no oracle, which is where these projects die.

- [ ] **0 — Ground truth.** numpy-only reference implementation (no torch). Dumps
      intermediate activations for a fixed prompt.
      *Gate: sensible greedy continuation; fixtures saved.*
- [ ] **1 — Read the file.** safetensors parser, config loader, bf16 → f32, mmap.
      *Gate: tensor values match the reference.*
- [ ] **2 — Kernels.** RMSNorm, matvec, softmax, SwiGLU, RoPE — each unit-tested alone.
      *Gate: every kernel within 1e-4 relative error.*
- [ ] **3 — Forward pass.** One token, no cache, all 28 layers.
      *Gate: logits within 1e-3; argmax identical.*
- [ ] **4 — Tokenizer.** Byte-level BPE.
      *Gate: byte-identical round-trip against the reference.*
- [ ] **5 — KV cache.** Incremental decode, O(T) instead of O(T²).
      *Gate: token-identical to full recompute.*
- [ ] **6 — Disk streaming.** *The actual thesis.* Repack into execution order, compare
      `mmap` vs `O_DIRECT`, double-buffered prefetch, configurable resident set.
      *Gate: identical output at every memory budget, with a bytes-read and ms/token table.*
- [ ] **7 — Quantisation.** Q8_0 and Q4_0. A bandwidth optimisation: 4× fewer bytes is ~4×
      faster when you're bandwidth-bound.
      *Gate: perplexity delta and speedup measured on both tiers.*
- [ ] **8 — Serving.** OpenAI-compatible streaming endpoint.

Full detail in [`docs/roadmap.md`](docs/roadmap.md).

---

## Getting started

Nothing to run yet. To set up the workspace:

```bash
git clone https://github.com/kashifnawazz/morphogenesis.git
cd morphogenesis
cargo build
```

Fetch the model (~1.4 GiB, gitignored):

```bash
mkdir -p models/qwen3-0.6b && cd models/qwen3-0.6b
B=https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main
for f in config.json generation_config.json tokenizer.json \
         tokenizer_config.json vocab.json merges.txt model.safetensors; do
  curl -L -o "$f" "$B/$f"
done
```

Verify — `model.safetensors` should be exactly **1,503,300,328** bytes, and its first
tensor should be `lm_head.weight`, `[151936, 1024]`, BF16.

---

## Layout

```
morphogenesis-core/       the engine — loader, kernels, model, streaming
morphogenesis-cli/        command-line entry point
docs/
  guide/                  the learning series (01–04)
  roadmap.md              phases and validation gates
models/                   downloaded weights (gitignored)
```

---

## Prior art

Directly inspired by [**FareedKhan-dev/kimi-k3-in-c**](https://github.com/FareedKhan-dev/kimi-k3-in-c),
which runs a 2.78-trillion-parameter MoE model on a single CPU in 8.24 GB of RAM by
streaming dormant experts from a 1.56 TB checkpoint. Worth stealing from it: trunk packing,
the LRU expert cache, and staged validation gates.

Where this project intends to differ:

- **Rust, not C99** — no manual buffer lifetime management around async I/O
- **Measure, don't assume** — the reference chose `O_DIRECT`; we benchmark it against `mmap`
- **Per-kernel validation**, not only end-to-end gates
- **Async prefetch** — the reference reads synchronously; overlap I/O with compute
- **Two-tier benchmarking** — identical code on HDD and NVMe

Also owes a debt to [karpathy/llama2.c](https://github.com/karpathy/llama2.c) and
[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp).

---

## License

[Apache 2.0](LICENSE)
