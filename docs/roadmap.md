# Morphogenesis — Roadmap

An inference engine that serves an LLM **from disk**, not from RAM or VRAM.
Target model: **Qwen3-0.6B**, bf16 safetensors.
Goal: understand LLM inference end to end by being forced to account for every byte that moves.

## The thesis

Decoding one token is **memory-bandwidth-bound, not compute-bound**. Each weight is read
exactly once per token and used for one multiply-accumulate against a vector — roughly
2 FLOPs per byte loaded. So:

```
time_per_token ≈ bytes_of_weights / bandwidth_of_the_tier_they_live_in
```

Disk inference is not a different algorithm. It is the same algorithm with the *bottom* of
the memory hierarchy as the weight store. Every optimisation therefore reduces to one of:

1. **Move fewer bytes** — quantisation, weight tying, sparsity.
2. **Move them in an order the device likes** — sequential beats random by ~100× on HDD.
3. **Move them while doing something else** — prefetch, overlap I/O with compute.

## Target hardware (this machine)

| | |
|---|---|
| CPU | Threadripper 1950X, 16C/32T, AVX2 (Zen 1 — double-pumped, no AVX-512) |
| RAM | 15 GB total, ~4 GB typically free |
| Tier A — project disk | `/dev/sda3`, 7200rpm HDD, ~150 MB/s seq, **~10 ms seek** |
| Tier B — `/home` | Samsung 970 EVO NVMe, ~3 GB/s, ~80 µs latency |

Two storage tiers three orders of magnitude apart. The same engine benchmarked on both is
the core experiment.

### Predicted ceilings (bf16, every weight streamed per token)

| tier | s/token | tok/s |
|---|---|---|
| RAM | 0.03 | ~33 |
| NVMe | 0.4 | ~2.5 |
| HDD | 8.0 | ~0.125 |

## Model facts (measured, not assumed)

From `config.json` and the safetensors header:

| field | value | note |
|---|---|---|
| `hidden_size` | 1024 | |
| `num_hidden_layers` | 28 | |
| `num_attention_heads` | 16 | |
| `head_dim` | **128** | **≠ hidden_size / n_heads (64)** — attention *expands* to 2048 |
| `num_key_value_heads` | 8 | GQA, 2 query heads share each KV head |
| `intermediate_size` | 3072 | SwiGLU |
| `vocab_size` | 151936 | |
| `rms_norm_eps` | 1e-6 | |
| `rope_theta` | 1e6 | |
| `tie_word_embeddings` | true | but see below |

Per-layer weights (31,461,888 B = 31.46 MB each):

| tensor | shape `[out, in]` | bytes |
|---|---|---|
| `input_layernorm` | [1024] | 2,048 |
| `self_attn.q_proj` | [2048, 1024] | 4,194,304 |
| `self_attn.k_proj` | [1024, 1024] | 2,097,152 |
| `self_attn.v_proj` | [1024, 1024] | 2,097,152 |
| `self_attn.q_norm` | [128] | 256 |
| `self_attn.k_norm` | [128] | 256 |
| `self_attn.o_proj` | [1024, 2048] | 4,194,304 |
| `post_attention_layernorm` | [1024] | 2,048 |
| `mlp.gate_proj` | [3072, 1024] | 6,291,456 |
| `mlp.up_proj` | [3072, 1024] | 6,291,456 |
| `mlp.down_proj` | [1024, 3072] | 6,291,456 |

Total params 596,049,920 → 1.11 GiB bf16.
Bytes actually read per decoded token: 880.9 MB (28 layers) + 311.2 MB (`lm_head`) + 2 KB
(one embedding row) ≈ **1.19 GB**.

### Two facts the file itself revealed

1. **`lm_head.weight` and `model.embed_tokens.weight` are both stored**, 311 MB each,
   despite `tie_word_embeddings: true`. 21% of the file is a duplicate. Load one, alias it.
2. **Physical tensor order is lexicographic, not execution order.** Layers appear as
   0, 1, 10, 11, …, 19, 2, 20, … and within a layer `down_proj` precedes `gate_proj`.
   Executing 0→27 against this layout means seeking across a 1.4 GB file, ~11 seeks/layer
   ≈ 308 seeks/token ≈ **3.1 s of pure seek latency per token on the HDD**, on top of
   the 7.9 s of transfer. This is what motivates the repacking step in Phase 6.

## Phases

Each phase has a **gate**: an objective check that must pass before moving on. Skipping a
gate means debugging a transformer with no oracle, which is where these projects die.

### Phase 0 — Ground truth
Build a numpy-only reference implementation of Qwen3-0.6B (no torch — Python 3.14 has no
wheels). ~200 lines. It is slow and that is fine; it exists to be *correct* and to dump
intermediate activations for a fixed prompt.
**Gate:** reference produces sensible greedy continuation for a known prompt; activation
dumps saved to `tests/fixtures/`.

### Phase 1 — Read the file
`safetensors` header parser, `config.json` loader, bf16→f32 conversion. Memory-map the
file; do not read it all.
**Gate:** print every tensor name/shape/offset; round-trip a known tensor's first 8 values
against the numpy reference.

### Phase 2 — Kernels
RMSNorm, matvec, softmax, SiLU/SwiGLU, RoPE. Each written against its own unit test using
Phase 0 fixtures.
**Gate:** every kernel matches numpy to < 1e-4 relative error, individually.

### Phase 3 — Forward pass, one token, no cache
Assemble the kernels into a full 28-layer forward. Watch for the two traps: `head_dim=128`
(not 64) and Qwen3's **QK-Norm** (RMSNorm applied per-head to Q and K after projection,
before RoPE) — absent in Llama/Qwen2, silently wrong if omitted.
**Gate:** logits match the numpy reference to < 1e-3; argmax token is identical.

### Phase 4 — Tokenizer
Byte-level BPE, Qwen's tiktoken-style vocab. Until this lands, feed token IDs produced by
the reference.
**Gate:** round-trip encode/decode byte-identical on a corpus of test strings, matching the
reference tokenizer.

### Phase 5 — KV cache and incremental decode
Carry K/V between steps so generating T tokens is O(T) forwards, not O(T²). GQA means the
cache is 8 heads wide, not 16.
**Gate:** incremental decode produces token-identical output to full recompute.

### Phase 6 — Disk streaming (the actual thesis)
Weights are never fully resident. Stream layer *n* while computing layer *n−1*.
- Repack tensors into execution order in a single `.morph` file (kills the 3.1 s/token seek tax)
- Compare `mmap` + `madvise` against explicit `O_DIRECT` reads, with real measurements
- Double-buffered prefetch on a background thread
- Configurable resident set: pin the hottest layers, stream the rest
**Gate:** identical output to Phase 5 at every memory budget; a benchmark table of
bytes-read and ms/token per configuration, HDD vs NVMe.

### Phase 7 — Quantisation
Implement Q8_0 and Q4_0 block quantisation. This is a *bandwidth* optimisation: 4× fewer
bytes is 4× faster on a bandwidth-bound engine.
**Gate:** perplexity delta vs bf16 measured and reported; speedup measured on both tiers.

### Phase 8 — Serving
OpenAI-compatible HTTP endpoint, streaming tokens.
**Gate:** works against a standard client.

## Where this improves on `kimi-k3-in-c`

The reference (FareedKhan-dev/kimi-k3-in-c) runs a 2.78T-param MoE in 8.24 GB by streaming
experts from disk. Worth stealing: trunk packing, LRU expert cache, staged validation gates.
Worth doing better:

- **Rust, not C99** — no manual buffer lifetime management around async I/O
- **Measure, don't assume** — the reference picked `O_DIRECT`; benchmark it against `mmap`
- **Per-kernel validation**, not just three end-to-end gates
- **Async prefetch** — the reference reads synchronously; overlap I/O with compute
- **Two-tier benchmarking** — HDD vs NVMe on identical code, which the reference never had
