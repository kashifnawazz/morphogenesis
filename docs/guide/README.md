# Morphogenesis Guide

Learning LLM inference from zero by building an engine that serves a model **from disk**.

These guides assume you can program but know nothing about AI. Read them in order.
Each one is written concept-first: the plain-language idea, then the technical detail,
then why it matters for a disk-based engine.

| # | guide | covers |
|---|---|---|
| 01 | [Reading the Config](01-reading-the-config.md) | What every field in `config.json` means. Tokens, the residual stream, what a *projection* is, attention, GQA, SwiGLU, RMSNorm, RoPE, weight tying. |
| 02 | [Model File Formats](02-model-file-formats.md) | pickle, safetensors, GGUF, ONNX and the rest — exact byte layouts. Block quantisation, K-quants, and why `Q4_K_M` isn't a type. Designing our own `.morph` format. |
| 03 | [bf16 and Numbers](03-bf16-and-numbers.md) | Floating-point bit layouts. Why ML chose bf16 over f16, why bf16 → f32 is a shift, and the rule that decides every kernel: store in bf16, compute in f32. |
| 04 | [The Tokenizer](04-the-tokenizer.md) | How text becomes numbers. Byte-level BPE, the pre-tokenizer regex, why `Ġ` means space, special tokens, ChatML, and the streaming-UTF-8 trap. |

Next up: **Phase 0** — build the numpy reference implementation. Guides 01–04 cover
everything you need for it.

## Reference material

- [`../roadmap.md`](../roadmap.md) — the eight build phases and their validation gates
- `models/qwen3-0.6b/config.json` — the model we're implementing

## The thesis, in one equation

Decoding a token reads every weight exactly once, for about 2 FLOPs per byte. The CPU is
never the bottleneck:

```
time_per_token  ≈  bytes_of_weights / bandwidth_of_the_tier_they_live_in
```

Every optimisation in this project reduces to one of three things:

1. **Move fewer bytes** — quantisation, weight tying, sparsity
2. **Move them in an order the device likes** — sequential beats random by ~100× on a HDD
3. **Move them while doing something else** — prefetch, overlap I/O with compute
