# 01 — Reading the Config

**What every number in `config.json` means, starting from zero.**

This is the first guide in the Morphogenesis series. It assumes you know how to program
but know nothing about AI. By the end you should be able to look at any model's
`config.json` and know what machine it describes.

Our subject is `models/qwen3-0.6b/config.json`:

```json
{
  "hidden_size": 1024,
  "num_hidden_layers": 28,
  "num_attention_heads": 16,
  "head_dim": 128,
  "num_key_value_heads": 8,
  "intermediate_size": 3072,
  "vocab_size": 151936,
  "rms_norm_eps": 1e-06,
  "rope_theta": 1000000,
  "tie_word_embeddings": true
}
```

Ten numbers. They completely specify the shape of a 596-million-parameter machine.

---

# Part 1 — The Concept

Read this part with no intention of remembering details. Just build the picture.

## What a language model actually does

A language model does exactly one thing:

> **Given some text, guess what comes next.**

That's the whole job. Not "understand", not "think" — just predict the next fragment of
text. Everything impressive that these systems do falls out of doing this one thing
extremely well, then feeding the answer back in and doing it again.

```
"The capital of France is"        → model → " Paris"
"The capital of France is Paris"  → model → "."
"The capital of France is Paris." → model → " It"
```

That loop — predict one piece, append it, predict again — is called **autoregressive
generation**, and it is the entire reason this project is hard. The model runs *completely*
for every single token it produces. A 500-token answer means 500 full passes through all
596 million parameters.

## The pipeline

Here's the journey of a prompt through the model:

```
   "The capital of France is"
            │
            │  ①  TOKENIZE — chop text into known fragments, look up their IDs
            ▼
   [785, 6722, 315, 9625, 374]
            │
            │  ②  EMBED — turn each ID into a list of 1024 numbers
            ▼
   5 vectors × 1024 numbers each
            │
            │  ③  28 LAYERS — refine those numbers, over and over
            ▼        (each layer has its own weights)
   5 vectors × 1024 numbers each   (now much more "informed")
            │
            │  ④  SCORE — turn the LAST vector into one score per vocabulary entry
            ▼
   151,936 scores
            │
            │  ⑤  PICK — choose one (highest score, or sample randomly)
            ▼
        " Paris"
```

Every config field describes one of these five stages. That's all the config is: the
dimensions of this pipeline.

## The single most important idea: the residual stream

If you take one thing from this guide, take this.

After step ②, each token is a list of **1024 numbers**. That list is called the
**residual stream**. Picture it as a whiteboard that belongs to that token position.

The 28 layers do **not** replace the whiteboard. Each layer *reads* what's on it, computes
a correction, and **adds** the correction back:

```
whiteboard = whiteboard + attention_correction
whiteboard = whiteboard + mlp_correction
```

In code that's literally:

```
x = x + attention(norm(x))
x = x + mlp(norm(x))
```

That `+` is the whole trick. Because every layer only *adds*, the original signal always
has a clean, unobstructed path from the input all the way to the output. Without it,
28 stacked layers would smear the signal into noise and the model would never train.

Two warnings about the whiteboard:

- **The 1024 numbers are not 1024 named features.** There is no "is-a-country" slot at
  index 412. Meaning is smeared across all 1024 dimensions in a distributed way that
  nobody designed and nobody fully understands.
- **It is per-position.** Five tokens means five separate 1024-number whiteboards, running
  in parallel down through the layers.

## What happens inside one layer

All 28 layers have identical structure and completely different weights. Each does two
things in order:

**1. Attention — "let the positions talk to each other."**
This is the *only* place in the entire model where information moves between token
positions. To predict the word after `"The capital of France is"`, the last position needs
to look back and find `"France"`. Attention is the mechanism that lets it.

**2. MLP — "let each position think on its own."**
Every position, independently and in isolation, runs its numbers through a small
feed-forward network. No communication here. The loose intuition is that attention
*retrieves* relevant context while the MLP *knows facts*.

```
   ┌─────────────── one layer, repeated 28× ───────────────┐
   │                                                        │
x ─┼──┬───────────────────────────────────────────┐        │
   │  │  normalise                                 │        │
   │  │  ATTENTION  ← positions exchange info      │        │
   │  └──────────────────► + ◄────────────────────┘        │
   │                       │                                │
   │  ┌────────────────────┴──────────────────────┐        │
   │  │  normalise                                 │        │
   │  │  MLP  ← each position alone                │        │
   │  └──────────────────► + ◄────────────────────┘        │
   │                       │                                │
   └───────────────────────┼────────────────────────────────┘
                           ▼  to the next layer
```

That's the concept. Now the details.

---

# Interlude — What a "Projection" Is

Before the fields, one piece of vocabulary you'll hit constantly. Almost every weight
tensor in the model is named `something_proj`:

```
q_proj   k_proj   v_proj   o_proj   gate_proj   up_proj   down_proj
```

**"Projection" is just linear-algebra vocabulary for a matrix multiply.** In code it's a
*linear layer*:

```
y = W · x
```

A vector goes in, a vector comes out. Nothing more exotic than that.

## The row-by-row intuition

The useful way to picture it: each **row** of the matrix is a learned direction, and each
output number is a dot product against that row.

```
        W (2048 × 1024)              x (1024)         y (2048)
   ┌──────────────────────┐          ┌───┐            ┌───┐
   │ row 0: learned dir.  │  ·       │   │       =    │ y₀ │  ← "how much does x
   ├──────────────────────┤          │ x │            ├───┤     point along row 0?"
   │ row 1: learned dir.  │  ·       │   │       =    │ y₁ │
   ├──────────────────────┤          │   │            ├───┤
   │ ...    2048 rows     │          └───┘            │...│
   └──────────────────────┘                           └───┘
```

So a matrix multiply is **2048 separate dot products**, each asking *"how much does this
token resemble the pattern I learned in row i?"* The collected answers form the output.

The model learned those numbers during training. At inference you only ever multiply.

## The seven projections

| tensor | name | shape `[out, in]` | what it does |
|---|---|---|---|
| `q_proj` | **q**uery | [2048, 1024] | residual → Query vectors (16 heads) |
| `k_proj` | **k**ey | [1024, 1024] | residual → Key vectors (8 heads, GQA) |
| `v_proj` | **v**alue | [1024, 1024] | residual → Value vectors (8 heads) |
| `o_proj` | **o**utput | [1024, 2048] | mixes the 16 heads back to residual width |
| `gate_proj` | gate | [3072, 1024] | MLP — the branch that gets SiLU'd |
| `up_proj` | up | [3072, 1024] | MLP — the branch that gets gated |
| `down_proj` | down | [1024, 3072] | MLP — back down to residual width |

Read the shapes as `[out, in]` and the whole dataflow appears: everything starts at 1024,
**expands** (2048 for attention, 3072 for the MLP), then comes back to 1024. The residual
stream is a fixed-width highway; projections are the on- and off-ramps.

Note that one matrix produces **all heads at once**. `q_proj` outputs 2048 numbers, which
you then slice into 16 heads of 128:

```
x  →  q_proj [2048,1024]  →  2048 numbers  →  reshape  →  16 heads × 128 dims
                                                           head 0 = q[0..128]
                                                           head 1 = q[128..256] ...
```

## Two implementation details

**Shapes are stored pre-transposed.** `[out_features, in_features]` is PyTorch's
`nn.Linear` convention, and it's the convenient one: row `i` is contiguous in memory, so
`y[i] = dot(row_i, x)` reads sequential bytes. Good for cache, good for SIMD.

**There is no bias.** The config says `"attention_bias": false`, and every tensor in the
file ends in `.weight` — there are no `.bias` tensors at all. So it really is `y = W · x`,
not `y = W · x + b`. Modern LLMs mostly dropped biases; they cost parameters and don't help.

---

# Part 2 — The Fields

## `vocab_size = 151936`

**The size of the model's dictionary.**

The model cannot read text — it only handles numbers. So text is first chopped into
**tokens** (usually word fragments) and each token is looked up in a fixed dictionary:

```
"The capital of France is"
   → ["The", " capital", " of", " France", " is"]
   → [785,   6722,       315,   9625,      374]
```

Qwen3's dictionary has **151,936 entries**, fixed at training time and never changed.

### Things beginners get wrong about tokens

**Tokens are not words.** Common words are usually one token. Rare words split up:
`"morphogenesis"` might become `"morph"` + `"ogenesis"`. Leading spaces are part of the
token — `"France"` and `" France"` are *different* entries.

**The model never sees letters.** This is why language models are famously bad at
"how many r's in strawberry" — they see two or three opaque integers, not letters.

**Vocabulary size is a trade-off.** A bigger dictionary means fewer tokens per sentence,
so generation is cheaper and context stretches further. But the lookup table grows
proportionally. GPT-2 used 50,257. Qwen3 uses 151,936 because it must cover English,
Chinese, code, and emoji.

### Where it shows up

`vocab_size` appears at both ends of the pipeline:

- **input:** `model.embed_tokens.weight`, shape `[151936, 1024]` — the lookup table
- **output:** `lm_head.weight`, shape `[151936, 1024]` — one score per possible next token

---

## `hidden_size = 1024`

**The width of the residual stream. The model's most fundamental dimension.**

Every token becomes a vector of 1024 floats:

```
" France"  →  [0.031, -1.204, 0.885, 0.442, ..., 0.017]
               └──────────── 1024 numbers ────────────┘
```

This is the whiteboard from Part 1. Everything in the model is measured against this
number — projections start from it and return to it.

`hidden_size` is the model's **width**, and width is roughly "how much can be held in mind
at one position." 1024 is small by modern standards; GPT-3 used 12,288.

> **Naming note.** You'll see this called `hidden_size`, `d_model`, `n_embd`, or
> `embedding dimension` depending on the paper or codebase. All the same thing.

---

## `num_hidden_layers = 28`

**How many times the refinement block repeats.**

The block from Part 1 runs 28 times. Critically, **each layer has its own separate
weights** — same operations, entirely different numbers. Layer 0 and layer 27 share
nothing but structure.

A rough (and genuinely imperfect) intuition for what depth buys you:

| layers | loosely responsible for |
|---|---|
| early | surface patterns, grammar, "what token is this" |
| middle | assembling facts, relationships, tracking entities |
| late | committing to a prediction |

Real models are far messier than this story. Treat it as a starting frame, not truth.

### Why this number matters most to *us*

28 layers means **28 identically-shaped chunks of 31.46 MB**. That regular structure is
what makes disk streaming possible at all:

```
read layer 0 (31.46 MB) → compute → discard
read layer 1 (31.46 MB) → compute → discard
...
read layer 27           → compute → discard
```

Peak memory is one layer, not the whole model. That single sentence is the Morphogenesis
thesis.

---

## The attention trio

These three fields work together and must be understood together.

```
num_attention_heads = 16
head_dim            = 128
num_key_value_heads = 8
```

### First: how attention works

Attention is the only place information moves *between* token positions. Each position
produces three vectors from its residual stream:

| vector | made by | meaning |
|---|---|---|
| **Query** (Q) | `q_proj` | "what am I looking for?" |
| **Key** (K) | `k_proj` | "what do I have to offer?" |
| **Value** (V) | `v_proj` | "here's what you get if you pick me" |

The mechanism, for one position:

1. Dot-product my Query against every earlier position's Key. High result = good match.
2. Push those scores through **softmax**, so they become weights that sum to 1.
3. Take that weighted average of the Values.
4. Add the result to my residual stream.

Worked example — position 4 (`" is"`) looking back:

```
   Q₄ · K₀ = 0.1     "The"
   Q₄ · K₁ = 0.3     " capital"
   Q₄ · K₂ = 0.1     " of"
   Q₄ · K₃ = 2.8     " France"    ← strong match
   Q₄ · K₄ = 0.2     " is"
            │
            ▼ softmax
   [0.03, 0.05, 0.03, 0.85, 0.04]
            │
            ▼ weighted sum of Values
   ≈ 0.85 × V₃  →  mostly France's Value vector
```

Note it only looks at positions **0 through 4**, never 5 and beyond. That's **causal
masking**: a position can never see the future. At generation time the future doesn't
exist yet, so training must match that constraint.

There's also a scaling factor: scores are divided by `√head_dim` before softmax, to stop
the dot products growing large enough to make softmax saturate.

### `num_attention_heads = 16`

Doing that lookup once is limiting — a position often needs several unrelated things at
once. So the model runs the whole mechanism **16 times in parallel**, each with its own
learned Q/K/V projections. One head might track the grammatical subject, another the most
recent noun, another simply attend to the previous token.

These parallel copies are called **heads**. Their outputs are concatenated and mixed back
together by `o_proj`.

### `head_dim = 128` — and the trap

Each head works in its own **128-dimensional** subspace.

Now the part that catches almost everyone. In most models, `head_dim` is *derived*:

```
head_dim = hidden_size / num_attention_heads = 1024 / 16 = 64
```

**Qwen3 does not do this.** It explicitly sets `head_dim = 128`. So:

```
16 heads × 128 dims = 2048        but hidden_size is only 1024
```

Attention gets a workspace **twice as wide** as the residual stream. You can see it
directly in the weight shapes:

```
q_proj:  [2048, 1024]     projects UP:    1024 → 2048
o_proj:  [1024, 2048]     projects DOWN:  2048 → 1024
```

> **⚠ Failure mode.** If you assume `head_dim = 64`, every shape in your code still lines
> up perfectly and nothing crashes. The model just produces confident, fluent nonsense.
> Always read `head_dim` from the config. Never derive it.

### `num_key_value_heads = 8` — Grouped Query Attention

Here's the problem this solves.

When generating one token at a time, you must **remember** the Key and Value vectors of
every previous token — otherwise you'd recompute the entire conversation on every single
step. That memory is the **KV cache**, and it grows linearly with conversation length. In
long chats it becomes the dominant memory cost, larger than the model itself.

**Grouped Query Attention (GQA)** shrinks it. Keep all 16 Query heads, but compute only
8 Key/Value heads. Query heads pair up and share:

```
Q heads:    0   1     2   3     4   5    ...   14  15
             \ /       \ /       \ /            \ /
KV heads:     0         1         2       ...     7
```

Half the cache, almost no quality loss. Three named variants you'll encounter:

| `num_key_value_heads` | name | trade-off |
|---|---|---|
| = `num_attention_heads` (16) | Multi-Head Attention (MHA) | best quality, biggest cache |
| between (8) | **Grouped Query (GQA)** ← Qwen3 | the modern default |
| = 1 | Multi-Query (MQA) | smallest cache, some quality loss |

Visible in the weights: `k_proj` and `v_proj` are `[1024, 1024]` (8 heads × 128), while
`q_proj` is `[2048, 1024]` (16 heads × 128). GQA saves us 4 MB of reading per layer.

### One Qwen3 extra: QK-Norm

Qwen3 adds two small tensors that Llama and Qwen2 do not have:

```
self_attn.q_norm.weight   [128]
self_attn.k_norm.weight   [128]
```

These apply RMSNorm to **each head individually, across its 128 dimensions**, *after* the
projection and *before* RoPE. It stabilises training.

> **⚠ Failure mode.** Omit these and the model is *almost* right — subtly degraded output
> with no error message. The worst kind of bug. They're only 256 bytes each, so they're
> easy to overlook.

---

## `intermediate_size = 3072` — the MLP

**The width of the feed-forward network inside each layer.**

After attention has moved information between positions, each position independently runs
through a small network: expand 1024 → 3072, apply a nonlinearity, contract 3072 → 1024.

Why expand and then contract? The wide middle is *room to compute*. The common (and still
debated) intuition is that this is where factual knowledge is stored — attention retrieves
context, the MLP knows things.

### SwiGLU

Qwen3 uses a **gated** MLP. Instead of one up-projection there are two, and they multiply:

```
gate = gate_proj(x)                        1024 → 3072
up   = up_proj(x)                          1024 → 3072
out  = down_proj( SiLU(gate) × up )        3072 → 1024
                              ↑
                    element-wise multiply
```

That multiply is a **gate**: one branch decides, value by value, how much of the other
branch gets through. It's a learned filter.

The nonlinearity is **SiLU** (also called Swish):

```
SiLU(x) = x · sigmoid(x)
```

A smooth version of "keep positive values, suppress negative ones". Smoothness helps
gradients during training.

### The cost

| tensor | shape | bytes (bf16) |
|---|---|---|
| `mlp.gate_proj` | [3072, 1024] | 6,291,456 |
| `mlp.up_proj` | [3072, 1024] | 6,291,456 |
| `mlp.down_proj` | [1024, 3072] | 6,291,456 |
| **total** | | **18,874,368** |

That's **60% of every layer's weights**. On a bandwidth-bound engine, the MLP *is* the bill.

---

## `rms_norm_eps = 1e-06`

**A tiny constant that prevents division by zero.**

Stacking 28 layers that each add to a vector risks the numbers exploding or collapsing.
So before each sub-layer reads the residual stream, the vector is rescaled to a consistent
magnitude.

**RMSNorm** divides by the root-mean-square, then applies a learned per-dimension scale:

```
              x_i
y_i =  ───────────────────── × g_i
        √( mean(x²) + eps )
```

- `mean(x²)` — average of the squared values across all 1024 dimensions
- `eps = 1e-6` — added so you never divide by zero if the vector is all zeros.
  **That is its entire job.** Nothing deeper.
- `g_i` — a *learned* scale, one per dimension. These are the `[1024]` tensors in the file
  (`input_layernorm.weight`, `post_attention_layernorm.weight`).

### RMSNorm vs LayerNorm

Older models used **LayerNorm**, which also subtracts the mean and adds a bias:

```
LayerNorm:  (x - mean) / std × g + b
RMSNorm:    x / rms(x) × g
```

RMSNorm drops the mean subtraction and the bias. It's cheaper and works just as well, so
essentially every modern model uses it. Don't accidentally implement LayerNorm.

### Where norms sit

Qwen3 is **pre-norm**: normalise *before* each sub-layer, never after.

```
x = x + attention(RMSNorm(x))      ← input_layernorm
x = x + mlp(RMSNorm(x))            ← post_attention_layernorm
```

Note that `post_attention_layernorm` is a confusing name — it does not normalise the
attention output. It's the norm that runs *after the attention block has finished*,
feeding the MLP. Blame the original authors.

---

## `rope_theta = 1000000` — position

**The base frequency for rotary position encoding.**

Go back and re-read the attention mechanism. It's a weighted sum — **it has no notion of
order**. Shuffle the input tokens and the outputs shuffle identically. `"dog bites man"`
and `"man bites dog"` would be indistinguishable to raw attention.

So position must be injected deliberately.

### How RoPE works

**RoPE** (Rotary Position Embedding) *rotates* the Q and K vectors by an angle proportional
to their position. Take the 128 dimensions as 64 `(x, y)` pairs, and rotate pair *j* at
position *m* by angle `m · θⱼ`:

```
θⱼ = 1 / rope_theta^(2j/128)

pair 0  → θ ≈ 1.0      fast rotation   → encodes fine, local position
pair 63 → θ ≈ 1e-6     slow rotation   → encodes coarse, long-range position
```

It's a ladder of clock hands spinning at wildly different speeds — like the hands of a
clock encoding seconds, minutes, and hours together.

### Why rotation instead of addition

This is the elegant part. Because rotation composes by *adding angles*, after rotating,
the dot product between a Query at position *m* and a Key at position *n* depends **only
on `m − n`** — their relative distance.

```
Q_m · K_n   depends only on   (m − n)
```

The model learns "three tokens back", not "at absolute position 47". That generalises far
better to lengths never seen in training.

### Why 1,000,000

The original RoPE paper used `rope_theta = 10000`. Qwen3 uses **1,000,000**.

A bigger base means slower rotations and longer wavelengths, so distant positions stay
distinguishable instead of the angles wrapping around and aliasing. This is the standard
modern trick for long context, and it's how Qwen3 supports `max_position_embeddings` of
**40,960** tokens.

### Two implementation notes

- RoPE is applied to **Q and K only** — never to V. Position affects *who attends to whom*,
  not *what information gets passed*.
- It's applied **after** the projection and (in Qwen3) after QK-Norm. Order matters.

---

## `tie_word_embeddings = true`

**Use the same matrix for the input lookup and the output scoring.**

There are two places the model converts between token IDs and vectors:

```
input:   embedding table   token id      → 1024-vector    [151936, 1024]
output:  lm_head           1024-vector   → 151936 scores  [151936, 1024]
```

Identical shapes. **Tying** means using literally the same matrix for both.

The intuition is clean: if token #785 is *embedded* as vector **v**, then a hidden state
pointing in direction **v** should *score highly* for token #785. Input and output are
inverse operations, so share the weights.

This saves **155,582,464 parameters — 26% of the entire model.** Very common in small
models, where the vocabulary table would otherwise dominate the parameter count.

### A quirk in our actual file

Despite `tie_word_embeddings: true`, the safetensors file stores the matrix **twice**:

```
lm_head.weight             [151936, 1024]   311,164,928 B   @ offset 0
model.embed_tokens.weight  [151936, 1024]   311,164,928 B   @ offset 311,164,928
```

That's **311 MB of pure duplication — 21% of the file.** The exporter simply wrote both.
Your loader should check the tie flag and keep one copy, which drops the model from
1.40 GiB to 1.11 GiB before any real optimisation.

---

# Part 3 — Why This Matters for Morphogenesis

Morphogenesis serves the model **from disk**, so every field above translates into bytes
that must physically move.

## The core equation

Decoding one token reads each weight **exactly once** and uses it for a single
multiply-accumulate against a vector — roughly **2 FLOPs per byte loaded**. That's terrible
arithmetic intensity, which means the CPU is never the bottleneck:

```
time_per_token  ≈  bytes_of_weights / bandwidth_of_the_tier_they_live_in
```

## Where the bytes go, per token

| component | formula | bytes | share |
|---|---|---:|---:|
| MLP | 28 × 18.87 MB | 528.5 MB | 44.3% |
| attention | 28 × 12.58 MB | 352.3 MB | 29.6% |
| `lm_head` | 151936 × 1024 × 2 | 311.2 MB | 26.1% |
| norms + QK-norms | 28 × 4,608 B | 0.13 MB | 0.01% |
| embedding lookup | 1 row × 1024 × 2 | 2 KB | ~0% |
| **total** | | **~1.19 GB** | |

Two things should jump out:

**The embedding table is 311 MB but you read only 2 KB of it.** It's a *lookup* — one row
per token — not a matrix multiply. But that's a *random* 2 KB read, which on a spinning
disk costs a full ~10 ms seek to fetch two kilobytes.

**`lm_head` alone is 26% of the per-token cost.** One matrix, read in full, every single
token, purely to score 151,936 candidates so you can pick one.

## What that predicts on this machine

| tier | bandwidth | s/token | tok/s |
|---|---|---:|---:|
| RAM | ~40 GB/s | 0.03 | ~33 |
| NVMe (970 EVO) | ~3 GB/s | 0.40 | ~2.5 |
| HDD (7200rpm) | ~150 MB/s | 8.0 | ~0.125 |

Same code, same model, **266× spread**. Which is the entire point of the project.

## Per-layer budget

| tensor | shape | bytes |
|---|---|---:|
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
| **total per layer** | | **31,461,888** |

28 layers × 31.46 MB = **880.9 MB**. Stream one at a time and peak memory is 31 MB, not
1.1 GB.

---

# Glossary

| term | meaning |
|---|---|
| **autoregressive** | generates one token at a time, feeding each output back as input |
| **causal mask** | a position may only attend to earlier positions, never later ones |
| **embedding** | the vector representation of a token |
| **GQA** | Grouped Query Attention — fewer K/V heads than Q heads, to shrink the KV cache |
| **head** | one parallel copy of the attention mechanism |
| **KV cache** | stored Keys and Values from previous tokens, so they aren't recomputed |
| **logits** | the raw output scores, one per vocabulary entry, before softmax |
| **MLP / FFN** | the per-position feed-forward network inside each layer |
| **pre-norm** | normalise before each sub-layer (modern standard) rather than after |
| **residual stream** | the 1024-number vector carried through all layers, added to by each |
| **RMSNorm** | normalisation by root-mean-square, no mean subtraction, no bias |
| **RoPE** | Rotary Position Embedding — encodes position by rotating Q and K |
| **softmax** | turns a list of scores into probabilities summing to 1 |
| **SwiGLU** | gated MLP where one branch filters the other via element-wise multiply |
| **token** | a word fragment; the atomic unit the model processes |
| **weight tying** | sharing one matrix between the input embedding and the output head |

---

# Traps Checklist

Keep this next to you when you implement the forward pass. Every one of these fails
*silently* — no crash, no error, just degraded or nonsense output.

- [ ] `head_dim` is **128**, read from config. Do **not** compute `hidden_size / n_heads`.
- [ ] **QK-Norm** exists. Apply `q_norm`/`k_norm` per head, after projection, before RoPE.
- [ ] Use **RMSNorm**, not LayerNorm. No mean subtraction, no bias.
- [ ] Q has **16** heads, K and V have **8**. Repeat each KV head twice to match.
- [ ] RoPE applies to **Q and K only**, never V.
- [ ] `rope_theta` is **1e6**, not the more common 1e4.
- [ ] Scale attention scores by `1/√head_dim` = `1/√128` before softmax.
- [ ] Apply the **causal mask** — no peeking at future positions.
- [ ] Embeddings are **tied**; the file stores the matrix twice.
- [ ] Weights are stored `[out_features, in_features]`, i.e. **already transposed**
      relative to the maths you'd write on paper.

---

**Next:** `02-bf16-and-safetensors.md` — how those 2-byte floats work, and how to find any
tensor in a 1.4 GB file without reading it.
