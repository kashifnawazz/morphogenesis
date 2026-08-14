# 03 — bf16 and Numbers

**How floating point actually works, why ML picked bf16, and why converting it to f32
is a bit-shift rather than arithmetic.**

Guide 01 told you what the weights mean. Guide 02 told you where they live. This one is
about the two bytes themselves — because you're about to write code that turns
1,503,300,328 bytes into 751 million numbers, and if you get this wrong nothing downstream
can possibly work.

Every example below is decoded from real bytes in `models/qwen3-0.6b/model.safetensors`.

---

# Part 1 — The Concept

## The problem floats solve

Suppose you have 16 bits and you want to store numbers. The obvious approach is **fixed
point**: pick a scale, say "the integer *n* means *n* / 1000", and you can represent
−32.768 through +32.767 in steps of 0.001.

Now look at actual weights from your model:

```
model.layers.0.self_attn.q_proj.weight  →  |w| ranges from 0.0000238 to 0.0747
model.norm.weight                       →  values up to 15.3125
```

That's a span of roughly **640,000×** between the smallest and largest magnitude in a
single model. Fixed point at 0.001 resolution rounds every single `q_proj` weight to zero.
Fixed point fine enough for `q_proj` overflows on `model.norm`.

**Floating point solves this by spending some of its bits on a scale factor** that moves
with the value. Instead of a fixed grid, you get a grid that stretches — dense near zero,
coarse far from it. Precision becomes *relative* rather than absolute.

## Scientific notation, in binary

You already know this idea:

```
        6.022 × 10²³
        └─┬─┘   └┬┘
      significand exponent
```

Floating point is the same thing in base 2:

```
   value = (−1)^sign × 2^exponent × 1.mantissa
             │            │             │
          1 bit      "where"        "which value there"
```

Three fields packed into a fixed number of bits:

```
┌───┬─────────────┬──────────────────────────┐
│ s │  exponent   │        mantissa          │
└───┴─────────────┴──────────────────────────┘
  1       E bits            M bits
```

**The exponent controls range. The mantissa controls precision.** How you split your bit
budget between them is the *entire* design decision, and it's why there are several
16-bit float formats rather than one.

## Two conventions you need

**The bias.** The exponent field is unsigned, but exponents must go negative (for values
below 1). So a fixed **bias** is subtracted: if the field holds 124 and the bias is 127,
the real exponent is −3. This lets a plain unsigned integer comparison also order the
floats correctly, which is a genuinely clever trick.

**The implicit leading 1.** In binary scientific notation, a normalised significand always
starts `1.something` — because if it started with 0 you'd shift the exponent. Since that
leading bit is *always* 1, it isn't stored. You get one bit of precision for free. This is
why a 7-bit mantissa gives you 8 bits of significand.

---

# Part 2 — The Formats

## f32 — the reference

```
┌───┬──────────┬───────────────────────────┐
│ s │ exponent │        mantissa           │   32 bits
└───┴──────────┴───────────────────────────┘
  1      8                 23
```

Bias 127. Significand 24 bits (23 stored + 1 implicit). Range roughly 1.2e−38 to 3.4e38,
about **7.2 decimal digits** of precision.

This is what we compute in. Not what we store.

## f16 — half precision

```
┌───┬───────┬──────────────┐
│ s │  exp  │   mantissa   │   16 bits
└───┴───────┴──────────────┘
  1     5           10
```

Bias 15. Significand 11 bits, about **3.3 decimal digits**.

The obvious way to halve f32 — take bits from both fields. And it was the standard for
years, because it's what graphics hardware supported.

**But look at the range it buys:** 5 exponent bits means the largest value is 65,504 and
the smallest normal value is 6.10e−5.

Now recall the real weights I measured from your file:

```
q_proj first 2048 weights:  |w| range 2.384e−05 .. 7.471e−02
                                      └────┬────┘
                            below f16's smallest normal value
```

**Real weights in this model fall below f16's normal range.** They'd be stored as
*subnormals* — representable, but with progressively fewer significant bits as they shrink.
And during training, gradients are far smaller than weights; whole tensors would flush to
zero. Training in f16 requires "loss scaling" — multiplying the loss by a large constant
purely to drag gradients back into range. It's a workaround for a format that doesn't fit
the problem.

## bf16 — brain float

```
┌───┬──────────┬─────────┐
│ s │ exponent │mantissa │   16 bits
└───┴──────────┴─────────┘
  1      8          7
```

Bias 127. Significand 8 bits, about **2.4 decimal digits**.

Here's the move: **bf16 keeps all 8 of f32's exponent bits** and pays for it entirely out of
the mantissa. So bf16 has *exactly the same range as f32* — 1.2e−38 to 3.4e38. It simply
carries fewer significant digits.

That is the whole insight, and it turned out to be the right trade for neural networks:

> **Networks care enormously about range and surprisingly little about precision.**

Weights and gradients span many orders of magnitude, so you must not overflow or flush to
zero. But each individual weight only needs a few significant digits — the network is
averaging over millions of them, and training is stochastic anyway. Noise in the low bits
is, roughly, just more noise in a process already full of it.

No loss scaling. No overflow. Drop-in for f32's range.

## Decoding real weights

Let's take an actual value from `model.layers.0.input_layernorm.weight`. The first two
bytes in the file are `0b 3e`. Little-endian, so the 16-bit value is `0x3e0b`:

```
0x3e0b  =  0 01111100 0001011
           │ └──┬───┘ └──┬──┘
           s    e=124    m=11

value = (−1)⁰ × 2^(124−127) × (1 + 11/128)
      = 1 × 2^−3 × 1.0859375
      = 0.125 × 1.0859375
      = 0.1357421875
```

And that is exactly what the file yields: `+0.13574219`.

More real values, decoded from your model:

```
model.layers.0.input_layernorm.weight
  0x3e0b   s=0 e=124 (2^-3)  m= 11   = +0.13574219
  0x3f35   s=0 e=126 (2^-1)  m= 53   = +0.70703125
  0x3f17   s=0 e=126 (2^-1)  m= 23   = +0.58984375

model.layers.0.self_attn.q_proj.weight
  0x3b5f   s=0 e=118 (2^-9)  m= 95   = +0.00340271
  0xbb63   s=1 e=118 (2^-9)  m= 99   = -0.0034637451
  0xbc50   s=1 e=120 (2^-7)  m= 80   = -0.012695312

model.norm.weight
  0x407c   s=0 e=128 (2^ 1)  m=124   = +3.9375
  0x4175   s=0 e=130 (2^ 3)  m=117   = +15.3125
```

Notice the norm weights are order 1–15 while the projection weights are order 0.003. Same
format, six orders of magnitude apart, no scale factor anywhere. That's what floating point
buys you.

## The comparison

| | bits | s / e / m | bias | significand | range | ≈ decimal digits |
|---|---:|---|---:|---:|---|---:|
| **f32** | 32 | 1/8/23 | 127 | 24 | 1.2e−38 … 3.4e38 | 7.2 |
| **bf16** | 16 | 1/**8**/7 | 127 | 8 | **1.2e−38 … 3.4e38** | 2.4 |
| **f16** | 16 | 1/5/10 | 15 | 11 | 6.1e−5 … 65504 | 3.3 |
| **f8 E4M3** | 8 | 1/4/3 | 7 | 4 | 1.95e−3 … 448 | 0.9 |
| **f8 E5M2** | 8 | 1/5/2 | 15 | 3 | 6.1e−5 … 57344 | 0.6 |

Read the bf16 row against the f32 row. **Identical range, one third the significand.**
That's the entire design.

## Special values

Reserved exponent patterns, consistent across all IEEE-style formats:

| exponent field | mantissa | meaning |
|---|---|---|
| all zeros | zero | **±0** (yes, there are two zeros) |
| all zeros | non-zero | **subnormal** — no implicit leading 1, gradual underflow |
| all ones | zero | **±infinity** |
| all ones | non-zero | **NaN** — not a number |

Two properties that will eventually bite you:

- `NaN != NaN` is **true**. NaN compares unequal to everything, including itself. This is
  actually the cheapest way to detect one.
- Once a NaN enters your computation it propagates through every subsequent operation. One
  bad weight silently poisons an entire forward pass. When debugging, check for NaN *early*
  and *often*.

---

# Part 3 — The Conversion

## Why bf16 → f32 is a shift

Line the two formats up:

```
f32:   ┌───┬──────────┬───────────────────────────────────────┐
       │ s │ exponent │             mantissa (23)             │
       └───┴──────────┴───────────────────────────────────────┘
        31   30 ... 23   22 ........................... 0

bf16:  ┌───┬──────────┬─────────┐
       │ s │ exponent │ mant(7) │
       └───┴──────────┴─────────┘
        15   14 ...  7   6 ... 0
```

Three facts stack up:

1. Both have **1 sign bit** in the top position.
2. Both have **8 exponent bits**, with **the same bias of 127**.
3. bf16's 7 mantissa bits are **the top 7 of f32's 23**.

So the bit patterns are *already identical* — bf16 is just f32 with the bottom 16 bits
chopped off. To widen it, you put those 16 zero bits back:

```
bf16:                    0011111000001011
f32:     0011111000001011 0000000000000000
         └── unchanged ─┘ └─ 16 zeros ──┘
```

```
f32_bits = (u32)bf16_bits << 16
```

**No exponent rebiasing. No mantissa renormalisation. No lookup table. No branches.**
One shift, and it's exact — every bf16 value has a precise f32 representation, including
subnormals, infinities, and NaN. Compare that to f16 → f32, which genuinely requires
rebiasing the exponent from 15 to 127 and handling subnormals as a special case.

This is a large part of why bf16 won. The conversion is nearly free, which matters a great
deal when you're doing it 596 million times per token.

> **Your Phase 0 exercise.** You now know the mechanism completely. The numpy expression is
> a one-liner: view the bytes as `uint16`, widen to `uint32`, shift, view as `float32`.
> Work out the incantation yourself — the value is in wiring the bit-level picture above to
> code you typed. Watch for two things: **endianness** (safetensors is little-endian) and
> the fact that `.view()` reinterprets bits while `.astype()` converts values. Confusing
> those two produces garbage that looks almost plausible.

## Going the other way: f32 → bf16

You'll need this in Phase 7 when you quantise. It's harder, because now you're *discarding*
information and have to decide how.

**Truncation** — just drop the low 16 bits. Fast, and biased: it always rounds toward zero.
Over millions of weights that bias accumulates into systematic drift.

**Round-to-nearest-even** — the IEEE default, and what PyTorch does. Add a rounding term
before truncating, breaking exact ties toward an even last bit so errors cancel rather than
accumulate:

```
rounding_bias = 0x7FFF + ((f32_bits >> 16) & 1)
bf16_bits     = (f32_bits + rounding_bias) >> 16
```

If you ever compare your quantiser against PyTorch's and see a consistent one-bit
disagreement, this is why.

---

# Part 4 — Precision in Practice

## What 8 bits of significand actually costs

**Machine epsilon** is the gap between 1.0 and the next representable number — the format's
relative resolution:

| format | epsilon | meaning |
|---|---|---|
| f32 | 2⁻²³ ≈ 1.19e−7 | ~7 good digits |
| f16 | 2⁻¹⁰ ≈ 9.77e−4 | ~3 good digits |
| **bf16** | **2⁻⁸ ≈ 3.91e−3** | **~2.4 good digits** |

Sit with that bf16 number. Near 1.0, **consecutive bf16 values are 0.4% apart.** You cannot
represent 1.001. You cannot represent 1.002. The nearest neighbours of 1.0 are 0.99609375
and 1.00390625.

A concrete consequence from guide 01. Recall `rms_norm_eps = 1e-6`:

```
in f32:    1.0 + 1e-6  =  1.000001      ✓ representable
in bf16:   1.0 + 1e-6  =  1.0           ✗ the epsilon vanishes entirely
```

**`rms_norm_eps` is three orders of magnitude below bf16's resolution near 1.0.** If you
computed RMSNorm in bf16, the epsilon term would do literally nothing. This isn't a
hypothetical — it's a direct instruction about how to write your kernel.

## The golden rule

> **Store in bf16. Compute in f32.**

Every serious inference engine does this, and here's the arithmetic reason.

A single dot product in this model sums 1024 terms. Rounding errors in a sum accumulate
roughly as `√n × ε`:

```
in bf16:  √1024 × 3.91e−3  =  32 × 3.91e−3  ≈  0.125     → 12% error. Useless.
in f32:   √1024 × 1.19e−7  =  32 × 1.19e−7  ≈  3.8e−6    → fine.
```

So: read bf16 from disk, widen to f32 immediately, accumulate in f32, and only narrow back
if you're writing to storage. The *storage* format is bf16; the *compute* format is f32.
Keeping these separate in your head prevents a whole class of bug.

This also explains why bf16 is safe at all. Each weight individually carries only 2.4
digits, but you're summing 1024 of them in full precision — the errors are independent and
partially cancel, rather than compounding.

## Catastrophic cancellation

The nastiest floating-point failure. Subtracting two nearly-equal numbers annihilates the
leading digits and promotes rounding noise into the most significant position:

```
1.0000001  −  1.0000000  =  0.0000001
└─ 7 good digits ─┘          └─ maybe 1 good digit ─┘
```

Where you'll meet it: **softmax**. Attention scores get exponentiated, and `exp` of a
moderately large number overflows fast. The standard fix is to subtract the maximum first:

```
naive:   exp(x_i) / Σ exp(x_j)              overflows when x is large
stable:  exp(x_i − max) / Σ exp(x_j − max)  mathematically identical, always safe
```

Subtracting the max makes the largest exponent exactly `exp(0) = 1`, so nothing overflows,
and the ratio is unchanged. **Always write softmax this way.** Every real implementation
does, and it's the single most common source of NaN in a hand-written transformer.

## Summation order matters

Floating-point addition is **not associative**:

```
(a + b) + c  ≠  a + (b + c)
```

This is why your results will differ slightly from PyTorch's even when your code is
perfectly correct — different summation orders, different rounding. It's also why parallel
reductions with `rayon` give slightly different answers run to run if the chunking varies.

Practical consequences:

- Don't chase bit-exactness with the reference. Chase **tolerance**.
- If you want reproducible output, fix your reduction order deterministically.
- For long sums, consider **Kahan summation** — it tracks the lost low-order bits in a
  compensation variable. Probably overkill here, worth knowing it exists.

---

# Part 5 — For Morphogenesis

## Tolerance targets

From the roadmap's validation gates, now with justification:

| stage | tolerance | why |
|---|---|---|
| individual kernels | **< 1e−4** relative | well inside f32 accumulation error; anything worse is a real bug |
| full 28-layer logits | **< 1e−3** relative | 28 layers of accumulated reordering |
| argmax token | **exact** | the actual thing that matters |

That last row is the one to care about. Logits differing by 1e−4 is fine. Picking a
different token is not. **If your argmax matches the reference across a few hundred tokens,
your engine is correct** — regardless of low-bit noise.

## Why this decides our data flow

```
DISK              bf16, 2 bytes/weight       ← the bandwidth that defines our speed
  │
  │  widen: one 16-bit shift, free
  ▼
COMPUTE           f32, 4 bytes               ← accumulate here, always
  │
  ▼
KV CACHE          f32 (or bf16 later)        ← a size/precision trade for Phase 5
```

**Note what we never do: hold the whole model as f32.** That would be 2.4 GB and defeats the
project. We widen one tile at a time, in registers, and throw it away. The f32 version of
any weight exists for microseconds.

## The bandwidth connection

Guide 02's core equation, restated with what you now know:

```
time_per_token ≈ bytes_of_weights / bandwidth
```

The choice of storage format is a *direct* multiplier on speed:

| storage | bytes/weight | bytes/token | HDD s/token |
|---|---:|---:|---:|
| f32 | 4 | 2.38 GB | ~16.0 |
| **bf16** | **2** | **1.19 GB** | **~8.0** |
| Q8_0 | 1.06 | 0.63 GB | ~4.2 |
| Q4_0 | 0.56 | 0.33 GB | ~2.2 |

Every halving of the storage format nearly halves your token latency, because you are
bandwidth-bound and nothing else changes. This is the whole argument for Phase 7 — and
it's why "store small, compute wide" isn't just a precision principle, it's the performance
strategy.

## Looking ahead to Q8_0

Guide 02 introduced block quantisation. Now you can see it as *building your own float
format*:

```
Q8_0 block:  [ f16 scale d ][ 32 × int8 ]   =  34 bytes for 32 weights
                    │
                    └── one shared exponent for all 32 values
```

That's the same idea as floating point — a significand and a scale — except the scale is
**shared across 32 weights** instead of carried per weight. You trade per-value range for a
much cheaper representation, and it works because neighbouring weights in a tensor have
similar magnitudes.

Look back at the measurement from your own file:

```
q_proj first 2048 weights:  |w| range 2.384e−05 .. 7.471e−02
```

Roughly 3000× spread across the whole tensor — far too wide for one shared scale. But
within any *32-weight block*, the spread is much smaller. **That's precisely why block
quantisation works, and why the block size is the central knob.**

---

# Glossary

| term | meaning |
|---|---|
| **bias** | constant subtracted from the stored exponent to allow negative exponents |
| **catastrophic cancellation** | precision destroyed by subtracting nearly-equal values |
| **denormal / subnormal** | values below the smallest normal, stored without the implicit leading 1 |
| **implicit leading 1** | the always-1 bit of a normalised significand, not stored |
| **machine epsilon** | gap between 1.0 and the next representable value |
| **mantissa** | the stored fraction bits (significand minus the implicit 1) |
| **NaN** | Not a Number; propagates through everything; unequal to itself |
| **normal** | a value with an implicit leading 1, i.e. exponent field not all-zero |
| **round-to-nearest-even** | IEEE default rounding; ties go to an even last bit |
| **significand** | full precision including the implicit bit — mantissa + 1 |

---

# Traps Checklist

- [ ] safetensors is **little-endian**. `0b 3e` on disk is the value `0x3e0b`.
- [ ] bf16 → f32 is `bits << 16`. If you're rebiasing exponents, you've confused it with f16.
- [ ] **Reinterpret** bits (`view`), don't **convert** values (`astype`). They differ silently.
- [ ] Accumulate dot products in **f32**, never bf16. `√1024 × 3.91e−3` ≈ 12% error.
- [ ] Compute RMSNorm in f32 — `rms_norm_eps = 1e-6` is invisible at bf16 resolution.
- [ ] Subtract the max before `exp` in softmax. Always.
- [ ] Check for NaN early; one poisons the whole forward pass.
- [ ] Compare against the reference with a **tolerance**, not equality. FP addition isn't
      associative, so your order will differ from PyTorch's.
- [ ] The gate that actually matters is **matching argmax**, not matching logits.

---

**Next:** Phase 0 — build the numpy reference implementation. You now have everything you
need: what the numbers mean (01), where they live in the file (02), and how to turn bytes
into floats (03).
