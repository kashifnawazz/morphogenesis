# 04 — The Tokenizer

**How text becomes numbers, and why it's stranger than you'd expect.**

Guides 01–03 covered the model. This one covers the thing that sits in front of it. The
tokenizer is not part of the neural network — it's a deterministic, hand-built algorithm
with no learned weights in the usual sense. It's also where a surprising number of "the
model is broken" bugs actually live.

Everything below was verified by implementing Qwen3's tokenizer from scratch against the
real files in `models/qwen3-0.6b/`. Every token ID in this guide is real.

---

# Part 1 — The Concept

## The problem

The model works on integers — an index into a table of 151,936 rows. So somebody has to
turn `"The capital of France is"` into a list of integers, and turn integers back into text.

The obvious approaches both fail:

**One token per character.** Vocabulary of ~150 for English. Beautifully simple, and it
means every sequence is *long* — "The capital of France is" becomes 24 tokens instead of 5.
Since cost scales with token count and attention scales with the square of it, you've made
the model ~5× more expensive and forced it to learn spelling before it can learn meaning.

**One token per word.** Short sequences, but the vocabulary is unbounded. New words,
typos, names, code identifiers, other languages — all become `<UNK>` and the information is
simply gone. You also can't tell the model that "run" and "running" are related.

## The compromise: subwords

Keep frequent words whole, split rare ones into pieces:

```
"The capital of France is"   →  ['The', 'Ġcapital', 'Ġof', 'ĠFrance', 'Ġis']     5 tokens
"morphogenesis"              →  ['morph', 'ogenesis']                             2 tokens
```

Common words cost one token. Rare words still get represented, from parts the model has
seen. Nothing is ever unknown.

Those are **real** splits from your model's tokenizer — `'morphogenesis'` genuinely becomes
`morph` + `ogenesis`, IDs `[89833, 51279]`.

## The four stages

Qwen3's tokenizer runs text through four transformations. Most confusion about tokenizers
comes from not knowing there are four:

```
   "The capital of France is"
            │
            │  ① NORMALISE     Unicode NFC
            ▼
   "The capital of France is"
            │
            │  ② PRE-TOKENISE  split on a regex; merges may never cross these boundaries
            ▼
   ['The', ' capital', ' of', ' France', ' is']
            │
            │  ③ BYTE-ENCODE   UTF-8 bytes → printable proxy characters
            ▼
   ['The', 'Ġcapital', 'Ġof', 'ĠFrance', 'Ġis']
            │
            │  ④ MERGE (BPE)   apply learned merge rules, then look up IDs
            ▼
   [785, 6722, 315, 9625, 374]
```

Stage ③ is where `Ġ` comes from, and it's the stage nobody explains.

---

# Part 2 — Byte-Level BPE, Stage by Stage

## ① Normalise — NFC

`tokenizer.json` says:

```json
"normalizer": { "type": "NFC" }
```

Unicode lets the same visible text be encoded multiple ways. "é" can be one code point
(U+00E9) or two (U+0065 + U+0301, "e" plus combining acute). **NFC** (Normalization Form
Canonical Composition) picks the composed form, so both spellings become identical bytes.

Without this, two visually identical prompts could tokenise differently. Cheap insurance.

## ② Pre-tokenise — the regex

This is the most under-appreciated stage. Before any merging, text is split by a regex, and
**BPE merges can never cross these boundaries**. The regex therefore decides which tokens
are even *possible*.

Qwen3's pattern, straight from `tokenizer.json`:

```
(?i:'s|'t|'re|'ve|'m|'ll|'d)      English contractions, case-insensitive
| [^\r\n\p{L}\p{N}]?\p{L}+        an optional leading symbol, then letters
| \p{N}                           exactly ONE digit
| ?[^\s\p{L}\p{N}]+[\r\n]*        optional space, then punctuation, then newlines
| \s*[\r\n]+                      whitespace ending in newlines
| \s+(?!\S)                       trailing whitespace
| \s+                             any remaining whitespace
```

`\p{L}` means "any Unicode letter", `\p{N}` "any Unicode number". Real behaviour, measured:

```
'The capital of France is'  →  ['The', ' capital', ' of', ' France', ' is']
"don't  stop"               →  ['don', "'t", ' ', ' stop']
'12345'                     →  ['1', '2', '3', '4', '5']
'def foo():\n    return 1'  →  ['def', ' foo', '():\n', '   ', ' return', ' ', '1']
```

Three things to notice:

**The leading space joins the *following* word.** `' capital'`, not `'capital '`. This is
why the vocabulary contains both `'hello'` (14990) and `'Ġhello'` (23811) as separate
entries — the same word at a word boundary versus mid-word.

**`\p{N}` matches exactly one digit.** So `'12345'` is five tokens, never `'123'` + `'45'`.
Qwen deliberately forbids multi-digit tokens; it makes arithmetic dramatically more reliable,
because the model always sees digits in a consistent, positional form. (GPT-2 didn't do
this, and its arithmetic was correspondingly awful.)

**Whitespace runs are handled carefully.** Note `"don't  stop"` splits the double space into
`' '` and `' stop'` — one space stands alone, the next attaches to the word.

## ③ Byte-encode — where `Ġ` comes from

Here's the trick that confuses everyone the first time.

**The goal:** never have an `<UNK>` token. Any input at all — any language, emoji, corrupt
bytes, binary — must be representable. The clean way is to work at the **byte** level: there
are only 256 possible bytes, and every possible input is a sequence of them. Perfect coverage
with a 256-symbol alphabet.

**The problem:** BPE implementations manipulate *strings*. If your alphabet includes raw
bytes like `0x20` (space), `0x0A` (newline), and `0x00` (null), you get whitespace and
control characters inside your merge tables — fragile, hard to debug, and easy to mangle in
a text file like `merges.txt`.

**The solution:** map each of the 256 byte values to a **printable, non-whitespace Unicode
character** used purely as a stand-in.

The rule: bytes that are already printable ASCII or printable Latin-1 (`33–126`, `161–172`,
`174–255`) map to themselves. The remaining 68 bytes (`0–32`, `127–160`, `173`) get pushed
up into the U+0100 block, in order:

```
byte   0 (0x00)  →  'Ā'  U+0100
byte  10 (0x0a)  →  'Ċ'  U+010A      ← newline
byte  32 (0x20)  →  'Ġ'  U+0120      ← space
byte  65 (0x41)  →  'A'  U+0041      ← printable, unchanged
byte  97 (0x61)  →  'a'  U+0061      ← printable, unchanged
byte 127 (0x7f)  →  'ġ'  U+0121
byte 173 (0xad)  →  'Ń'  U+0143
byte 255 (0xff)  →  'ÿ'  U+00FF      ← printable, unchanged
```

**`Ġ` is simply "space".** It is not a special marker, not a word-boundary symbol — it's
U+0120 standing in for byte 0x20 so that no actual whitespace appears in the merge tables.
`Ċ` is newline. Once you know that, vocabulary dumps stop looking like line noise.

I verified that the first 256 entries of `vocab.json` are exactly these 256 proxy characters.

This also explains a thing that looks broken but isn't:

```
'日本語'  →  pieces ['æĹ¥æľ¬', 'èªŀ']  →  ids [101059, 102819]  →  roundtrip OK
```

That "mojibake" is the UTF-8 bytes of the Japanese text shown through the proxy mapping.
`日本語` is 9 UTF-8 bytes; the tokenizer sees 9 proxy characters and merges them into 2
tokens. Decoding maps them back to bytes and the original text returns exactly.

## ④ Merge — BPE proper

**Byte Pair Encoding** is the actual algorithm. It has a training phase and an inference
phase; we only ever do inference.

**Training** (already done, by Qwen): start with every symbol separate. Count adjacent pairs
across a huge corpus. Merge the most frequent pair into a new symbol. Repeat 151,387 times,
recording each merge in order. That ordered list *is* `merges.txt`.

**Inference** (what we implement): for each pre-token, start with individual characters and
repeatedly apply the **lowest-ranked applicable merge** until none applies. Lower rank means
learned earlier, means more frequent.

The first lines of your `merges.txt`:

```
#version: 0.2
Ġ Ġ          ← rank 0: two spaces
ĠĠ ĠĠ        ← rank 1: four spaces
i n          ← rank 2
Ġ t          ← rank 3
ĠĠĠĠ ĠĠĠĠ    ← rank 4: eight spaces (indentation — this corpus had a lot of code)
e r          ← rank 5
```

Here is a real trace, `" morphogenesis"` merged step by step:

```
start:   ['Ġ','m','o','r','p','h','o','g','e','n','e','s','i','s']
step  1: merge 'e'+'n'          (rank 12)
step  2: merge 'o'+'r'          (rank 13)
step  3: merge 'i'+'s'          (rank 29)
step  4: merge 'e'+'s'          (rank 32)
step  5: merge 'Ġ'+'m'          (rank 40)
step  6: merge 'o'+'g'          (rank 282)
step  7: merge 'p'+'h'          (rank 503)
step  8: merge 'Ġm'+'or'        (rank 4057)
step  9: merge 'og'+'en'        (rank 11450)
step 10: merge 'es'+'is'        (rank 13518)
step 11: merge 'Ġmor'+'ph'      (rank 26095)
step 12: merge 'ogen'+'esis'    (rank 51023)
final:   ['Ġmorph', 'ogenesis']  →  [26351, 51279]
```

Notice it always takes the globally lowest rank available, not left-to-right. `'e'+'n'`
(rank 12) fires before `'o'+'r'` (rank 13) even though `'or'` appears earlier in the word.
**This ordering is the whole algorithm** — get it wrong and you'll produce valid-looking but
different tokens, which is the worst kind of tokenizer bug.

## A lovely piece of arithmetic

```
        256  byte tokens
 +  151,387  merge rules
 ─────────────
    151,643  =  exactly the number of entries in vocab.json  ✓
```

**Every vocabulary entry is either one of the 256 raw bytes or the result of exactly one
merge rule.** The vocab and the merge list are the same information viewed two ways —
which is why you can rebuild `vocab.json` from `merges.txt` alone.

---

# Part 3 — Qwen3's Tokenizer Specifically

## The files

| file | size | what it is |
|---|---:|---|
| `tokenizer.json` | 11.4 MB | **the complete tokenizer** — normalizer, regex, vocab, merges, special tokens. Modern, self-contained. |
| `vocab.json` | 2.8 MB | legacy: token string → id |
| `merges.txt` | 1.7 MB | legacy: the ordered merge rules |
| `tokenizer_config.json` | 9.7 KB | chat template, special token names, `model_max_length` |

`tokenizer.json` supersedes the legacy pair. Either works; the legacy files are easier to
parse if you're writing a loader from scratch.

## The vocabulary arithmetic

This is worth walking through, because the numbers don't line up with `config.json` and
that surprises people:

```
vocab.json entries        151,643      ids 0 .. 151,642
added special tokens         + 26      ids 151,643 .. 151,668
                        ──────────
reachable tokens          151,669
config.json vocab_size    151,936
                        ──────────
UNUSED embedding rows         267      546,816 bytes that are never read
```

**267 rows of the embedding matrix can never be produced by the tokenizer.** Why? Because
`151,936 = 128 × 1,187` — the vocabulary was padded up to a multiple of 128 so matrix
tiles align nicely on GPU. Half a megabyte of dead weight, deliberately.

For us that's mostly trivia, but it does mean: **size your embedding table from
`config.json`, not from the tokenizer.** They disagree, and the model file follows the
config.

## The 26 special tokens

| id | token | purpose |
|---|---|---|
| 151643 | `<\|endoftext\|>` | padding / document separator |
| 151644 | `<\|im_start\|>` | **ChatML** — begins a message |
| 151645 | `<\|im_end\|>` | **ChatML** — ends a message. This is the EOS. |
| 151646–151651 | `<\|object_ref_*\|>`, `<\|box_*\|>`, `<\|quad_*\|>` | grounding / bounding boxes |
| 151652–151656 | `<\|vision_*\|>`, `<\|image_pad\|>`, `<\|video_pad\|>` | inherited from Qwen-VL |
| 151657–151658 | `<tool_call>`, `</tool_call>` | function calling |
| 151659–151664 | `<\|fim_*\|>`, `<\|repo_name\|>`, `<\|file_sep\|>` | fill-in-the-middle, for code |
| 151665–151666 | `<tool_response>`, `</tool_response>` | function results |
| **151667–151668** | **`<think>`, `</think>`** | **Qwen3's reasoning mode** |

Two observations. The vision tokens exist in a text-only 0.6B model because the tokenizer is
shared across the whole Qwen family — harmless, but it explains the odd entries.

And `<think>`/`</think>` are real tokens, not text conventions. Qwen3 emits them to
delimit its chain of thought, which is why you'll see hybrid reasoning behaviour from a
600M-parameter model.

## Special tokens must bypass the regex

Critical implementation point. Special tokens are matched **as literal strings, before the
pre-tokenizer regex runs**. If you let `<|im_start|>` reach the regex it gets shredded into
`<`, `|`, `im`, `_`, `start`, `|`, `>` — the model sees punctuation soup instead of a
control token, and your chat formatting silently breaks.

The order is always: split out special tokens → regex the remaining text → byte-encode →
merge.

## ChatML — the chat template

`tokenizer_config.json` carries a Jinja template that formats conversations. Qwen uses
**ChatML**:

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant
```

The prompt ends right after `<|im_start|>assistant\n`, and the model generates from there
until it emits `<|im_end|>` (151645).

**The template is part of the model's interface, not decoration.** The model was trained on
exactly this layout; feed it raw text and quality drops noticeably. Qwen3's real template
also handles tool definitions and `<think>` blocks, which is why it's ~4 KB of Jinja.

Note the config quirk: `config.json` lists `bos_token_id: 151643`, but
`tokenizer_config.json` says `add_bos_token: false` and `bos_token: null`. **Qwen3 does not
prepend a BOS token.** Trust `add_bos_token`. Adding a spurious BOS is a classic
off-by-one-token bug that subtly degrades output.

---

# Part 4 — Decoding

Decoding reverses the pipeline:

```
ids  →  look up strings  →  concatenate  →  proxy chars back to bytes  →  UTF-8 decode
```

```
[785, 6722, 315, 9625, 374]
  → ['The','Ġcapital','Ġof','ĠFrance','Ġis']
  → 'TheĠcapitalĠofĠFranceĠis'
  → bytes: 54 68 65 20 63 61 70 ...
  → "The capital of France is"
```

## The streaming trap

**A single token can hold a fragment of a UTF-8 character.** Look again at the Japanese
example: `日本語` is 9 UTF-8 bytes split across 2 tokens, and the split does not fall on
character boundaries.

So if you decode each token independently as it's generated — which is exactly what a
streaming server does — you will sometimes produce invalid UTF-8 and emit `�`.

The fix: **decode to bytes, buffer, and only emit complete UTF-8 sequences.** Hold back any
trailing incomplete multi-byte sequence until the next token arrives. This will matter in
Phase 8; it's the single most common bug in hand-rolled streaming endpoints.

## Round-trip results

My from-scratch implementation, verified against the real files:

| input | tokens | ids | round-trip |
|---|---:|---|---|
| `'The capital of France is'` | 5 | `[785, 6722, 315, 9625, 374]` | ✓ |
| `' Paris'` | 1 | `[12095]` | ✓ |
| `'Hello, world!'` | 4 | `[9707, 11, 1879, 0]` | ✓ |
| `'12345'` | 5 | `[16, 17, 18, 19, 20]` | ✓ |
| `'morphogenesis'` | 2 | `[89833, 51279]` | ✓ |
| `'日本語'` | 2 | `[101059, 102819]` | ✓ |
| `'def foo():\n    return 1'` | 7 | `[750, 15229, 3932, 262, 470, 220, 16]` | ✓ |
| `'🦀'` | 1 | `[147579]` | ✓ |

Note `'morphogenesis'` → `[89833, 51279]` but `' morphogenesis'` → `[26351, 51279]`. The
leading space changes the *first* token entirely. Tokenization is not composable — you
cannot tokenize two strings separately and concatenate the IDs.

---

# Part 5 — Implementing It (Phase 4)

## Data structures

```
vocab      : HashMap<String, u32>     token string → id
inv_vocab  : Vec<String>              id → token string  (a Vec, ids are dense)
merges     : HashMap<(String,String), u32>   pair → rank
byte_enc   : [char; 256]              byte → proxy char
byte_dec   : HashMap<char, u8>        proxy char → byte
specials   : Vec<(String, u32)>       matched literally, before the regex
```

`inv_vocab` should be a `Vec`, not a `HashMap` — IDs are dense from 0, so indexing is free.

## Algorithm and complexity

The naive merge loop rescans every adjacent pair on every iteration: for a pre-token of *n*
symbols that's **O(n²)** pair lookups, or O(n³) if you're careless with string allocation.

In practice *n* is tiny — pre-tokens are usually under 15 characters — so naive is
genuinely fine, and you should write it first. If you later want speed:

- keep the symbols in a doubly-linked list so merging is O(1) and you only re-examine the
  two neighbouring pairs
- keep candidate merges in a binary heap keyed by rank
- cache results per pre-token string; natural text repeats words constantly

## Order of operations — get this right

```
1. split out special tokens (literal string match)
2. NFC normalise the rest
3. apply the pre-tokenizer regex
4. byte-encode each piece via the 256-entry table
5. BPE-merge within each piece, never across
6. look up ids
```

## Validation gate

The roadmap's Phase 4 gate: **round-trip encode/decode byte-identical on a corpus, matching
the reference.** Concretely, build a test set covering ASCII, whitespace runs, code with
indentation, CJK, emoji, and the special tokens — then assert your Rust IDs equal the
Python reference's IDs exactly.

Tokenizer bugs are unforgiving: one wrong ID and the model receives a different prompt than
you think it did. Debugging that from garbled output alone is miserable. Test this layer
properly and you'll never suspect it again.

> **You can defer this.** Phases 0–3 only need *some* token IDs, and you can hardcode ones
> the Python reference produced. Building the forward pass first is more motivating, and it
> means when you finally write the tokenizer you already have a working model to test it on.

---

# Traps Checklist

- [ ] `Ġ` is **space** (byte 0x20 → U+0120), `Ċ` is **newline**. Not special markers.
- [ ] Special tokens are matched **before** the regex, as literal strings.
- [ ] BPE merges **never cross pre-token boundaries**.
- [ ] Always apply the **lowest-rank** merge available, not the leftmost.
- [ ] Leading spaces attach to the **following** word — `'Ġhello'` ≠ `'hello'`.
- [ ] `\p{N}` matches **one digit**; multi-digit tokens don't exist in this vocab.
- [ ] **No BOS token.** `add_bos_token: false`, despite `config.json` naming a `bos_token_id`.
- [ ] EOS is `<|im_end|>` = **151645**, not `<|endoftext|>`.
- [ ] Size the embedding table from **`config.json`** (151,936), not the tokenizer (151,669).
- [ ] Tokenization isn't composable — never concatenate independently tokenized strings.
- [ ] When streaming, **buffer bytes** and emit only complete UTF-8 sequences.
- [ ] Use the **chat template**. Raw text degrades output.

---

# Glossary

| term | meaning |
|---|---|
| **BPE** | Byte Pair Encoding — repeatedly merge the most frequent adjacent pair |
| **byte-level** | the base alphabet is the 256 byte values, so nothing is ever unknown |
| **ChatML** | Qwen/OpenAI conversation format using `<\|im_start\|>` / `<\|im_end\|>` |
| **chat template** | Jinja template turning a message list into the exact string the model expects |
| **merge rank** | position in the ordered merge list; lower = learned earlier = higher priority |
| **NFC** | Unicode normalisation that composes characters into canonical form |
| **pre-tokenization** | regex split applied before BPE; merges may never cross its boundaries |
| **proxy character** | printable Unicode char standing in for a raw byte (`Ġ` = 0x20) |
| **special token** | a token with a control meaning, matched literally rather than merged |
| **subword** | a token between a character and a word in size |

---

**Next:** Phase 0 — the numpy reference. You now have the complete picture: what the numbers
mean (01), where they live (02), how bytes become floats (03), and how text becomes token
IDs (04).
