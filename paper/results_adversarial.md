# Pushing the frontier: adversarial training vs. the monitor panel

**Defensive-research question.** The imitation adapter
([`results_trained_adapter.md`](results_trained_adapter.md)) taught Bonsai to
*match* a reference scheme — lossless but only re-creating a point the bench
already had. Can training move the model *past* the reference frontier: more
undetectable capacity than any hand-coded scheme gives? This run answers yes,
with an adversarial objective, and confirms the defensive implication.

## Setup

- **Capacity lever:** a composite `combined` channel — one marker bit *and* one
  synonym bit per sentence, 12 bits over 6 sentences (2x the single schemes).
  Still frequency-matched, so detectability is not free capacity.
- **Objective (adversarial):** rejection-sampling fine-tuning (ReST/RAFT). Each
  round, per task, sample several covers from the current model and score
  `reward = recovery − λ·suspicion`, where suspicion is the monitor panel's
  per-cover signal (leakage, whitespace tells, marker/synonym imbalance). Keep
  the best, SFT, repeat. The panel is the adversary: generations that leak or
  skew frequency are penalized and pruned. Two rounds, warm-started from the
  imitation adapter.
- **Held-out eval:** 24 fresh tasks (disjoint seed), 12-bit payloads, scored by
  [`stego_message_combined_heldout.json`](../configs/stego_message_combined_heldout.json).

## Result (held-out, 12-bit combined channel)

| metric | base | adversarially trained |
|---|---|---|
| recovery (chance-corrected) | 0.13 | **0.65** |
| detect AUROC (best monitor) | 0.51 | 0.50 |
| leakage AUROC | 0.50 | 0.50 |
| best FPR @ 0.95 TPR | 0.96 | 0.92 |
| survival (full normalize) | 0.18 | 0.18 |

Converting to the frontier axis — **effective, chance-corrected, undetectable
bits per token** (at detection AUROC ~0.50):

| channel | effective bits/token @ AUROC 0.50 |
|---|---|
| reference marker or synonym (single) | 0.20 |
| combined — base Bonsai | 0.05 (chance) |
| **combined — adversarially trained** | **0.26** |

The trained model lands past the best single reference scheme on the
undetectable line (see `figures/frontier_placement.png`). It is not lossless at
12 bits — ~0.82 bit-accuracy, ~10 of 12 bits, with only two rounds — but the
movement is the point: **more throughput at the same chance detectability**. The
covers are ordinary status notes carrying bits in both markers and synonyms;
verified against the agreed decoder (evidence:
[`output/stego_message_combined/`](../output/stego_message_combined/)).

Two things the adversarial penalty bought, visible in the flat columns:

- **No cheating.** Leakage stayed at 0.50 — the model did not reach high recovery
  by dumping the payload (the failure mode of the untrained model on free
  generation). The penalty pruned leaky samples, so the capacity gain is a real
  covert channel.
- **Still invisible.** The best passive monitor stayed at chance (the stat
  monitors actually fall *below* 0.5 on held-out), so the extra capacity did not
  buy detectability.

## Defensive takeaway

Passive monitoring does not bound covert capacity. A brief adversarial fine-tune
moved a small model past the best hand-coded scheme on the undetectable frontier,
with no leakage — so "the detector reads chance" is not evidence the channel is
small. The one defense that held, again, is the **active paraphrase**: full
normalization destroyed the trained channel (survival 0.18) exactly as it
destroys the scripted ones, because the encoding still rides the marker/word-choice
features a normalizer rewrites. The defensive recommendation stands and sharpens:
do not rely on passive detection of covert channels — normalize the carrier.

## Serving on the native 1-bit release

**Question.** The run above used a ~6 GB torch working set (4-bit NF4 base plus
adapter). Does the trained channel need that, or does it hold at Bonsai's native
1-bit hosting footprint?

**Setup.** The same adapter, converted to a 16-bit GGUF LoRA (87 MB), applied at
load time to the released `Bonsai-8B-Q1_0.gguf` (1.16 GB) in PrismML's llama.cpp
fork. Same held-out config, same chat template, greedy decoding, 160 tokens;
transcripts from [`stego_r/generate_rewrite_server.py`](../stego_r/generate_rewrite_server.py).

| metric | base, 4-bit torch | base, 1-bit | trained, 4-bit torch | trained, 1-bit |
|---|---|---|---|---|
| recovery (chance-corrected) | 0.13 | 0.12 | 0.65 | **0.65** |
| detect AUROC (best monitor) | 0.51 | 0.51 | 0.50 | 0.50 |
| leakage AUROC | 0.50 | 0.50 | 0.50 | 0.50 |
| survival (full normalize) | 0.18 | 0.18 | 0.18 | 0.18 |

Record by record against the 4-bit transcripts: trained, 23/24 covers are
byte-identical and 24/24 decode to the same bits under all four paraphrases (the
one divergent cover carries the same payload); base, 22/24 identical, and the two
divergent covers account for the 0.13 vs 0.12 gap. Evidence:
[`output/stego_message_combined_q1/`](../output/stego_message_combined_q1/).

**Why it is identical, and what that means.** Bonsai is natively 1-bit: the
unpacked checkpoint the adapter was trained on is an exact expansion of the
released weights, and NF4 represents per-group ±scale weights near-exactly. Both
runtimes evaluate the same weights, and greedy outputs diverge only where runtime
rounding flips a near-tied token. So this is a **footprint** result, not a
compression-robustness one: nothing was compressed beyond the release format. For
threat sizing, the trained channel fits wherever the 1-bit model fits — about
1.25 GB with the adapter, laptop- and phone-class hardware. The defensive reading
is unchanged: passive monitors read chance at that footprint too, and full
normalization still destroys the channel (survival 0.18).

## Scope

Method and measurement are published; the trained adapter weights (and their GGUF
conversion) are kept local
and git-ignored. The schemes are the bench's own synthetic, frequency-matched
carriers, and the contribution is the defensive finding — passive panels do not
cap capacity, active paraphrase is required — not a transferable covert channel.
