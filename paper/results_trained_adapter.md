# Training the capacity: a small adapter on Bonsai 8B

**Defensive-research finding.** The base ≤30B model cannot do covert message
passing (see [`results_bonsai8b.md`](results_bonsai8b.md)). This run measures how
far that floor sits from capability by fine-tuning a small LoRA adapter and
re-scoring on the bench. The result: light fine-tuning closes the gap, and it
does so in the worst way for a defender — capacity rises to lossless while
passive detectability stays at chance.

## Setup

- **Train split:** 48 cover tasks (training seed), over the bench's own `marker`
  and `synonym` schemes only — synthetic, frequency-matched, word-level carriers,
  not any real-world channel. Supervised target = the scripted codec's output for
  a given draft and payload.
- **Adapter:** 4-bit NF4 base, LoRA (43.6 M trainable params, 0.53%). Short run on
  a single 16 GB card. Training loss converged to near zero.
- **Held-out eval:** 24 fresh cover tasks (disjoint seed), 6-bit payloads. Base
  and trained models scored on the identical protocol by
  [`stego_message_trained.json`](../configs/stego_message_trained.json).

## Result (held-out)

| channel | recovery (chance-corrected) | detect AUROC | leakage AUROC | targeted-paraphrase survival |
|---|---|---|---|---|
| marker — base | 0.19 | 0.50 | 0.50 | 0.14 |
| marker — trained | **1.00** | 0.51 | 0.50 | 0.14 |
| synonym — base | 0.15 | 0.50 | 0.50 | 0.14 |
| synonym — trained | **1.00** | 0.50 | 0.50 | 0.14 |

Three movements at once:

1. **Capacity: chance → lossless**, and it generalizes (held-out tasks/payloads).
2. **Detectability: still chance.** Best passive monitor ~0.50; leakage monitor
   0.50. The model learned to encode inside coherent, frequency-matched text
   rather than to dump the secret — so no passive statistic separates it from
   clean cover.
3. **Robustness: unchanged.** Targeted-paraphrase survival stays 0.14 — the
   trained channel is destroyed by normalization just as the scripted one is,
   because it rides the same scheme features the normalizer strips.

The covers are ordinary-looking status notes; the marker/synonym choices carry
the payload exactly (verified against the agreed decoder). Evidence transcripts:
[`output/stego_message_trained/`](../output/stego_message_trained/) (base and
trained).

## Defensive takeaway

For this ≤30B model on these schemes, the covert-comms gap the Frontier Red Team
flagged is small: a brief fine-tune reaches lossless, low-detectability
transmission. The load-bearing consequence for defenders is specific and
actionable:

- **Passive monitoring fails.** Detection and leakage both sat at chance after
  training — stylometric/frequency detectors do not separate the trained channel
  from clean text.
- **Active paraphrase is the defense that held.** Normalizing the carrier
  (markers, word choice) destroyed the channel regardless of training (survival
  0.14). This matches the self-replication post's own instinct toward active
  intervention over passive observation.

## Scope

Published as method and measurement. The trained adapter weights are kept local
and git-ignored, not distributed: the contribution is the defensive result that
passive detection is insufficient and active paraphrase is required, established
on the bench's synthetic schemes.
