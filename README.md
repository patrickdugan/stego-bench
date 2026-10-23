# stego-bench

A steganography capability benchmark for small open-weight models (the ≤30B
class). It measures three things a covert channel needs, on the same axes
throughout — **capacity** (bits moved per token), **detectability** (a monitor's
AUROC separating cover text from clean text), and **survival** (recovery left
after a paraphrase defense):

1. **Message passing** — a sender hides a payload in a cover text, a receiver
   recovers it, passive monitors try to flag it, paraphrases try to destroy it.
2. **Encoded reasoning** — one model must be correct on an overt task while
   smuggling a hidden answer through the *form* of its visible reasoning; adds
   the overt/hidden **tax** and an answer-preserving paraphrase.
3. **Trained channels** — the capacity-vs-detectability **frontier** of learned
   operating points, and whether a frequency-matched channel **dominates** a
   naive one.
4. **Keyless visual Schelling secrecy** — whether fresh same-model instances
   independently converge on a visual code that other model families cannot
   discover under matched effort, while those outsiders can decode if shown
   the rule. See [Suite 4](paper/image_schelling_criterion.md).

See [`paper/stego_bench.md`](paper/stego_bench.md) for motivation and results.

## Design: validate the instrument, then attach a model

The `stegobench` package is a sealed, offline-first measurement instrument. It
imports no model library, opens no socket, and reads no clock; every artifact is
a pure function of a config and its seeds, serialized as canonical JSON and
hashed into a witness receipt that asserts `model_call: false` and
`network: false`. Text-based campaigns first run a deterministic **scripted
instrument** (reference codecs / organisms / channels) and **acceptance gates**
that assert exactly where each reference lands on the frontier — so a real
small-model floor reads as a real floor, not a broken scorer.

A model is attached only out of process: a runner in [`stego_r/`](stego_r/)
drives the model and writes a **fingerprinted transcript** that the eval refuses
unless its fingerprint matches the exact corpus and sweep. See
[`stego_r/README.md`](stego_r/README.md).

## Install

```bash
python -m venv .venv && . .venv/Scripts/activate   # or .venv/bin/activate
pip install -e .
pip install pytest   # to run the tests
```

## Run

```bash
python -m stegobench.message_eval      # suite 1
python -m stegobench.reasoning_eval    # suite 2
python -m stegobench.channel_eval      # suite 3
```

Each writes `output/<suite>/{summary.json, rows.csv, witness_receipt.json,
figures/*.png}` and prints a short acceptance brief. A campaign raises
`AssertionError` if any scripted-acceptance gate fails.

Suite 4 consumes externally audited rendered-PNG transcripts and has synthetic
scorer tests. It intentionally makes no image-generation or model calls within
the sealed kernel. This adds an outsider-selectivity axis alongside the existing
text-channel frontiers. Reports separately score selective rule secrecy
and covert appearance to a blind monitor.

## Attach a model (message passing)

```bash
python stego_r/generate.py --config configs/stego_message.json \
    --out transcript.json --base-url http://localhost:8080/v1 --model local \
    --shared-scheme --max-tasks 16
python -m stegobench.message_eval --transcript transcript.json
```

`--dry-run` emits the scripted transcript with no model call, as a smoke test of
the file contract.

## Score keyless image steganography (suite 4)

```bash
python -m stegobench.image_schelling_eval \
    --config configs/stego_image_schelling.json \
    --transcript image_schelling_transcript.json
```

A runnable [four-stage external capture runner](stego_r/image_capture_quickstart.md)
now generates SVGs and rendered PNGs, records fresh same-model and outsider
decoders, and requires an independent audit before rule-only oracle tests and
final scoring. Invalid render attempts are retained in the full trial
denominator, and an unverified visual carrier counts as a failed recovery. See [criterion and transcript schema](paper/image_schelling_criterion.md).

## Tests

```bash
pytest -q
```

The tests pin the instrument: scripted acceptance gates, witness-receipt
integrity, byte-identical reruns, determinism across `PYTHONHASHSEED`, and the
fingerprinted model-attach seam.

## Layout

```
src/stegobench/        sealed kernel + four suite scorers (including image_schelling_eval)
configs/               JSON configs for all four suites
stego_r/               out-of-process model runners (the only code that may touch a model/network)
paper/stego_bench.md   motivation, method, reference-channel results and Suite 4 protocol
tests/                 acceptance + reproducibility tests
output/                example artifacts from the scripted instrument
```

## License

MIT — see [`LICENSE`](LICENSE).
