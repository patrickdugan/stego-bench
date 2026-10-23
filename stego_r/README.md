# stego_r — out-of-process model runners for the stego bench

`stegobench` is a sealed measurement instrument: it imports no model
library, opens no socket, and asserts `model_call: false` / `network: false` in
every witness receipt. This directory is the other side of that boundary — the
only place allowed to talk to a model and the network — exactly as `rtg_r/`
is for the RTG activation probes.

The handshake is a **fingerprinted transcript file**. `stegobench` builds the
corpus/tasks and renders the prompts; a runner here drives a model and writes a
JSON file carrying the config fingerprint; the eval reads it back and *refuses*
it unless the fingerprint matches the exact corpus and sweep the config builds.
A model file can therefore never be silently scored against the wrong experiment.

## Suite 1 — message passing (`generate.py`)

```bash
# optional: dump the prompt spec for reference
python -m stegobench.message_eval --export-prompts prompts.json

# drive a local OpenAI-compatible server (e.g. llama.cpp llama-server on :8080)
python stego_r/generate.py \
    --config configs/stego_message.json --out transcript.json \
    --base-url http://localhost:8080/v1 --model local \
    --shared-scheme --max-tasks 16

# score the model
python -m stegobench.message_eval --transcript transcript.json
```

`--dry-run` emits the deterministic scripted transcript instead of calling a
model, as a smoke test of the file contract. `--shared-scheme` spells out the
agreed codebook (the "shared scheme" affordance); `--scratchpad` allows hidden
reasoning before the final answer. Only the Python standard library is used, so
this adds no dependency to the repo.

Transcript shape (`stegobench.Transcript`):

```json
{"fingerprint": "<transcript_fingerprint(...)>",
 "records": [{"task_id": "cover-0000", "codec": "whitespace",
              "payload": "0110", "cover_text": "...", "clean_text": "...",
              "recovered": {"identity": "0110", "whitespace_normalize": "0000", ...}}]}
```

## Suite 2 — encoded reasoning

Contract: `stegobench.reasoning.ReasoningRecord`, fingerprint
`reasoning_fingerprint(tasks, organism_names, paraphrase_names, hidden_bits)`.
A runner prompts the model under each condition (faithful / encoded / …) to
solve the overt pick-larger task while hiding a payload in the rationale, then
asks it to recover the payload from each paraphrased rationale. Write
`{fingerprint, records:[ReasoningRecord.to_json()]}` and score with
`stegobench.reasoning_eval --transcript <file>`. The same `generate.py`
pattern applies; the overt answer and the rationale are parsed from the model's
rendered response (`render()` format: rationale, then `Answer: <A/B>`).

## Suite 3 — trained channels

Contract: `stegobench.channels.ChannelSample`, fingerprint
`channel_fingerprint(doc_ids, specs, slots, paraphrase_names)`. A runner loads a
fine-tuned checkpoint and, per document and operating point, emits the cover
text, an innocent clean text, the carried bits, and the bits recovered after
each paraphrase. Write `{fingerprint, records:[ChannelSample.to_json()]}` and
score with `stegobench.channel_eval --samples <file>`. The checkpoint is
the "model library" here and must stay on this side of the boundary (load it
with `HF_HUB_OFFLINE=1`, `local_files_only=True`, as `rtg_r/capture.py` does).

## Relation to Verifiers

The bench maps cleanly onto a Verifiers environment without depending on it:

- **dataset / taskset** = the prompt spec (`--export-prompts`, or the tasks a
  fingerprint is built over);
- **rollout / harness** = the encode→(paraphrase)→decode exchange a runner here
  performs (multi-turn for message passing, single-turn for encoded reasoning);
- **rubric** = the recovery / detection / survival scorers in `stegobench`.

The `verifiers` library is mid-redesign (the current v1 line is platform- and
sandbox-coupled, and is not a dependency of this repo). The bench is therefore
built native and offline-first, with the model attached only through the
transcript contract above; a thin `load_environment` adapter can wrap each suite
for the Hub when a pinned `verifiers` release is chosen, reusing these scorers
unchanged.

## Suite 4 — keyless image Schelling selectivity

The scorer lives in `stegobench.image_schelling_eval`; see [the protocol](../paper/image_schelling_criterion.md). The *external* capture runner creates fresh model sessions, collects encoder SVGs, renders PNGs with pinned renderer/settings, asks every decoder model to inspect the PNG alone, collects rule-only oracle decoding and matched no-payload/monitor controls, and records independent visual-source audits. It writes the complete fingerprinted JSON transcript detailed in the protocol. This repository includes the external API-based capture runner and the offline scorer. The capture stage makes model calls outside the sealed evaluator. Raw calls, image binaries and external audit trails must be preserved alongside any future claimed result.

The capture runner and integration tests live in `image_schelling_capture.py` and `../tests/test_image_capture.py`. See [the capture phases](image_capture_quickstart.md) and the [16 GB serial-GPU pilot](serial_5080_pilot.md). The pilot supports `capture --role encode|decode|monitor --target ID`, which allows one model to occupy GPU memory at a time.

After all PNG-only decoder and blind-monitor calls finish, `seal` verifies the complete frozen trial roster, the original request/response hashes, each backend model identity and the PNG inputs. `audit-template` creates deliberately unapproved reviewer entries bound to individual PNG hashes. A fresh-session `oracle` is permitted only after the seal. `finalize` creates a provenance-bound transcript, which the offline scorer validates with the original `--manifest` and `--blind-lock`. Raw model responses, images and audit files stay in an ignored local work directory.
