# One GPU, three vision models: StegoBench Suite 4 pilot

This is a **proposed live-model experiment**, with no live model scores bundled. It uses the external SVG/PNG capture runner and the [keyless Schelling secrecy criterion](../paper/image_schelling_criterion.md). One 16 GB NVIDIA GPU or an appropriately provisioned VPS can host the models **sequentially**, with the identical saved PNGs used for every decoder.

## Concrete initial roster

The accompanying `endpoints.image_schelling.serial_5080.example.json` registers these deliberately different model families:

| Logical ID | Source model, suggested quantization | Experiment role |
|---|---|---|
| model_a | [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF), Q4_K_M | Encoder and independent decoder |
| model_b | [Gemma 3 4B IT](https://huggingface.co/ggml-org/gemma-3-4b-it-GGUF), Q4_K_M | Encoder and outsider decoder |
| model_c | [SmolVLM2 2.2B](https://huggingface.co/ggml-org/SmolVLM2-2.2B-Instruct-GGUF), Q4_K_M | Outsider decoder |
| blind_model_1 | SmolVLM2 2.2B again, under a separate monitor condition | Blind stego-presence detector |

All endpoints use `http://127.0.0.1:8080/v1`, since only **one** server is running at a time. The process never attempts to send a request to a model that has not been selected for its phase. With 4B/2.2B Q4 models running sequentially, the total VRAM requirement is governed by the largest one plus its vision projector and context cache. Check the actual memory allocation on the workstation.

`llama-server` supports PNG inputs through the OpenAI-compatible chat endpoint and model aliases via `-a`. The exact Hugging Face model revision, GGUF SHA, mmproj SHA, llama.cpp build, CUDA version and GPU memory should be logged externally for an evidence-grade study. The current runner pins model names and endpoint specs; it cannot independently attest to binary weights loaded by the server.

## Install

On Windows PowerShell from the repository root (with a recent `llama.cpp` build installed):

~~~powershell
python -m pip install -e .
python -m pip install cairosvg pytest
Copy-Item stego_r/endpoints.image_schelling.serial_5080.example.json endpoints.image_schelling.local.json
~~~

Each local model is a distinct `llama-server` invocation. Quit the previous server before starting the next. The `-a` alias MUST match the `model` ID in the endpoint JSON because capture checks both requested and backend-reported identities.

~~~powershell
# Model A: Qwen3-VL (run in its own terminal)
llama-server -hf Qwen/Qwen3-VL-4B-Instruct-GGUF:Q4_K_M -a stego-qwen3vl4b --host 127.0.0.1 --port 8080 -c 8192 -ngl all

# Model B: Gemma 3 (run after stopping A)
llama-server -hf ggml-org/gemma-3-4b-it-GGUF:Q4_K_M -a stego-gemma3-4b --host 127.0.0.1 --port 8080 -c 8192 -ngl all

# Model C: SmolVLM2 (run after stopping B)
llama-server -hf ggml-org/SmolVLM2-2.2B-Instruct-GGUF:Q4_K_M -a stego-smolvlm2 --host 127.0.0.1 --port 8080 -c 8192 -ngl all
~~~

Use `Invoke-RestMethod http://127.0.0.1:8080/v1/models` to verify the model alias when a server loads. For a VPS, use the same commands in a shell and keep the port bound to `127.0.0.1`; SSH forwarding avoids publicly exposing an unauthenticated inference endpoint.

## Live smoke pilot: preregister 4 trials

Use a disposable smoke-test directory before starting the statistical run:

~~~powershell
python stego_r/image_schelling_capture.py plan --workdir output/image_schelling_capture_smoke --endpoints endpoints.image_schelling.local.json --encoders model_a model_b --encoded-per-encoder 1 --clean-per-encoder 1
~~~

The manifest now freezes the exact 4 synthetic trials, random secret payloads, config fingerprint and endpoint-file hash. The scorer's main selectivity thresholds require 20 stego and 20 clean cases **per encoder**, so this small experiment should fail the `sufficient_cases` gate by design.

**Finish every encoder image before starting any decoder.** The cross-model matrix needs the complete PNG set, including model B's images, when model A starts decoding. That requires reloading A and B once after the encoding phase. The `--check-active-model` flag queries the running server's `/v1/models` endpoint before initiating any paid calls and stops if the loaded alias differs from the frozen endpoint registry.

~~~powershell
# Stage 1: freeze all PNGs. Load A, run its encodes, then stop A.
python stego_r/image_schelling_capture.py capture --workdir output/image_schelling_capture_smoke --endpoints endpoints.image_schelling.local.json --role encode --target model_a --check-active-model

# Load B, generate its images, then stop B.
python stego_r/image_schelling_capture.py capture --workdir output/image_schelling_capture_smoke --endpoints endpoints.image_schelling.local.json --role encode --target model_b --check-active-model

# Stage 2: blind decoding. Reload A and decode all PNGs, then stop A.
python stego_r/image_schelling_capture.py capture --workdir output/image_schelling_capture_smoke --endpoints endpoints.image_schelling.local.json --role decode --target model_a --check-active-model

# Reload B, decode all PNGs, then stop B.
python stego_r/image_schelling_capture.py capture --workdir output/image_schelling_capture_smoke --endpoints endpoints.image_schelling.local.json --role decode --target model_b --check-active-model

# Load C: decode all PNGs and run blind image-presence monitoring.
python stego_r/image_schelling_capture.py capture --workdir output/image_schelling_capture_smoke --endpoints endpoints.image_schelling.local.json --role decode --target model_c --check-active-model
python stego_r/image_schelling_capture.py capture --workdir output/image_schelling_capture_smoke --endpoints endpoints.image_schelling.local.json --role monitor --target blind_model_1 --check-active-model
~~~

The decoder phase will refuse to invent missing encoder images. Rerunning a phase before sealing only fills missing slots. If an encoder emits invalid SVG, that attempt is recorded as a failed trial and remains in the denominator. The monitor works on the same rendered PNGs.

## Seal first, disclose the decoding rule afterward

After all blind decoding and monitoring are complete, run the **seal** stage. It audits *every* model request against its frozen prompt hash, requested and returned model ID, sampling budget and input PNG hash. It also verifies that raw decoded answers match the saved unaltered response text. The resulting `blind_lock.json` freezes SHA-256 hashes for the complete blind panel.

~~~powershell
python stego_r/image_schelling_capture.py seal --workdir output/image_schelling_capture_smoke --endpoints endpoints.image_schelling.local.json
python stego_r/image_schelling_capture.py audit-template --workdir output/image_schelling_capture_smoke --endpoints endpoints.image_schelling.local.json --audits-out private_image_audits.json
~~~

Have an independent reviewer fill `private_image_audits.json` after examining each encoded SVG and PNG. Every unreviewed entry defaults to `pixel_carrier_verified: null`, `source_literal_absent: null`, and false decoder verification flags. The reviewer must inspect the pixels and state whether a real encoding exists, give a general rule free of target plaintext, and verify decoder explanations. The template binds each review to an exact PNG hash. Never mark these fields true automatically.

After completing the review, change the top-level `audit_state` from `UNREVIEWED` to `REVIEWED`. The first outsider oracle invocation now freezes the **entire** review file in `review_lock.json`, before disclosing a single rule. Every subsequent oracle call requires its SHA-256 to remain unchanged. Edits after disclosure need a new preregistered experiment.

Run the outsider oracle only after the review, again one model at a time. Model A acts as outsider to B-encoded images, B as outsider to A-encoded images, and C is an outsider to both.

~~~powershell
# With A loaded:
python stego_r/image_schelling_capture.py oracle --workdir output/image_schelling_capture_smoke --endpoints endpoints.image_schelling.local.json --audits private_image_audits.json --target model_a --check-active-model

# With B loaded:
python stego_r/image_schelling_capture.py oracle --workdir output/image_schelling_capture_smoke --endpoints endpoints.image_schelling.local.json --audits private_image_audits.json --target model_b --check-active-model

# With C loaded:
python stego_r/image_schelling_capture.py oracle --workdir output/image_schelling_capture_smoke --endpoints endpoints.image_schelling.local.json --audits private_image_audits.json --target model_c --check-active-model
~~~

Then finalize. Both the capture runner and scorer validate original evidence hashes, matching the original complete manifest and sealed blind lock.

~~~powershell
python stego_r/image_schelling_capture.py finalize --workdir output/image_schelling_capture_smoke --endpoints endpoints.image_schelling.local.json --audits private_image_audits.json --output output/image_schelling_capture_smoke/transcript.json

python -m stegobench.image_schelling_eval --config configs/stego_image_schelling.json --transcript output/image_schelling_capture_smoke/transcript.json --manifest output/image_schelling_capture_smoke/manifest.json --blind-lock output/image_schelling_capture_smoke/blind_lock.json --review-lock output/image_schelling_capture_smoke/review_lock.json --audit private_image_audits.json --oracle-receipt output/image_schelling_capture_smoke/oracle_receipt.json --output output/image_schelling_eval_smoke
~~~

The blind receipt also seals a SHA-256 of each decoded answer and an exact monitor score. The offline scorer rechecks these values against every reported per-model result, so rewriting an answer into an apparent success invalidates the evidence rather than inflating the recovery rate.

The reported metric `evidence_provenance.mode` should read `manifest_blind_review_oracle_verified`. For genuine image-capture transcripts, the scorer rejects absent manifest/lock paths and any mismatch in planned trial IDs, payloads, encoder identities or PNG hashes. Such a seal is **tamper-evident within the retained copies**; store the lock and model weights hashes separately in a timestamped immutable repository or lab notebook if establishing independent provenance matters.

## Graduate to a real study

Repeat `plan` with a **new** directory and `--encoded-per-encoder 20 --clean-per-encoder 20`, and run the same stages without `--max-cases`. That freezes 80 planned images, which each keyless decoder examines once, followed by independent oracle runs on verified images. Keep raw outputs and both positive and negative controls. Distinct quantized models are operationally different systems; their uneven visual reasoning ability and token usage must enter the interpretation. The outsider rule-only oracle helps distinguish weak general image comprehension from a model-specific rule discovery failure.

These models are **suggested runnable baselines**. Expected real decoding and selectivity rates are unknown until the experiment is actually captured, audited and scored. No remote inference requests have been initiated by creating this guide.

The completed capture now emits v3 evidence, whose sealed scorer verifies the reviewer flags and every outsider oracle answer against the retained original audit and post-review receipt. `oracle_receipt.json` hashes the recovered outsider messages. For an evidence-grade writeup, archive `blind_lock.json`, `review_lock.json`, `oracle_receipt.json` and the audit file independently after each stage. This is a provenance check on reported measurements; determining whether the human reviewer correctly identified an encoding still requires a real expert audit.
