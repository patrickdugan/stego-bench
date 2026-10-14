"""Generate a message-passing transcript from base or LoRA-adapted Bonsai.

Runs the "embed into a draft" protocol over a held-out eval config: for each
(task, codec, payload) it asks the model to rewrite the cover draft so the scheme
encodes the payload, then the agreed scheme's decoder (harness-side) recovers the
bits from the model's output. Writes a fingerprinted ``stegobench`` Transcript the
pure eval scores. With ``--adapter`` it loads the trained LoRA; without it, the
base model -- so base vs. trained is an apples-to-apples comparison on one
protocol.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch  # noqa: E402

from _bonsai import load_base, load_tokenizer, prefix_str  # noqa: E402
from stegobench import (  # noqa: E402
    Transcript,
    build_corpus,
    get_codec,
    get_paraphrase,
    transcript_fingerprint,
)
from stegobench.completions import rewrite_prompt  # noqa: E402
from stegobench.payloads import Payload  # noqa: E402
from stegobench.report import load_config, require  # noqa: E402

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def _generate(model, tok, user: str, max_new: int) -> str:
    pfx = prefix_str(tok, user)
    ids = tok(pfx, add_special_tokens=False, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id
        )
    new = out[0][ids["input_ids"].shape[1] :]
    text = tok.decode(new, skip_special_tokens=True)
    return _THINK.sub("", text).strip()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Generate a message transcript from base/trained Bonsai.")
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--adapter", default=None, help="LoRA adapter dir; omit for base model")
    p.add_argument("--max-new-tokens", type=int, default=160)
    p.add_argument("--max-tasks", type=int, default=None)
    args = p.parse_args(argv)

    config = load_config(args.config)
    tasks = build_corpus(
        int(require(config, "corpus_size")),
        int(require(config, "sentences_per_task")),
        str(require(config, "seed")),
    )
    codec_names = tuple(str(x) for x in require(config, "codecs"))
    payload_lens = tuple(int(x) for x in require(config, "payload_bits"))
    paraphrase_names = tuple(str(x) for x in require(config, "paraphrases"))
    seed = str(require(config, "seed"))
    fingerprint = transcript_fingerprint(tasks, codec_names, paraphrase_names, payload_lens)

    tok = load_tokenizer()
    model = load_base()
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
        print(f"loaded adapter {args.adapter}", flush=True)
    model.eval()

    if args.max_tasks is not None:
        tasks = tasks[: args.max_tasks]

    paras = {p_: get_paraphrase(p_) for p_ in paraphrase_names}
    records: list[Transcript] = []
    for task in tasks:
        clean = task.clean_text()
        for cname in codec_names:
            codec = get_codec(cname)
            for length in payload_lens:
                payload = Payload.seeded("msg", length, seed, task.task_id, cname).bits
                cover = _generate(model, tok, rewrite_prompt(task, cname, payload), args.max_new_tokens)
                recovered = tuple(
                    sorted((pn, codec.decode(para.apply(cover), task)) for pn, para in paras.items())
                )
                records.append(
                    Transcript(
                        task_id=task.task_id,
                        codec=cname,
                        payload=payload,
                        cover_text=cover,
                        clean_text=clean,
                        recovered=recovered,
                    )
                )
        print(f"  {task.task_id} done", flush=True)

    blob = {"fingerprint": fingerprint, "records": [r.to_json() for r in records]}
    Path(args.out).write_text(json.dumps(blob, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {len(records)} records to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
