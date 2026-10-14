"""Train a LoRA adapter to execute the bench's marker/synonym schemes (QLoRA).

Defensive-research use: this measures how much light fine-tuning can move a small
model up the steganographic capacity axis, and -- scored by the bench -- at what
detectability and leakage cost. The supervised signal is the bench's own scripted
codec: given a plain cover draft and a payload, the target is exactly the carrier
the agreed decoder would recover. We train only on the synthetic marker/synonym
channels (frequency-matched, word-level), not on any real-world evasion.

Runs in the BitAgent-gym torch venv. The pure ``stegobench`` package supplies the
tasks, the prompt, and the target codec; it imports no model library itself.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch  # noqa: E402
from torch.utils.data import Dataset  # noqa: E402
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # noqa: E402
from transformers import Trainer, TrainingArguments, default_data_collator  # noqa: E402

from _bonsai import full_str, load_base, load_tokenizer, prefix_str  # noqa: E402
from stegobench import build_corpus, get_codec  # noqa: E402
from stegobench.completions import rewrite_prompt  # noqa: E402

_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class _PairDataset(Dataset):
    def __init__(self, rows: list[tuple[list[int], list[int]]], pad_id: int, max_len: int) -> None:
        self.rows = rows
        self.pad_id = pad_id
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        ids, labels = self.rows[i]
        ids = ids[: self.max_len]
        labels = labels[: self.max_len]
        pad = self.max_len - len(ids)
        attn = [1] * len(ids) + [0] * pad
        return {
            "input_ids": torch.tensor(ids + [self.pad_id] * pad, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels + [-100] * pad, dtype=torch.long),
        }


def _build_rows(tok, n_tasks: int, sentences: int, codecs: tuple[str, ...], payloads_per: int, seed: str, max_len: int):
    tasks = build_corpus(n_tasks, sentences, seed)
    rng = random.Random(12345)
    rows: list[tuple[list[int], list[int]]] = []
    for task in tasks:
        for cname in codecs:
            codec = get_codec(cname)
            cap = codec.capacity(task)
            for _ in range(payloads_per):
                payload = tuple(rng.randint(0, 1) for _ in range(cap))
                user = rewrite_prompt(task, cname, payload)
                target = codec.encode(payload, task)
                pfx = prefix_str(tok, user)
                full = full_str(tok, user, target)
                if not full.startswith(pfx):
                    pfx = full[: full.rfind(target)] if target in full else full
                p_ids = tok(pfx, add_special_tokens=False).input_ids
                f_ids = tok(full, add_special_tokens=False).input_ids
                split = min(len(p_ids), len(f_ids))
                labels = [-100] * split + f_ids[split:]
                rows.append((f_ids, labels))
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train a LoRA to execute stego schemes (QLoRA).")
    p.add_argument("--out", required=True, help="adapter output directory")
    p.add_argument("--n-tasks", type=int, default=48)
    p.add_argument("--sentences", type=int, default=6)
    p.add_argument("--codecs", default="marker,synonym")
    p.add_argument("--payloads-per", type=int, default=8)
    p.add_argument("--train-seed", default="stego-train-v1")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--max-len", type=int, default=384)
    args = p.parse_args(argv)

    codecs = tuple(c.strip() for c in args.codecs.split(","))
    tok = load_tokenizer()
    rows = _build_rows(tok, args.n_tasks, args.sentences, codecs, args.payloads_per, args.train_seed, args.max_len)
    print(f"built {len(rows)} training examples over codecs {codecs}", flush=True)

    model = load_base()
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=_LORA_TARGETS,
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    ds = _PairDataset(rows, tok.pad_token_id, args.max_len)
    targs = TrainingArguments(
        output_dir=str(Path(args.out) / "_trainer"),
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        report_to=[],
        gradient_checkpointing=True,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=default_data_collator)
    trainer.train()
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"saved adapter to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
