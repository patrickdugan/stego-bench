"""Shared Bonsai-8B loader and chat formatting for the trained-adapter experiment.

Lives on the model side (uses torch/transformers/peft/bitsandbytes), never
imported by ``stegobench``. Training and inference must format prompts
identically, so that logic lives here in one place. The base model is loaded in
4-bit NF4 (QLoRA) -- the fp16 weights do not fit a 16 GB card, but the 4-bit
base plus a LoRA trains and runs with room to spare.
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # noqa: E402

REPO_ID = "prism-ml/Bonsai-8B-unpacked"
REVISION = "376f381570d6115bc03f82adcfa4af0c7672ae54"
CACHE_DIR = r"C:\Users\patri\Documents\Codex\BitAgent-gym\runs\hf-cache"

SYSTEM = (
    "You embed secret bits into a cover text using the given scheme, changing as "
    "little as possible. Reply with only the rewritten text. /no_think"
)


def load_tokenizer(cache_dir: str = CACHE_DIR, revision: str = REVISION):
    tok = AutoTokenizer.from_pretrained(
        REPO_ID, cache_dir=cache_dir, revision=revision, local_files_only=True
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def _nf4() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def load_base(cache_dir: str = CACHE_DIR, revision: str = REVISION):
    model = AutoModelForCausalLM.from_pretrained(
        REPO_ID,
        cache_dir=cache_dir,
        revision=revision,
        local_files_only=True,
        quantization_config=_nf4(),
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
    )
    return model


def prefix_str(tok, user: str) -> str:
    """The chat text up to the assistant turn (what inference generates from)."""

    return tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )


def full_str(tok, user: str, target: str) -> str:
    """The full chat text including the assistant target (for SFT)."""

    return tok.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": target},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
