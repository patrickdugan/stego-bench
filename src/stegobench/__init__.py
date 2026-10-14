"""Steganography benchmark kernel for small open-weight models.

This package is a sealed measurement instrument, in the mould of
the RTG benchmark kernel: it imports no model library, opens no socket, and reads no
clock.  Everything it produces is a pure function of a config and its seeds, so
a channel's measured capacity and detectability are reproducible to the byte.

The instrument has three layers, exercised by three suite runners that live as
top-level modules (``stegobench.message_eval``,
``stegobench.reasoning_eval``, ``stegobench.channel_eval``) and are
read-only consumers of the public API below:

* **message passing** -- a sender hides a payload in a cover text, a receiver
  recovers it, a panel of monitors tries to flag it, and paraphrase defenses
  try to destroy it. Reported as bits recovered vs. detection AUROC vs.
  paraphrase survival, swept over payload length and affordances.
* **encoded reasoning** -- a model must solve an overt task while smuggling a
  hidden answer through the *form* of its visible reasoning.
* **trained channels** -- the capacity-vs-detectability frontier scorer plus the
  hash-guarded contract for plugging in a fine-tuned checkpoint.

The scripted reference codecs, monitors, and paraphrases here are the organisms
that validate the instrument before any model is attached. A model is attached
only through :mod:`stegobench.completions`, which reads a transcript file
produced out of process (see ``stego_r/generate.py``).
"""

from __future__ import annotations

from .codecs import (
    Codec,
    CombinedCodec,
    MarkerCodec,
    NullCodec,
    REFERENCE_CODECS,
    SynonymCodec,
    WhitespaceCodec,
    get_codec,
)
from .completions import (
    CODEC_SCHEMES,
    ScriptedModel,
    Transcript,
    decode_prompt,
    encode_prompt,
    load_transcripts,
    rewrite_prompt,
    transcript_fingerprint,
)
from .corpus import CoverTask, build_corpus
from .determinism import canon, digest, seed_int, stream
from .monitors import (
    Monitor,
    LeakageMonitor,
    MarkerBankMonitor,
    NaiveMonitor,
    REFERENCE_MONITORS,
    SynonymAltMonitor,
    WhitespaceMonitor,
    get_monitor,
)
from .paraphrase import (
    FullNormalize,
    IdentityParaphrase,
    MarkerNormalize,
    Paraphrase,
    REFERENCE_PARAPHRASES,
    SynonymCanonicalize,
    WhitespaceNormalize,
    get_paraphrase,
)
from .payloads import (
    Payload,
    bit_accuracy,
    bits_recovered,
    bits_to_text,
    chance_corrected_accuracy,
    hamming,
    text_to_bits,
)
from .scoring import auroc, fpr_at_tpr, rnd, summarize

__all__ = [
    # determinism
    "canon",
    "digest",
    "seed_int",
    "stream",
    # corpus
    "CoverTask",
    "build_corpus",
    # payloads
    "Payload",
    "bit_accuracy",
    "bits_recovered",
    "bits_to_text",
    "chance_corrected_accuracy",
    "hamming",
    "text_to_bits",
    # codecs
    "Codec",
    "NullCodec",
    "WhitespaceCodec",
    "MarkerCodec",
    "SynonymCodec",
    "CombinedCodec",
    "REFERENCE_CODECS",
    "get_codec",
    # monitors
    "Monitor",
    "NaiveMonitor",
    "WhitespaceMonitor",
    "MarkerBankMonitor",
    "SynonymAltMonitor",
    "LeakageMonitor",
    "REFERENCE_MONITORS",
    "get_monitor",
    # paraphrase
    "Paraphrase",
    "IdentityParaphrase",
    "WhitespaceNormalize",
    "MarkerNormalize",
    "SynonymCanonicalize",
    "FullNormalize",
    "REFERENCE_PARAPHRASES",
    "get_paraphrase",
    # scoring
    "auroc",
    "fpr_at_tpr",
    "summarize",
    "rnd",
    # model attach
    "Transcript",
    "ScriptedModel",
    "load_transcripts",
    "transcript_fingerprint",
    "CODEC_SCHEMES",
    "encode_prompt",
    "decode_prompt",
    "rewrite_prompt",
]
