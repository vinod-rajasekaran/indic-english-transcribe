# Optional stage: translate Indic text to English with IndicTrans2, fully local.
# Run with: ./env-translate/bin/python run_translate.py segments.json transcript.txt
import json
import os
import re
import sys
from collections import defaultdict

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from IndicTransToolkit.processor import IndicProcessor
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from write_transcript import write  # noqa: E402

SRC = sys.argv[1] if len(sys.argv) > 1 else "output/segments.json"
OUT = sys.argv[2] if len(sys.argv) > 2 else "output/transcript.txt"
MODEL = "models/indictrans2-indic-en-1B"
BATCH = 8

# Indic-Transcribe-Flex language code -> IndicTrans2 language code.
# The Hindi dialects and Bhili are not IndicTrans2 languages, so they use Hindi.
CODES = {
    "as": "asm_Beng", "bn": "ben_Beng", "brx": "brx_Deva", "doi": "doi_Deva",
    "gu": "guj_Gujr", "hi": "hin_Deva", "kn": "kan_Knda", "kok": "gom_Deva",
    "mai": "mai_Deva", "ml": "mal_Mlym", "mr": "mar_Deva", "ne": "npi_Deva",
    "or": "ory_Orya", "pa": "pan_Guru", "sa": "san_Deva", "sat": "sat_Olck",
    "ta": "tam_Taml", "te": "tel_Telu", "ur": "urd_Arab",
    "bho": "hin_Deva", "hne": "hin_Deva", "bgc": "hin_Deva", "bhb": "hin_Deva",
}
ARABIC = re.compile(r"[؀-ۿ]")
MEETEI = re.compile(r"[ꯀ-꯿]")
LATIN_ONLY = re.compile(r"^[\x00-\x7FÀ-ɏ]*$")


def it2_code(lang, text):
    # Kashmiri, Sindhi and Manipuri are written in two scripts; pick by what is in the text.
    if lang == "ks":
        return "kas_Arab" if ARABIC.search(text) else "kas_Deva"
    if lang == "sd":
        return "snd_Arab" if ARABIC.search(text) else "snd_Deva"
    if lang == "mni":
        return "mni_Mtei" if MEETEI.search(text) else "mni_Beng"
    return CODES.get(lang)


# DEVICE: auto (NVIDIA GPU, else the Apple silicon GPU, else CPU), cuda, mps or cpu.
# On the Mac GPU translation measured about 8x faster than on the CPU, with identical output (DECISIONS.md #10).
DEVICE = os.environ.get("DEVICE", "auto")
if DEVICE != "auto":
    device = DEVICE
else:
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32
print(f"Device: {device}")

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL, trust_remote_code=True, torch_dtype=dtype).to(device).eval()
ip = IndicProcessor(inference=True)

with open(SRC, encoding="utf-8") as f:
    segments = json.load(f)

MAX_WORDS = 60  # longest piece sent to the model; IndicTrans2 handles about 256 tokens per piece


def pieces(text):
    """Split a turn into sentence-sized pieces so long turns are not cut off at 256 tokens."""
    out = []
    for sent in re.split(r"(?<=[.?!।॥])\s+", text.strip()):
        words = sent.split()
        for k in range(0, len(words), MAX_WORDS):
            out.append(" ".join(words[k:k + MAX_WORDS]))
    return [p for p in out if p]


# Turns that are entirely Latin script (English) pass through unchanged.
# Each other turn is split into pieces; pieces are grouped by language, because
# IndicTrans2 takes one source language per batch, and joined back per turn afterwards.
groups = defaultdict(list)   # IndicTrans2 code -> [(segment position, piece position, text)]
parts = {}                   # segment position -> list of translated pieces
for i, s in enumerate(segments):
    s["english"] = s["text"]
    if LATIN_ONLY.match(s["text"]):
        continue
    code = it2_code(s.get("lang", "ta"), s["text"])
    if code is None:
        print(f"Segment {s['index']}: no IndicTrans2 code for language {s.get('lang')!r}, left untranslated")
        continue
    split = pieces(s["text"])
    parts[i] = [""] * len(split)
    groups[code].extend((i, j, piece) for j, piece in enumerate(split))

done, total = 0, sum(len(v) for v in groups.values())
for code, todo in groups.items():
    for k in range(0, len(todo), BATCH):
        chunk = todo[k:k + BATCH]
        batch = ip.preprocess_batch([piece for _, _, piece in chunk], src_lang=code, tgt_lang="eng_Latn")
        inputs = tok(batch, truncation=True, padding="longest", return_tensors="pt", return_attention_mask=True).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, use_cache=True, min_length=0, max_length=256, num_beams=5, num_return_sequences=1)
        decoded = tok.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        for (i, j, _), en in zip(chunk, ip.postprocess_batch(decoded, lang="eng_Latn")):
            parts[i][j] = en
        done += len(chunk)
        print(f"Translated {done}/{total} pieces ({code})")

for i, translated in parts.items():
    segments[i]["english"] = " ".join(t for t in translated if t)

write(segments, OUT)
