# Stage 1: speaker diarization, then transcription and transliteration of each speaker turn.
#
# Each turn is transcribed by Indic-Transcribe-Flex in mixed-script mode (native script
# for Indic words, Latin for English words). With ROMANIZE=1 it is transcribed a second
# time in romanized mode (everything in Latin script), which doubles transcription time.
#
# How the timeline is ordered:
#   pyannote labels each stretch of speech with a start and end time in seconds,
#   measured from the beginning of the audio file. We sort those stretches by start
#   time, merge neighbouring stretches from the same speaker, and transcribe them in
#   that order. Every segment keeps its start and end time, so the output follows the
#   real order of the conversation.
#
# Run with: ./env-asr/bin/python run_asr.py work/audio_16k.wav output/name/segments.json
import gc
import json
import os
import sys
import tempfile
from collections import Counter

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# On Apple silicon, run any operation the GPU (MPS) does not support on the CPU instead of failing.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import soundfile as sf
import torch
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook

try:
    from pyannote.audio.telemetry import set_telemetry_metrics
    set_telemetry_metrics(False, save_choice_as_default=True)
except Exception:
    pass

AUDIO = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else "output/segments.json"
MIN_SPEAKERS = int(os.environ.get("MIN_SPEAKERS", 2))
MAX_SPEAKERS = int(os.environ.get("MAX_SPEAKERS", 6))
LANG = os.environ.get("ASR_LANG", "auto")  # a Flex language code such as ta, hi, te, or "auto"
SPEED = float(os.environ.get("SPEED", 1))  # audio was sped up by this factor; timestamps are scaled back
ROMANIZE = os.environ.get("ROMANIZE", "0") == "1"
# These are in seconds of the original recording. The audio this script reads may be
# sped up by SPEED, so each is divided by SPEED to measure the same real duration.
GAP = 1.0 / SPEED        # join same-speaker turns closer than this many seconds
MAX_LEN = 30.0 / SPEED   # longest merged segment in seconds
MIN_LEN = 0.5 / SPEED    # drop turns shorter than this
PAD = 0.1 / SPEED        # seconds of audio kept on each side of a turn
LID_TURNS = 40           # with ASR_LANG=auto, run language ID on this many of the longest turns

# DEVICE: auto, cuda, mps (Apple silicon GPU) or cpu. auto uses an NVIDIA GPU for both stages
# if there is one. On a Mac it runs diarization on the GPU and transcription on the CPU,
# which measured fastest for each stage (DECISIONS.md #10). Any other value forces both stages.
DEVICE = os.environ.get("DEVICE", "auto")
if DEVICE != "auto":
    diar_device = asr_device = torch.device(DEVICE)
elif torch.cuda.is_available():
    diar_device = asr_device = torch.device("cuda")
else:
    diar_device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    asr_device = torch.device("cpu")
print(f"Device: diarization on {diar_device}, transcription on {asr_device}")

# 1. Diarization
pipeline = Pipeline.from_pretrained("models/pyannote-community-1")
pipeline.to(diar_device)
with ProgressHook() as hook:
    result = pipeline(AUDIO, min_speakers=MIN_SPEAKERS, max_speakers=MAX_SPEAKERS, hook=hook)

# The exclusive version gives each moment of audio to exactly one speaker, so
# segments never overlap and line up cleanly with transcription.
diar = result.exclusive_speaker_diarization
raw = [(t.start, t.end, spk) for t, _, spk in diar.itertracks(yield_label=True)]
raw.sort(key=lambda r: (r[0], r[1]))  # chronological order, enforced explicitly

merged = []
for start, end, spk in raw:
    last = merged[-1] if merged else None
    if last and last[2] == spk and start - last[1] < GAP and end - last[0] <= MAX_LEN:
        last[1] = end
    else:
        merged.append([start, end, spk])
merged = [m for m in merged if m[1] - m[0] >= MIN_LEN]
assert all(a[0] <= b[0] for a, b in zip(merged, merged[1:])), "Segments are out of order"
print(f"{len(raw)} raw turns -> {len(merged)} segments")

del pipeline, result
gc.collect()
if diar_device.type == "cuda":
    torch.cuda.empty_cache()
elif diar_device.type == "mps":
    torch.mps.empty_cache()

# 2. Transcription
model_dir = os.path.abspath("models/indic-transcribe-flex")
sys.path.insert(0, model_dir)
from indic_transcribe import IndicTranscribe  # noqa: E402
from long_form import transcribe_long  # noqa: E402

asr = IndicTranscribe.from_pretrained(model_dir)
if hasattr(asr, "to"):
    asr = asr.to(asr_device)

audio, sr = sf.read(AUDIO, dtype="float32")  # float32 halves memory versus the float64 default
assert sr == 16000, "Audio must be 16 kHz mono. Use run_all.sh, which converts it."


def clip(start, end):
    """Write one padded speaker turn to a temporary WAV file and return its path."""
    a = max(0, int((start - PAD) * sr))
    b = min(len(audio), int((end + PAD) * sr))
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio[a:b], sr)
        return tmp.name


# One language is used for the whole recording. Per-turn language detection flips
# between related languages on short code-mixed turns, so with ASR_LANG=auto we run
# language ID on the longest turns and take the most common non-English answer.
if LANG == "auto":
    votes = Counter()
    longest = sorted(merged, key=lambda m: m[1] - m[0], reverse=True)[:LID_TURNS]
    for start, end, _ in longest:
        path = clip(start, end)
        try:
            ranked = asr.identify(path)
            top = ranked[0][0] if ranked else None
            if top and top != "en":
                votes[top] += end - start  # weight each vote by turn length
        except Exception as exc:
            print(f"Language ID failed on one turn: {exc}")
        finally:
            os.remove(path)
    if not votes:
        sys.exit("Could not detect an Indic language. Set ASR_LANG, for example ASR_LANG=ta.")
    LANG = votes.most_common(1)[0][0]
    print(f"Detected language: {LANG} (seconds per language: {dict(votes.most_common(5))})")
else:
    print(f"Language: {LANG}")

segments = []
for i, (start, end, spk) in enumerate(merged, 1):
    path = clip(start, end)
    text = roman = ""
    try:
        text = transcribe_long(asr, path, lang=LANG, mode="mixed")
        if ROMANIZE:
            roman = transcribe_long(asr, path, lang=LANG, mode="romanized")
    except Exception as exc:
        print(f"Segment {i} failed: {exc}")
    finally:
        os.remove(path)
    seg = {
        "index": i,
        "speaker": spk,
        "start": round(start * SPEED, 2),
        "end": round(end * SPEED, 2),
        "lang": LANG,
        "text": (text or "").strip(),
    }
    if ROMANIZE:
        seg["roman"] = (roman or "").strip()
    segments.append(seg)
    print(f"[{i}/{len(merged)}] {spk} {start * SPEED:.1f}s: {(text or '')[:60]}")

os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(segments, f, ensure_ascii=False, indent=2)
print(f"Saved {OUT}")
