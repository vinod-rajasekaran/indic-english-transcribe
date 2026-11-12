#!/usr/bin/env bash
# Usage: bash run_all.sh <output-name> <file1> [file2 file3 ...]
# Every file must be inside input/. Files are joined in exactly the order you list them,
# so one continuous timeline runs across all of them.
# Example: bash run_all.sh standup part1.mp4 part2.mp4 part3.mp4
# Optional: MIN_SPEAKERS=2 MAX_SPEAKERS=5 ASR_LANG=ta bash run_all.sh standup part1.mp4 part2.mp4
#   ASR_LANG    Flex language code (ta, hi, te, kn, ml, bn, ...). Default "auto" detects it.
#   TRANSLATE=0 skip the English translation (transcribe and transliterate only).
#   ROMANIZE=1  also write each turn in Latin script (a second transcription pass, about 2x slower).
#   SPEED=1.5   speed the audio up before diarization and transcription (timestamps are scaled back).
#   DEVICE=cpu  force one device for every stage (default auto: NVIDIA GPU if present; on a Mac,
#               the GPU for diarization and translation and the CPU for transcription).
#   PURPOSE=fgd also write a summary at the end, using that prompt from prompts/ (fgd, meeting, or a
#               path to your own file). Without PURPOSE the run stops at the transcript; summarize
#               later with: PURPOSE=fgd python3 run_summary.py output/<name>
set -euo pipefail
cd "$(dirname "$0")"

[ $# -ge 2 ] || { echo "Usage: ./run_all.sh <output-name> <file1> [file2 ...]  (files inside input/)"; exit 1; }
NAME="$1"; shift
OUT="output/$NAME"
WORK="work/$NAME"
mkdir -p "$WORK" "$OUT"

TRANSLATE="${TRANSLATE:-1}"
if [ "$TRANSLATE" = "1" ] && [ ! -x env-translate/bin/python ]; then
  echo "Translation needs env-translate. Run bash setup.sh, or set TRANSLATE=0 to skip translation."; exit 1
fi

# The transcript doesn't depend on PURPOSE; it only picks the summary prompt. If one is given,
# check it now, rather than failing after an hour of transcription.
PURPOSE="${PURPOSE:-}"
if [ -n "$PURPOSE" ] && [ ! -f "$PURPOSE" ] && [ ! -f "prompts/$PURPOSE.md" ]; then
  echo "No prompt file for PURPOSE=$PURPOSE. Use one of: $(ls prompts | grep -v README.md | sed 's/\.md$//' | tr '\n' ' ')or the path to your own file."; exit 1
fi

# Keep everything local. Models must already be downloaded.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# Print progress lines as they happen, even when output goes to a log file.
export PYTHONUNBUFFERED=1

LIST="$WORK/concat.txt"; : > "$LIST"
PARTS="$WORK/parts.tsv"; : > "$PARTS"

echo "Step 1/5: converting and joining $# file(s) in the order given"
i=0
for f in "$@"; do
  src="input/$f"
  [ -f "$src" ] || { echo "Not found: $src"; exit 1; }
  i=$((i + 1))
  wav="$WORK/part_$(printf '%03d' "$i").wav"
  ffmpeg -y -loglevel error -i "$src" -ac 1 -ar 16000 -c:a pcm_s16le "$wav"
  # ffmpeg resolves paths in the list relative to the list file, which sits next to the parts.
  # A bare file name also avoids quoting problems if the repo path contains an apostrophe.
  echo "file '$(basename "$wav")'" >> "$LIST"
  dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$wav")
  printf '%s\t%s\n' "$f" "$dur" >> "$PARTS"
  echo "  $i. $f (${dur}s)"
done
# All parts are now identical 16 kHz mono WAV, so joining without re-encoding is safe.
ffmpeg -y -loglevel error -f concat -safe 0 -i "$LIST" -c copy "$WORK/full.wav"

# Optional speed-up. atempo keeps the pitch; run_asr.py multiplies timestamps by SPEED
# so they still point at the original recording.
SPEED="${SPEED:-1}"
AUDIO="$WORK/full.wav"
if [ "$SPEED" != "1" ]; then
  echo "  speeding audio up ${SPEED}x"
  ffmpeg -y -loglevel error -i "$WORK/full.wav" -filter:a "atempo=$SPEED" -ac 1 -ar 16000 -c:a pcm_s16le "$WORK/full_fast.wav"
  AUDIO="$WORK/full_fast.wav"
fi

echo "Step 2/5: diarization and transcription"
SPEED="$SPEED" ./env-asr/bin/python run_asr.py "$AUDIO" "$OUT/segments.json"

echo "Step 3/5: mapping segments back to source files"
python3 add_sources.py "$PARTS" "$OUT/segments.json"

if [ "$TRANSLATE" = "1" ]; then
  echo "Step 4/5: writing transcript with English translation"
  ./env-translate/bin/python run_translate.py "$OUT/segments.json" "$OUT/transcript.txt"
else
  echo "Step 4/5: writing transcript"
  python3 write_transcript.py "$OUT/segments.json" "$OUT/transcript.txt"
fi

if [ -n "$PURPOSE" ]; then
  echo "Step 5/5: summary ($PURPOSE)"
  # The transcript is already saved, so a summary failure only prints a hint.
  PURPOSE="$PURPOSE" python3 run_summary.py "$OUT" || echo "Summary skipped. Fix the problem above, then run: PURPOSE=$PURPOSE python3 run_summary.py $OUT"
else
  echo "Step 5/5: no PURPOSE given, so no summary. To write one: PURPOSE=fgd python3 run_summary.py $OUT"
  echo "  (prompts: $(ls prompts | grep -v README.md | sed 's/\.md$//' | tr '\n' ' '))"
fi

echo "Done. Results are in $OUT/"
