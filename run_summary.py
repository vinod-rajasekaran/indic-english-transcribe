# Stage 5: focus group summary with a local LLM served by Ollama. Fully local.
# Uses only the standard library, so it runs with system python3.
#
# Run with: python3 run_summary.py output/<name>
#
# Reads  output/<name>/transcript.json   (written by the transcription stages)
# Uses   output/<name>/speakers.json     (speaker label -> role; created on first run)
# Writes output/<name>/summary.md
#
# Options (environment variables):
#   SUMMARY_MODEL  Ollama model tag            default qwen3.5:9b
#   NUM_CTX        context window in tokens    default 32768
#   OLLAMA_URL     Ollama server address       default http://localhost:11434
#   SUMMARY_THINK  1 lets Qwen reason before answering (slower, sometimes better)
#   SUMMARY_SOURCE what the model reads:
#                    english   the IndicTrans2 translation (default)        -> summary.md
#                    original  the transcript as spoken, Indic + English    -> summary_original.md
#                    both      the original with the translation as a hint  -> summary_both.md
import datetime
import json
import os
import re
import sys
import urllib.error
import urllib.request
from collections import defaultdict

MODEL = os.environ.get("SUMMARY_MODEL", "qwen3.5:9b")
NUM_CTX = int(os.environ.get("NUM_CTX", 32768))
URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
THINK = os.environ.get("SUMMARY_THINK", "0") == "1"
SOURCE = os.environ.get("SUMMARY_SOURCE", "english")
if SOURCE not in ("english", "original", "both"):
    sys.exit("SUMMARY_SOURCE must be english, original or both")
OUTPUT_TOKENS = 4096     # room kept for the model's answer
PROMPT_OVERHEAD = 1500   # instructions and headings around the transcript
CHARS_PER_TOKEN = 3.5    # rough estimate for English text; errs on the safe side
INDIC_CHARS_PER_TOKEN = 1.2  # Indic scripts take far more tokens per character

if len(sys.argv) < 2:
    sys.exit("Usage: python3 run_summary.py output/<name>")
OUT_DIR = sys.argv[1].rstrip("/")
TRANSCRIPT = os.path.join(OUT_DIR, "transcript.json")
SPEAKERS = os.path.join(OUT_DIR, "speakers.json")


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from write_transcript import stamp  # noqa: E402  same timestamp format as transcript.txt


def tokens(text):
    indic = sum(1 for ch in text if ord(ch) > 0x7F)
    return int((len(text) - indic) / CHARS_PER_TOKEN + indic / INDIC_CHARS_PER_TOKEN) + 1


# ---------- Ollama ----------

def check_ollama():
    try:
        with urllib.request.urlopen(f"{URL}/api/tags", timeout=5) as r:
            names = {m["name"] for m in json.load(r).get("models", [])}
    except (urllib.error.URLError, OSError):
        sys.exit(f"Ollama is not running at {URL}. Start the Ollama app, or run: ollama serve")
    if MODEL not in names and f"{MODEL}:latest" not in names:
        sys.exit(f"Model {MODEL} is not downloaded. Run: ollama pull {MODEL}")


def chat(system, user, label):
    """Send one request to Ollama, stream the answer, and return the text."""
    body = {
        "model": MODEL,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "stream": True,
        "think": THINK,
        # Ollama's default context is small and it cuts long input silently, so set it explicitly.
        "options": {"num_ctx": NUM_CTX, "temperature": 0.2, "num_predict": OUTPUT_TOKENS},
    }
    req = urllib.request.Request(f"{URL}/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    parts, n, prompt_tokens = [], 0, None
    print(f"  {label}: ", end="", flush=True)
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            for line in r:
                if not line.strip():
                    continue
                msg = json.loads(line)
                if "error" in msg:
                    sys.exit(f"\nOllama error: {msg['error']}")
                parts.append(msg.get("message", {}).get("content", ""))
                n += 1
                if n % 200 == 0:
                    print(".", end="", flush=True)
                if msg.get("done"):
                    prompt_tokens = msg.get("prompt_eval_count")
    except urllib.error.HTTPError as exc:
        sys.exit(f"\nOllama error {exc.code}: {exc.read().decode(errors='replace')}")
    print(" done")
    if prompt_tokens and prompt_tokens >= NUM_CTX - 64:
        print(f"  WARNING: the prompt filled the {NUM_CTX}-token context, so Ollama may have cut it. "
              f"Raise NUM_CTX if memory allows.")
    text = "".join(parts)
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


# ---------- Transcript ----------

with open(TRANSCRIPT, encoding="utf-8") as f:
    segments = json.load(f)

if SOURCE != "original" and not any(s.get("english") for s in segments):
    print("No English translation found (was TRANSLATE=0?), so summarizing the original transcript.")
    SOURCE = "original"
SUMMARY = os.path.join(OUT_DIR, "summary.md" if SOURCE == "english" else f"summary_{SOURCE}.md")

LANG_NAMES = {"ta": "Tamil", "hi": "Hindi", "te": "Telugu", "kn": "Kannada", "ml": "Malayalam",
              "bn": "Bengali", "mr": "Marathi", "gu": "Gujarati", "pa": "Punjabi", "or": "Odia",
              "as": "Assamese", "ur": "Urdu", "ne": "Nepali"}
LANGUAGE = LANG_NAMES.get(next((s.get("lang") for s in segments if s.get("lang")), ""), "an Indian language")


def turn_text(s):
    """The text of one turn, as the model will see it for the chosen SUMMARY_SOURCE."""
    original = (s.get("text") or "").strip()
    english = (s.get("english") or "").strip()
    if SOURCE == "english":
        return english
    if SOURCE == "both" and english and english != original:
        return f"{original} (machine translation: {english})"
    return original


turns = [s for s in segments if turn_text(s)]
if not turns:
    sys.exit("The transcript has no text to summarize.")

stats = defaultdict(lambda: {"turns": 0, "seconds": 0.0, "words": 0})
for s in turns:
    st = stats[s["speaker"]]
    st["turns"] += 1
    st["seconds"] += s["end"] - s["start"]
    st["words"] += len(turn_text(s).split())

check_ollama()

# ---------- Speaker roles ----------
# speakers.json maps diarization labels to roles. On the first run the model suggests roles
# from the opening of the discussion. Edit the file and rerun to correct them.

roles = None
if os.path.exists(SPEAKERS):
    with open(SPEAKERS, encoding="utf-8") as f:
        roles = json.load(f)
    if set(roles) != set(stats):
        # The transcript was redone and the labels changed, so the old roles may name the wrong people.
        os.replace(SPEAKERS, SPEAKERS + ".old")
        print(f"speakers.json lists {sorted(roles)} but the transcript has {sorted(stats)}. "
              f"Moved it to speakers.json.old and suggesting roles again.")
        roles = None
    else:
        print(f"Using speaker roles from {SPEAKERS}")
if roles is None:
    print("No speakers.json yet, so asking the model to suggest speaker roles")
    sample, used = [], 0
    for s in turns:
        line = f"{s['speaker']}: {turn_text(s)}"
        if used + tokens(line) > 6000:
            break
        sample.append(line)
        used += tokens(line)
    talk = "\n".join(f"{spk}: {v['turns']} turns, {v['seconds'] / 60:.1f} minutes"
                     for spk, v in sorted(stats.items()))
    answer = chat(
        "You label speakers in focus group transcripts. Reply with JSON only.",
        "Speakers were labelled automatically, so labels like SPEAKER_00 carry no meaning.\n"
        "Decide which speaker is the moderator or facilitator: they usually ask the questions, "
        "introduce topics and invite others to speak. Label everyone else Participant 1, "
        "Participant 2 and so on, in order of first appearance. If someone is clearly a note-taker "
        "or observer, label them that way.\n\n"
        f"Talk time:\n{talk}\n\nOpening of the discussion:\n" + "\n".join(sample) +
        '\n\nReply with a JSON object mapping every label to a role, '
        'for example {"SPEAKER_00": "Moderator", "SPEAKER_01": "Participant 1"}.',
        "suggesting roles",
    )
    try:
        roles = json.loads(re.search(r"\{.*\}", answer, re.S).group(0))
    except (AttributeError, ValueError):
        print("  Could not read the model's suggestion, so keeping the original labels")
        roles = {}
    roles = {spk: str(roles.get(spk) or spk) for spk in sorted(stats)}
    with open(SPEAKERS, "w", encoding="utf-8") as f:
        json.dump(roles, f, indent=2)
    print(f"  Saved suggested roles to {SPEAKERS}. Check them, edit if wrong, and rerun.")

for spk, role in sorted(roles.items()):
    print(f"  {spk} -> {role}")


def name(spk):
    return roles.get(spk) or spk


lines = [f"[{stamp(s['start'])}] {name(s['speaker'])}: {turn_text(s)}" for s in turns]

# ---------- Summary ----------

SOURCE_NOTE = {
    "english": f"""The transcript was produced by speech recognition in {LANGUAGE} and English, then machine
translated into English, so some lines are garbled or awkward. Work only from what is in the transcript.
- Quotes must be copied word for word from the transcript. Do not tidy or invent quotes.""",
    "original": f"""The transcript was produced by speech recognition. It is in {LANGUAGE}, written in {LANGUAGE}
script, mixed with English words in Latin script, as people actually spoke. Some lines are garbled.
Work only from what is in the transcript, and write the summary in English.
- Quotes must copy the original {LANGUAGE} script words exactly, never a translation, followed by your
  English translation in brackets, like: "original words" (English: your translation). If the quoted
  words are already English, quote them as they are with no translation.""",
    "both": f"""The transcript was produced by speech recognition. Each turn is shown as spoken, in {LANGUAGE}
script mixed with English, followed by a machine translation in brackets. The machine translation is often
wrong on names, colloquial {LANGUAGE} and idioms. Where it disagrees with the original, trust the original
and translate it yourself. Work only from what is in the transcript, and write the summary in English.
- Quotes must copy the original {LANGUAGE} script words exactly, never the machine translation, followed
  by your own English translation in brackets, like: "original words" (English: your translation).
  If the quoted words are already English, quote them as they are with no translation.""",
}[SOURCE]

SYSTEM = f"""You are an experienced qualitative researcher writing up a focus group discussion.
{SOURCE_NOTE}

Rules:
- Every point must cite evidence as [hh:mm:ss] Speaker, copied exactly from the transcript lines.
- Say how widely a view was shared (one participant, several, most) rather than implying consensus.
- If a line is too garbled to interpret, leave it out rather than guess.
- Do not add facts, names, numbers or recommendations that nobody in the discussion said.
- Do not comment on transcription or translation quality. It is not a theme, highlight or action item.
- Write in plain English. Use Markdown."""

FORMAT = """Write the summary with exactly these sections:

## Overview
Three to five sentences: what the discussion was about, who took part, and the overall tone.

## Key themes
For each major theme (usually 3 to 7): a bold theme name, two or three sentences explaining it,
how widely it was shared, and one or two supporting quotes with [hh:mm:ss] Speaker.

## Key highlights
The most important or surprising individual insights, as bullets, each with [hh:mm:ss] Speaker.

## Points of agreement
Bullets, each with citations.

## Points of disagreement or tension
Bullets, each naming the differing views and who held them, with citations. Write "None observed" if there were none.

## Action items and recommendations
A Markdown table with columns: Item | Raised by | Owner | Source.
Owner is the person named in the discussion, or "Unassigned". Source is "Stated" if someone said it
in the discussion, or "Suggested" if it is your recommendation based on the discussion.
Put stated items first. Each item needs a [hh:mm:ss] citation.

## Open questions and follow-ups
Questions left unresolved, or topics worth exploring in a future session, with citations."""

budget = NUM_CTX - OUTPUT_TOKENS - PROMPT_OVERHEAD
full = "\n".join(lines)
print(f"Transcript: {len(lines)} turns, about {tokens(full)} tokens; context budget {budget} tokens")

if tokens(full) <= budget:
    body = chat(SYSTEM, f"{FORMAT}\n\nTranscript:\n{full}", "writing summary")
else:
    # Too long for one pass: take notes on each part, then summarize the notes.
    chunks, current, used = [], [], 0
    for line in lines:
        if current and used + tokens(line) > budget:
            chunks.append(current)
            current, used = [], 0
        current.append(line)
        used += tokens(line)
    chunks.append(current)
    print(f"Too long for one pass, so working in {len(chunks)} parts")
    notes = []
    for k, chunk in enumerate(chunks, 1):
        notes.append(chat(
            SYSTEM,
            f"This is part {k} of {len(chunks)} of the discussion. Write detailed notes on it: "
            "themes, notable views, agreements, disagreements, action items and open questions. "
            "Keep every [hh:mm:ss] Speaker citation and copy the best quotes word for word.\n\n"
            "Transcript part:\n" + "\n".join(chunk),
            f"notes on part {k}/{len(chunks)}",
        ))
    # If all the notes together are still too long, condense neighbouring notes until they fit.
    while len(notes) > 1 and tokens("\n\n".join(notes)) > budget:
        print(f"Notes are too long for one pass ({tokens(chr(10).join(notes))} tokens), so condensing them")
        merged = []
        for k in range(0, len(notes), 2):
            pair = notes[k:k + 2]
            if len(pair) == 1:
                merged.append(pair[0])
                continue
            merged.append(chat(
                SYSTEM,
                "Combine these two sets of consecutive notes into one shorter set. Keep the themes, "
                "agreements, disagreements, action items and open questions, the best quotes word for "
                "word, and every [hh:mm:ss] Speaker citation you keep.\n\n" + "\n\n".join(pair),
                f"condensing notes {k + 1}-{k + 2}",
            ))
        notes = merged

    body = chat(
        SYSTEM,
        f"{FORMAT}\n\nWork from these notes on the parts of the discussion, in order. "
        "Merge repeated themes across parts and keep the citations.\n\n" +
        "\n\n".join(f"### Notes on part {k}\n{n}" for k, n in enumerate(notes, 1)),
        "writing summary from notes",
    )

# ---------- Checks and output ----------

# Flag citations that do not match the start of any turn, as a hint of invented evidence.
valid = {stamp(s["start"]) for s in turns}
cited = re.findall(r"\[(\d{2}:\d{2}:\d{2})\]", body)
unknown = sorted(set(c for c in cited if c not in valid))
if unknown:
    print(f"WARNING: {len(unknown)} cited timestamp(s) do not match any turn: {', '.join(unknown[:10])}")

total = sum(v["seconds"] for v in stats.values()) or 1
table = ["| Speaker | Label | Turns | Talk time | Share |", "|---|---|---|---|---|"]
for spk, v in sorted(stats.items(), key=lambda kv: -kv[1]["seconds"]):
    table.append(f"| {name(spk)} | {spk} | {v['turns']} | {stamp(v['seconds'])} | {v['seconds'] / total:.0%} |")

duration = stamp(max(s["end"] for s in turns))
with open(SUMMARY, "w", encoding="utf-8") as f:
    f.write(f"# Focus group summary: {os.path.basename(OUT_DIR)}\n\n")
    f.write(f"Recording length {duration}, {len(stats)} speakers, {len(turns)} turns. "
            f"Generated {datetime.date.today()} by {MODEL} from the "
            f"{ {'english': 'English translation', 'original': 'original transcript', 'both': 'original transcript with translation hints'}[SOURCE] }.\n\n")
    f.write("> Machine-generated from an automatic transcript and translation. "
            "Check quotes against transcript.txt before relying on them.\n\n")
    f.write(body + "\n\n")
    f.write("## Participation\n\n" + "\n".join(table) + "\n")
    if unknown:
        f.write(f"\n> {len(unknown)} cited timestamp(s) did not match a transcript turn: "
                f"{', '.join(unknown)}. Check those points.\n")
print(f"Saved {SUMMARY}")
