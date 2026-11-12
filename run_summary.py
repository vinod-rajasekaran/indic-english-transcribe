# Stage 5: summary with a local LLM served by Ollama. Fully local. The prompt depends on PURPOSE.
# Uses only the standard library, so it runs with system python3.
#
# Run with: PURPOSE=fgd python3 run_summary.py output/<name>
#
# Reads  output/<name>/transcript.json   (written by the transcription stages)
# Uses   output/<name>/speakers.json     (speaker label -> role; created on first run)
# Writes output/<name>/summary_<purpose>.md
#
# PURPOSE (required) says what the recording is and so which prompt to use: the name of a file in
# prompts/ (fgd for a focus group discussion, meeting for meeting notes), or the path to your own
# prompt file. See prompts/README.md for the file format.
#
# Options (environment variables):
#   SUMMARY_MODEL  Ollama model tag            default qwen3.5:9b
#   NUM_CTX        context window in tokens    default 65536 (see DECISIONS.md #12)
#   OLLAMA_URL     Ollama server address       default http://localhost:11434
#   SUMMARY_THINK  1 lets Qwen reason before answering (slower, sometimes better)
#   SUMMARY_SOURCE what the model reads:
#                    both      the original with the translation as a hint (default) -> summary_<purpose>.md
#                    english   the IndicTrans2 translation only                -> summary_<purpose>_english.md
#                    original  the transcript as spoken, Indic + English       -> summary_<purpose>_original.md
#                  See DECISIONS.md #9 for why both is the default.
import datetime
import json
import os
import re
import sys
import urllib.error
import urllib.request
from collections import defaultdict

MODEL = os.environ.get("SUMMARY_MODEL", "qwen3.5:9b")
NUM_CTX = int(os.environ.get("NUM_CTX", 65536))
URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
THINK = os.environ.get("SUMMARY_THINK", "0") == "1"
SOURCE = os.environ.get("SUMMARY_SOURCE", "both")
if SOURCE not in ("english", "original", "both"):
    sys.exit("SUMMARY_SOURCE must be english, original or both")
OUTPUT_TOKENS = 4096     # room kept for the model's answer
PROMPT_OVERHEAD = 3000   # instructions around the transcript, given before and after it
CHARS_PER_TOKEN = 3.5    # rough estimate for English text; errs on the safe side
INDIC_CHARS_PER_TOKEN = 1.2  # Indic scripts take far more tokens per character

if len(sys.argv) < 2:
    sys.exit("Usage: python3 run_summary.py output/<name>")
OUT_DIR = sys.argv[1].rstrip("/")

# ---------- Purpose prompt ----------
# The purpose-specific parts of the prompt live in prompts/<purpose>.md, in sections marked
# [title], [role], [rules], [speakers], [notes] and [format]. The rules that keep the summary
# faithful to the transcript (citations, no invented facts, how to quote) stay in this script.

PROMPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")
PROMPT_SECTIONS = ("title", "role", "rules", "speakers", "notes", "format")
available = sorted(f[:-3] for f in os.listdir(PROMPT_DIR) if f.endswith(".md") and f != "README.md")
PURPOSE = os.environ.get("PURPOSE", "").strip()
if not PURPOSE:
    sys.exit(f"Set PURPOSE to say what the recording is, for example PURPOSE=fgd. "
             f"Available: {', '.join(available)}, or the path to your own prompt file.")
PROMPT_FILE = PURPOSE if os.path.isfile(PURPOSE) else os.path.join(PROMPT_DIR, PURPOSE + ".md")
if not os.path.isfile(PROMPT_FILE):
    sys.exit(f"No prompt file for PURPOSE={PURPOSE}. Available: {', '.join(available)}, "
             f"or the path to your own prompt file.")
PURPOSE_NAME = os.path.splitext(os.path.basename(PROMPT_FILE))[0]

prompt, section = {}, None
with open(PROMPT_FILE, encoding="utf-8") as f:
    for line in f:
        m = re.fullmatch(r"\[(\w+)\]\s*", line)
        if m:
            section = m.group(1)
            prompt[section] = ""
        elif section:
            prompt[section] += line
prompt = {k: v.strip() for k, v in prompt.items()}
missing = [k for k in PROMPT_SECTIONS if not prompt.get(k)]
if missing:
    sys.exit(f"{PROMPT_FILE} is missing these sections: {', '.join('[' + k + ']' for k in missing)}")
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

# The default source (both) writes summary_<purpose>.md, even when it falls back to the original below.
SUMMARY = os.path.join(OUT_DIR, f"summary_{PURPOSE_NAME}" + ("" if SOURCE == "both" else f"_{SOURCE}") + ".md")
if SOURCE != "original" and not any(s.get("english") for s in segments):
    print("No English translation found (was TRANSLATE=0?), so summarizing the original transcript.")
    SOURCE = "original"
print(f"Purpose: {PURPOSE_NAME} ({PROMPT_FILE})")

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
        "You label speakers in transcripts. Reply with JSON only.",
        "Speakers were labelled automatically, so labels like SPEAKER_00 carry no meaning.\n"
        f"{prompt['speakers']}\n\n"
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

SYSTEM = f"""{prompt['role']}
{SOURCE_NOTE}

Rules:
- Every point must cite evidence as [hh:mm:ss] Speaker in square brackets, copied exactly from the transcript lines.
{prompt['rules']}
- If a line is too garbled to interpret, leave it out rather than guess.
- Do not add facts, names, numbers or recommendations that nobody in the discussion said.
- Do not comment on transcription or translation quality. It is not a theme, highlight or action item.
- Write in plain English. Use Markdown."""

FORMAT = prompt["format"]

# With tens of thousands of tokens of transcript, instructions given only before it get lost: in
# testing, a one-pass summary dropped the square-bracket citations and the quoting rule. So the
# instructions are repeated after the transcript as well.
REMINDER = f"Now write it. Your instructions again:\n\n{SYSTEM}\n\n{FORMAT}"

budget = NUM_CTX - OUTPUT_TOKENS - PROMPT_OVERHEAD
full = "\n".join(lines)
print(f"Transcript: {len(lines)} turns, about {tokens(full)} tokens; context budget {budget} tokens")

if tokens(full) <= budget:
    body = chat(SYSTEM, f"{FORMAT}\n\nTranscript:\n{full}\n\n{REMINDER}", "writing summary")
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
            f"{prompt['notes']}. "
            "Keep every [hh:mm:ss] Speaker citation and copy the best quotes word for word.\n\n"
            "Transcript part:\n" + "\n".join(chunk) +
            "\n\nNow write the notes. Keep every [hh:mm:ss] Speaker citation exactly as it appears above.",
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
                f"Combine these two sets of consecutive notes into one shorter set. Keep the {prompt['notes']}, "
                "the best quotes word for word, and every [hh:mm:ss] Speaker citation you keep.\n\n"
                + "\n\n".join(pair),
                f"condensing notes {k + 1}-{k + 2}",
            ))
        notes = merged

    body = chat(
        SYSTEM,
        f"{FORMAT}\n\nWork from these notes on the parts of the discussion, in order. "
        "Merge repeated themes across parts and keep the citations.\n\n" +
        "\n\n".join(f"### Notes on part {k}\n{n}" for k, n in enumerate(notes, 1)) +
        "\n\nCite only [hh:mm:ss] Speaker citations that appear in the notes above. Never make up a time.\n\n"
        + REMINDER,
        "writing summary from notes",
    )

# ---------- Checks and output ----------

# Flag citations that do not match the start of any turn, as a hint of invented evidence.
valid = {stamp(s["start"]) for s in turns}
cited = re.findall(r"\[(\d{2}:\d{2}:\d{2})\]", body)
unknown = sorted(set(c for c in cited if c not in valid))
if not cited:
    print("WARNING: the summary has no [hh:mm:ss] citations, so none of its points can be checked.")
if unknown:
    print(f"WARNING: {len(unknown)} cited timestamp(s) do not match any turn: {', '.join(unknown[:10])}")

total = sum(v["seconds"] for v in stats.values()) or 1
table = ["| Speaker | Label | Turns | Talk time | Share |", "|---|---|---|---|---|"]
for spk, v in sorted(stats.items(), key=lambda kv: -kv[1]["seconds"]):
    table.append(f"| {name(spk)} | {spk} | {v['turns']} | {stamp(v['seconds'])} | {v['seconds'] / total:.0%} |")

duration = stamp(max(s["end"] for s in turns))
with open(SUMMARY, "w", encoding="utf-8") as f:
    f.write(f"# {prompt['title']}: {os.path.basename(OUT_DIR)}\n\n")
    f.write(f"Recording length {duration}, {len(stats)} speakers, {len(turns)} turns. "
            f"Generated {datetime.date.today()} by {MODEL} from the "
            f"{ {'english': 'English translation', 'original': 'original transcript', 'both': 'original transcript with translation hints'}[SOURCE] }.\n\n")
    f.write("> Machine-generated from an automatic transcript and translation. "
            "Check quotes against transcript.txt before relying on them.\n\n")
    f.write(body + "\n\n")
    f.write("## Participation\n\n" + "\n".join(table) + "\n")
    if not cited:
        f.write("\n> This summary has no [hh:mm:ss] citations, so its points could not be checked.\n")
    if unknown:
        f.write(f"\n> {len(unknown)} cited timestamp(s) did not match a transcript turn: "
                f"{', '.join(unknown)}. Check those points.\n")
print(f"Saved {SUMMARY}")
