# Decisions

Why the pipeline works the way it does. Each entry records the decision, the options we weighed, and the evidence. The newest decisions are at the top. When a decision changes, add a new entry that supersedes the old one rather than editing it.

Tests were run on an Apple M3 Pro with 18 GB of memory on macOS 27, using a 2-minute clip from an hour-long group discussion in Tamil and English.

| # | Decision | Status |
|---|---|---|
| 8 | [Summary source: English, original, or both](#8-summary-source-english-original-or-both) | Trial; default stays `english` pending a full-recording comparison |
| 7 | [Do not speed up the audio](#7-do-not-speed-up-the-audio) | Accepted |
| 6 | [Pin SciPy 1.14.1 on macOS](#6-pin-scipy-1141-on-macos) | Accepted |
| 5 | [Qwen 3.5 9B in Ollama for the summary](#5-qwen-35-9b-in-ollama-for-the-summary) | Accepted |
| 4 | [Speaker roles suggested by the model, corrected by a person](#4-speaker-roles-suggested-by-the-model-corrected-by-a-person) | Accepted |
| 3 | [Romanized transcript off by default](#3-romanized-transcript-off-by-default) | Accepted |
| 2 | [Translation on by default](#2-translation-on-by-default) | Accepted |
| 1 | [One detected language per recording](#1-one-detected-language-per-recording) | Accepted |

---

## 8. Summary source: English, original, or both

*2026-11-12. Trial.*

**Problem.** Machine translation of colloquial Tamil is the weakest link in the pipeline. IndicTrans2 turned a participant's self-introduction ("my name is ...") into "my people have come to ...". It also mangled a technical term. The summary model then repeated these mistakes.

**Options.** `SUMMARY_SOURCE` lets the summary model read one of three inputs:

- `english`: the IndicTrans2 translation.
- `original`: the transcript as spoken, in Tamil script mixed with English.
- `both`: the original, with the translation in brackets as a hint. The model is told to trust the original when the two disagree.

**Evidence.** We ran all three on the 2-minute clip with `qwen3.5:9b` and the same speaker roles. `both` read best:

- `english` kept the mistranslated introduction and flagged it as garbled. Its quotes were the translation, mistakes included.
- `original` read the name correctly. On the first prompt it quoted its own English translation instead of the Tamil, and once left the placeholder "(English: your translation)" in the text. It also read more into a garbled line than was there.
- `both` got the name and role right, quoted the Tamil word for word with a correct translation, and added nothing that wasn't said.

The first version of the prompt had two problems that affected `original` and `both`. Two prompt rules, which now apply to all three options, fixed them:

- **Quotes.** The model quoted its own English translation as if it were the speaker's words. Quotes must now be in the original script, and English speech is quoted as it is.
- **Meta-commentary.** The `both` summary invented a "speech recognition challenges" theme and an action item to "calibrate speech recognition tools". Comments on transcription or translation quality are now banned.

**Decision.** Ship all three options. Keep `english` as the default until all three have been compared on the full recording, since a 2-minute clip is a small sample. If `both` still wins, make it the default and record that as a new entry.

**Costs.** Indic script takes several times more tokens than English, so `original` and `both` fit less of the transcript into the context window. The summary script estimates this and splits long transcripts into parts.

## 7. Do not speed up the audio

*2026-11-12. Accepted.*

**Question.** Speeding the audio up 1.5x or 2x before transcription often works for English with Whisper. Does it work here?

**Evidence.** We ran the same clip at 1x and 1.5x and compared the transcripts word for word with `compare_runs.py`, treating 1x as the reference:

| | 1x | 1.5x |
|---|---|---|
| Run time | 155 s | 88 s |
| Speaker turns found | 13 | 10 |
| Word changes in English-speaking turns | | 6% |
| Word changes in Tamil turns | | 62% |

English came through almost unchanged. Tamil did not:

- A participant's name came out as a different name.
- An English word was replaced by an unrelated phrase.
- One Tamil line was replaced by "Chiping Chow butter, idli, mutton", which nobody said.
- Three short turns disappeared.

**Decision.** Keep 1x. `SPEED` stays as an option, off by default, for recordings where the Indic speech matters less. Speed was not the problem anyway: at 1x, an hour of audio transcribes in roughly 75 to 90 minutes on the M3 Pro.

**Follow-up.** The code review found that the turn-length thresholds (minimum turn length, merge gap, padding) had been applied on the sped-up timeline. They are now scaled by `SPEED`. That may recover some of the lost short turns, but the Tamil errors come from the transcription model itself.

## 6. Pin SciPy 1.14.1 on macOS

*2026-11-12. Accepted.*

**Problem.** In `env-asr`, `import scipy.sparse.linalg` failed on macOS 27 with:

```
dlopen(.../scipy/sparse/linalg/_propack/_spropack.cpython-310-darwin.so): section '__DATA/__thread_bss' has a zero-fill section type, but offset field is not zero
```

The macOS loader rejects the compiled file in the SciPy 1.15.x wheels. SciPy 1.15.3 is the newest release that supports Python 3.10.

**Options.**

- Rebuild the environments on Python 3.12 to get a newer SciPy. That means retesting every pinned dependency, including the `transformers==4.46.3` pin for IndicTrans2.
- Use an older SciPy.

**Evidence.** SciPy 1.14.1 and 1.13.1 both load on macOS 27. pip then warns that `pyannote-metrics 4.1 requires scipy>=1.15.1`. We ignore the warning: pyannote-metrics only scores diarization against reference labels, which this pipeline never does, and diarization ran correctly on the test clip.

**Decision.** `setup.sh` tests whether SciPy loads and installs 1.14.1 only if it doesn't. This avoids pinning an old version on systems where the newer one works.

## 5. Qwen 3.5 9B in Ollama for the summary

*2026-11-11. Accepted.*

**Constraint.** The summary must run on an 18 GB Mac without sending the transcript anywhere, with enough context for an hour-long transcript (about 15,000 to 20,000 tokens in English).

**Options.**

- **DeepSeek V3 or R1, GLM-4.5 or GLM-4.6.** These are 100B to 671B parameter models and don't fit in 18 GB even when quantized. Their small distilled versions are weaker than Qwen at the same size.
- **Gemma.** A reasonable alternative. We chose the Qwen family on preference.
- **Qwen 3.5.** Ollama offers 0.8B, 2B, 4B and 9B, then 27B and up. There is no 14B, even though one comparison site listed it. 27B needs about 17 GB at 4-bit, which is too much.

**Decision.**

- Use `qwen3.5:9b` (6.6 GB, 256K native context), with `qwen3:14b` (9.3 GB) as the larger alternative.
- Serve it through Ollama, which uses the Mac's GPU.
- Set Ollama's context to 32,768 tokens explicitly. Its default is much smaller and silently cuts long input.
- Turn Qwen's thinking mode off by default for speed. `SUMMARY_THINK=1` turns it back on.

**Safeguards.** The summary model can invent things, so:

- Every point must cite `[hh:mm:ss] Speaker`.
- After writing, the script checks every cited timestamp against the transcript and lists any that don't match a turn.
- Action items are marked Stated (someone said it) or Suggested (the model inferred it).
- The participation table is computed from the transcript, not written by the model.

## 4. Speaker roles suggested by the model, corrected by a person

*2026-11-11. Accepted.*

**Problem.** Diarization labels speakers `SPEAKER_00`, `SPEAKER_01` and so on. A focus group summary needs to know who the moderator is.

**Decision.** On the first run, the model reads the opening of the discussion and talk-time figures, then suggests roles in `speakers.json`. A person checks and edits the file, then reruns only the summary, which takes minutes. If a later transcript has different speaker labels, the old file is moved to `speakers.json.old` and roles are suggested again, so old roles are never attached to the wrong people.

**Evidence.** On the test clip the model identified the moderator correctly. It labelled a one-line speaker as "Note-taker", which is probably wrong, so the human check is needed.

## 3. Romanized transcript off by default

*2026-11-12. Accepted.*

**Background.** Indic-Transcribe-Flex can write a turn in Latin script (`mode="romanized"`) as well as in mixed script (`mode="mixed"`). Getting both meant transcribing every turn twice.

**Decision.** Run only the mixed-script pass by default. `ROMANIZE=1` adds the second pass. This halves transcription time, the slowest stage on a Mac. Nothing downstream needs the romanized text.

## 2. Translation on by default

*2026-11-11. Accepted.*

We briefly made translation opt-in, then switched back. The English translation is what most readers of the transcript and summary need. It is also cheap: translating the clip took under 2 minutes on the CPU, compared with over an hour to transcribe a full recording. `TRANSLATE=0` skips it, along with the second Python environment and the 7.6 GB model download.

We also considered dropping translation and letting Qwen read the original. We kept translation because the English transcript lets people who don't read Tamil check quotes. #8 covers how the summary uses it.

## 1. One detected language per recording

*2026-11-11. Accepted.*

**Problem.** Indic-Transcribe-Flex needs a language code. Detecting the language turn by turn is unreliable on short code-mixed turns: on the test clip, language detection put Telugu, Marathi and Konkani behind Tamil.

**Decision.** With `ASR_LANG=auto`, run language detection on the 40 longest turns. Weight each vote by turn length, ignore English, and use the winner for the whole recording. On the test clip it chose Tamil, with 37 seconds of votes against 3.6 for the runner-up, Telugu. A recording that mixes two Indian languages will transcribe the less common one poorly. `ASR_LANG` overrides the detection.
