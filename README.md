# indic-english-transcribe

A fully local pipeline that takes a recording of several people speaking an Indian language mixed with English, works out who spoke when, transcribes each turn, translates it into English, and writes a focus group summary with a local Qwen model. Tamil was the first target, but any of the 22 scheduled Indian languages that Indic-Transcribe-Flex supports will work, including Hindi, Telugu, Kannada, Malayalam, Bengali and Marathi. No audio or text leaves your machine once the models are downloaded.

For why it works this way, including the models chosen, the tests we ran and the options we rejected, see [DECISIONS.md](DECISIONS.md).

## What it does

The pipeline runs in four stages. First, pyannote community-1 labels each stretch of speech with a speaker and a start and end time. Second, Indic-Transcribe-Flex transcribes each speaker turn in mixed-script mode, which writes Indic words in their own script and English words in Latin script, for example `நாளைக்கு meeting இருக்கு`. With `ROMANIZE=1` it also transcribes each turn in romanized mode, which writes the whole turn in Latin script (`naalaikku meeting irukku`). This second pass roughly doubles transcription time, so it is off by default.

The language is detected automatically. The script runs language ID on every turn and uses the most common non-English language for the whole recording. Set `ASR_LANG` to skip detection, for example `ASR_LANG=hi`.

Third, IndicTrans2 translates each turn into English. Turns written entirely in Latin script pass through unchanged. Translation runs in a second Python environment, because IndicTrans2 needs an older transformers release.

Fourth, a Qwen model running in Ollama reads the English transcript and writes a focus group summary: themes, highlights, agreement and disagreement, action items and open questions, each point cited with a timestamp and speaker. See [Focus group summary](#focus-group-summary).

## Requirements

Either of these:

- **Mac** with Apple silicon, 16 GB of memory or more, and Homebrew. Tested on an M3 Pro with 18 GB running macOS 27.
- **Linux**: Ubuntu or WSL2 with an NVIDIA GPU with 12 GB of VRAM or more.

You also need about 30 GB of free disk space (the models take about 23 GB) and internet access for the one-time setup. A free Hugging Face account is needed to accept the licences of two gated models.

**What to expect on a Mac.** Diarization and translation run on the Mac's GPU, transcription runs on the CPU, and the summary model runs on the GPU through Ollama. On the M3 Pro, a 2-minute clip takes about a minute from start to translated transcript, including loading the models. An hour-long recording took 114 minutes on the CPU alone (diarization 43, transcription 26, translation 45), plus about 40 minutes for the summary. With the GPU split the clip ran 4 times faster, so expect roughly 40 to 50 minutes for the transcript, but this hasn't been measured on a full recording yet ([decision 10](DECISIONS.md#10-use-the-mac-gpu-for-diarization-and-translation)).

| Model | Purpose | Download |
|---|---|---|
| `pyannote/speaker-diarization-community-1` | Who spoke when | 32 MB |
| `bodhan-ai/Indic-Transcribe-Flex` | Speech to text | 9 GB |
| `ai4bharat/indictrans2-indic-en-1B` | Text to English | about 8 GB |
| `qwen3.5:9b` (Ollama) | Focus group summary | 6.6 GB |

## Setup

Accept the licence on the Hugging Face pages for `pyannote/speaker-diarization-community-1` and `bodhan-ai/Indic-Transcribe-Flex`, then create a read token at hf.co/settings/tokens. Then run:

```bash
bash setup.sh
echo "HF_TOKEN=hf_your_token_here" > .env
./env-asr/bin/python download_models.py
```

`setup.sh` detects macOS or Linux:

- **On a Mac**, it installs ffmpeg, Python 3.10 and Ollama with Homebrew if they are missing.
- **On Linux**, it uses apt.

It then creates the two Python environments, `env-asr` and `env-translate`. It checks that each one imports cleanly, and works around a SciPy problem on recent macOS (see [Troubleshooting](#troubleshooting)).

`download_models.py` reads the token from `.env`, which git ignores. An exported `HF_TOKEN` also works. It downloads the models one at a time: diarization, then transcription, then translation. If it is interrupted, run it again and it resumes.

If you only want the transcript in the original language, add `TRANSLATE=0` to both setup commands: `TRANSLATE=0 bash setup.sh` and `TRANSLATE=0 ./env-asr/bin/python download_models.py`. This skips the second environment and the IndicTrans2 download, which saves a few GB of disk. Then run with `TRANSLATE=0` as well.

For the summary, start Ollama and download the model. On a Mac, `setup.sh` has already installed Ollama. On Linux, install it from ollama.com/download first.

```bash
brew services start ollama     # Mac: starts Ollama now and at login. Or open the Ollama app, or run: ollama serve
ollama pull qwen3.5:9b
```

After the downloads finish you can disconnect the machine from the network.

## Walkthrough: from recording to summary

1. Accept the two model licences on Hugging Face and create a read token, as described in Setup.
2. Run `bash setup.sh` and put `HF_TOKEN=hf_...` in `.env`.
3. Run `./env-asr/bin/python download_models.py`. This takes a while; you can do step 4 and 5 meanwhile.
4. Start Ollama and run `ollama pull qwen3.5:9b`.
5. Copy the recordings into `input/`.
6. Try a 2-minute clip first, to check the language, the translation and the summary format. Starting a minute in skips setup chatter:

   ```bash
   ffmpeg -ss 60 -i input/part-1.m4a -t 120 -c copy input/test.m4a
   bash run_all.sh test test.m4a
   ```

7. Read `output/test/transcript.txt` and `output/test/summary.md`.
8. Run the full discussion, with the speaker range set to the number of people in the room:

   ```bash
   MIN_SPEAKERS=2 MAX_SPEAKERS=10 bash run_all.sh fgd-1 part-1.m4a part-2.m4a part-3.m4a part-4.m4a
   ```

   On a Mac this runs for over an hour. To keep it running after you close the terminal, add `nohup` at the start and `> work/fgd-1.log 2>&1 &` at the end, then follow progress with `tail -f work/fgd-1.log`. To transcribe now and summarize later, add `SUMMARIZE=0`.

9. Open `output/fgd-1/speakers.json`. The model has suggested a role for each speaker label. Correct any that are wrong. Listening to a few turns from `transcript.txt` helps, since each turn gives the source file and time.
10. Rerun only the summary, which takes minutes rather than hours:

    ```bash
    python3 run_summary.py output/fgd-1
    ```

## Run

Copy your recordings into `input/`, then run the script with an output name followed by the files in the order they should be joined:

```bash
bash run_all.sh standup part1.mp4 part2.mp4 part3.mp4
```

A single recording works the same way: `bash run_all.sh standup meeting.mp4`. Set the expected speaker range for better speaker counting:

```bash
MIN_SPEAKERS=2 MAX_SPEAKERS=5 bash run_all.sh standup part1.mp4 part2.mp4
```

Other options are set the same way:

| Variable | Default | Meaning |
|---|---|---|
| `ASR_LANG` | `auto` | Language code such as `ta`, `hi`, `te`, `kn`, `ml`, `bn`, `mr`. `auto` detects it |
| `TRANSLATE` | `1` | Set to `0` to skip the English translation |
| `MIN_SPEAKERS`, `MAX_SPEAKERS` | `2`, `6` | Expected speaker range |
| `ROMANIZE` | `0` | Set to `1` to add a romanized (Latin script) version of each turn. Roughly doubles transcription time |
| `SPEED` | `1` | Speed the audio up (for example `1.5`) before diarization and transcription. Not recommended for Indic speech: see [Speeding up the audio](#speeding-up-the-audio) |
| `SUMMARIZE` | `1` | Set to `0` to skip the focus group summary |
| `SUMMARY_MODEL` | `qwen3.5:9b` | Ollama model used for the summary |
| `NUM_CTX` | `32768` | Context window for the summary model, in tokens |
| `SUMMARY_SOURCE` | `both` | What the summary model reads: `both` (the original with the translation as a hint), `english` (the translation only), or `original` (the transcript as spoken). See [Focus group summary](#focus-group-summary) |
| `DEVICE` | `auto` | `auto` uses an NVIDIA GPU if there is one. On a Mac it uses the GPU for diarization and translation and the CPU for transcription. `cpu`, `mps` or `cuda` forces every stage onto that device |
| `SUMMARY_THINK` | `0` | Set to `1` to let Qwen reason before answering. Slower, sometimes more careful |

Results are written to `output/standup/`:

| File | Contents |
|---|---|
| `segments.json` | Ordered segments with index, speaker, start, end, language, source file, source start, original text (`text`), and romanized text (`roman`) with `ROMANIZE=1` |
| `transcript.txt` | Readable transcript with timestamp, speaker, source file, original text, romanized text (with `ROMANIZE=1`), and English translation (omitted with `TRANSLATE=0`) |
| `transcript.json` | The same data as JSON, with an `english` field (omitted with `TRANSLATE=0`) |
| `speakers.json` | Role for each speaker label, such as `"SPEAKER_00": "Moderator"`. Suggested by the model on the first run, then yours to edit |
| `summary.md` | The focus group summary. With `SUMMARY_SOURCE=english` or `original` it is written to `summary_english.md` or `summary_original.md` instead, so you can compare versions |

## Focus group summary

`run_summary.py` sends the transcript to a Qwen model in Ollama, which runs on your machine, so the summary is as private as the rest of the pipeline. Each turn goes to the model with its timestamp and speaker role.

**What the model reads.** `SUMMARY_SOURCE` has three options, because machine translation of colloquial speech is the weakest link in the pipeline.

| Option | The model reads | Writes |
|---|---|---|
| `both` (default) | The original, with the translation in brackets as a hint. The model is told to trust the original when they disagree | `summary.md` |
| `english` | The IndicTrans2 English translation only | `summary_english.md` |
| `original` | The transcript as spoken: Indic script mixed with English | `summary_original.md` |

With `both` and `original`, quotes are given in the original script followed by the model's own English translation. You can produce all three versions from one transcript without redoing transcription, for example: `SUMMARY_SOURCE=english python3 run_summary.py output/<name>`.

On an hour-long Tamil and English group discussion, `both` gave the richest summary with the most accurate citations, but took about 40 minutes against 6 for `english`, because Tamil script needs about three times as many tokens. Use `SUMMARY_SOURCE=english` when you need a quick first look. The comparison is in [decision 9](DECISIONS.md#9-summarize-from-both-the-original-and-the-translation).

**Speaker roles.** Diarization labels speakers `SPEAKER_00`, `SPEAKER_01` and so on. On the first run, the model reads the opening of the discussion and suggests who is the moderator and who are participants, and saves this to `speakers.json`. Edit the file to fix mistakes or to use names or descriptions such as `"Participant 3 (teacher)"`, then rerun `python3 run_summary.py output/<name>`. Delete the file to have the model suggest roles again.

**Summary format.** `summary.md` has an overview, key themes with supporting quotes, key highlights, points of agreement, points of disagreement, an action items table, and open questions. Every point cites `[hh:mm:ss] Speaker`. The action items table marks each item as Stated, meaning someone said it, or Suggested, meaning the model inferred it. A participation table at the end, computed directly from the transcript, shows each speaker's turns and share of talk time.

**Checks.** The script sets Ollama's context window to 32,768 tokens, because Ollama's smaller default would silently cut a long transcript. It warns if the prompt still fills the window. An hour-long discussion fits in one pass from the English translation, but not from the original script, which takes several times more tokens. A transcript that doesn't fit is split into parts, the model takes notes on each part, and then writes the summary from the notes. After writing, the script checks every cited timestamp against the transcript and lists any that match no turn, which is a sign of an invented point or a misplaced citation.

**Choosing a model.** The default is `qwen3.5:9b`. To try the larger `qwen3:14b`, run `ollama pull qwen3:14b`, then `SUMMARY_MODEL=qwen3:14b python3 run_summary.py output/<name>`. The 27B and larger Qwen models need 32 GB of memory or more. See [decision 5](DECISIONS.md#5-qwen-35-9b-in-ollama-for-the-summary) for why not DeepSeek or GLM.

## Speeding up the audio

`SPEED=1.5` speeds the audio up before diarization and transcription. Pitch is kept, and timestamps are scaled back to the original recording. We don't recommend it for Indic speech. In our test, English came through with 6% of words changed, but 62% of Tamil words changed, including a participant's name ([decision 7](DECISIONS.md#7-do-not-speed-up-the-audio)). To test it on your own audio, run the same clip at both speeds and compare:

```bash
bash run_all.sh test-1x test.m4a
SPEED=1.5 bash run_all.sh test-15x test.m4a
python3 compare_runs.py output/test-1x output/test-15x
```

`compare_runs.py` reports the word error rate of the second run against the first, for the original text and the English. It lists the changed words and shows both versions of each moment side by side.

## Multiple files

The script converts each file to 16 kHz mono WAV, then joins them into one file in the order you typed them. The order on the command line is the only thing that decides sequence. The script does not read file names or modified dates. If you use a shell glob such as `part*.mp4`, the shell sorts names alphabetically, so `part10` lands before `part2` unless you number with leading zeros.

Joining first has two benefits. The timeline runs continuously across all parts, and speaker labels stay consistent, so `SPEAKER_00` in part 1 is the same person as `SPEAKER_00` in part 3. Segment `start` and `end` values are positions on the joined timeline. Each segment also carries `source_file` and `source_start`, which point to the moment in the original recording. A speaker turn that crosses a file boundary is assigned to the file where it begins.

## How the timeline is ordered

pyannote reports every stretch of speech with a start and end time in seconds from the beginning of the file. `run_asr.py` sorts those stretches by start time, merges neighbouring stretches from the same speaker when the gap is under one second, and transcribes them in that order. Each segment carries an `index` and its timestamps through to the final transcript. The script uses pyannote's exclusive diarization, which assigns each moment to a single speaker, so segments never overlap. The cost is that when two people talk at once, only one of them is transcribed for that moment.

## Known limits

The Flex model transcribes single-speaker audio only, which is why diarization runs first. Each run handles one Indian language plus English. A group where some people speak Tamil and others Telugu will have the less common language transcribed poorly (see [Hypotheses](#hypotheses)). Language detection can confuse related languages on short recordings, so set `ASR_LANG` when you know the language. With `ROMANIZE=1`, romanized spelling is not standardized, so the same word may be spelled differently across turns. Mixed-script mode can occasionally mishandle numbers. The summary can only be as good as the translation: a garbled turn may be left out or misread, so check quotes against `transcript.txt` before relying on them. Speaker roles depend on diarization, which can split one person into two labels or merge two people into one. The pinned transformers version for translation (4.46.3) is untested on every setup, so loosen it in `setup.sh` if installation fails.

## Hypotheses

Ideas we expect to help but have not built or tested yet.

**Detect the language per speaker for multilingual groups.** Today the pipeline detects one Indian language for the whole recording ([decision 1](DECISIONS.md#1-one-detected-language-per-recording)). In a group where some people speak Tamil and others Telugu, the minority language would be forced into the wrong script and vocabulary.

- **Hypothesis.** People usually keep to one language through a discussion. Running language detection on each speaker's longest turns, and giving each speaker their own language, would fix mixed groups without the errors of detecting language turn by turn. Turn-by-turn detection misread short code-mixed Tamil as Telugu, Marathi or Konkani on our test clip.
- **What would change.** Only the language-detection step in `run_asr.py`. Translation already groups turns by language.
- **How to test.** Run a recording with speakers of two different Indian languages, and compare per-speaker detection against a single `ASR_LANG`.

## Troubleshooting

**`ImportError: dlopen(... scipy/sparse/linalg/_propack/_spropack...so ...): section '__DATA/__thread_bss' has a zero-fill section type, but offset field is not zero`**

SciPy 1.15.x is the newest SciPy for Python 3.10, and its Mac build does not load on macOS 27. `setup.sh` detects this and installs SciPy 1.14.1, which works. To fix an existing environment by hand, run:

```bash
./env-asr/bin/pip install "scipy==1.14.1"
```

pip then warns that `pyannote-metrics 4.1 requires scipy>=1.15.1`. You can ignore this warning: pyannote-metrics is only used to score diarization against reference labels, which this pipeline does not do.

**`HF_TOKEN not found`.** Put `HF_TOKEN=hf_...` in `.env` in the repo folder, or export it in your shell.

**`401` or `403` errors during the download.** Accept the licence on both gated model pages while signed in to the same Hugging Face account that made the token.

**`Ollama is not running`.** Start it with `brew services start ollama` or open the Ollama app, then rerun `python3 run_summary.py output/<name>`. The transcript is already saved.

**`Model qwen3.5:9b is not downloaded`.** Run `ollama pull qwen3.5:9b`.

**`TRANSLATE=1 needs the translation environment`.** Run `bash setup.sh` again, or add `TRANSLATE=0` to skip translation.

**The wrong language was detected.** Set it explicitly, for example `ASR_LANG=ta`.

**Ignorable warnings.** These appear in the log and are harmless:

- pyannote's `std(): degrees of freedom is <= 0`
- torch's `Redirects are currently not supported in MacOs`

## Data handling

`input/`, `output/`, `work/`, `models/`, `.env` and the two Python environments are ignored by git, so recordings, transcripts, summaries and your token cannot be committed by accident. Ollama serves the summary model on your own machine, so the transcript is not sent anywhere.

## Licences

Check the terms of each model: pyannote community-1 (CC-BY-4.0), Indic-Transcribe-Flex (Indic Open Model License v1.0, plus the NVIDIA Canary base model terms), IndicTrans2 (AI4Bharat), and the Qwen model you use (see its page on ollama.com).
