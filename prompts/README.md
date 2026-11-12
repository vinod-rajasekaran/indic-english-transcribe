# Summary prompts

The summary prompt depends on what the recording is. Each file here is one purpose, and `PURPOSE` picks it. Transcription and translation don't depend on the purpose, so you transcribe once and then summarize as many times as you like:

```
# Once per recording: transcription and translation
bash run_all.sh fgd-1 part-1.m4a part-2.m4a

# Then one or more summaries of that transcript, about 5 minutes each
PURPOSE=fgd python3 run_summary.py output/fgd-1
PURPOSE=meeting python3 run_summary.py output/fgd-1
PURPOSE=path/to/my-prompt.md python3 run_summary.py output/fgd-1
```

`PURPOSE=fgd bash run_all.sh fgd-1 ...` does the transcript and the first summary in one go.

| File | Purpose | Status |
|---|---|---|
| `fgd.md` | Focus group discussion: themes, agreement and disagreement, action items | Tested on an hour-long Tamil and English group discussion |
| `meeting.md` | Work meeting: topics, decisions, action items with owners and deadlines | Run only on a focus group recording (19 of 21 citations exact); not yet tested on a real meeting |

## Writing your own

Copy `fgd.md` or `meeting.md` to a new file in this folder, for example `interview.md`, then edit it and run with `PURPOSE=interview`. A file has six sections, each starting with its name in square brackets on a line of its own:

| Section | What it does |
|---|---|
| `[title]` | Heading of the summary file, such as `Meeting notes` |
| `[role]` | Who the model should act as, such as "an experienced note-taker writing up the minutes of a work meeting" |
| `[rules]` | Extra rules for this purpose, as bullets. Added to the standard rules below |
| `[speakers]` | How to suggest roles in `speakers.json`: who leads, and what to call everyone else |
| `[notes]` | What to note on each part of a long transcript, as a short list. Used when the transcript is too long for one pass |
| `[format]` | The sections the summary should have, with what goes in each |

All six are required. Write them as plain text, with no `{` placeholders.

## What stays the same for every purpose

`run_summary.py` adds these to every prompt, so a new purpose can't weaken them:

- Every point cites `[hh:mm:ss] Speaker` from the transcript, and the script checks every cited timestamp afterwards.
- No facts, names, numbers or recommendations that nobody said. Lines too garbled to interpret are left out.
- Quotes are word for word, in the original script with an English translation, depending on `SUMMARY_SOURCE`.
- No comments on transcription or translation quality.
- A participation table computed from the transcript, not written by the model.

Each purpose writes its own file, `summary_<purpose>.md`, so you can run several purposes on one transcript. `speakers.json` is shared, so if the role names suit one purpose and not another (Moderator against Chair), edit the file between runs.
