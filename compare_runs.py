# Compares two runs of the same recording word for word, for example 1x against 1.5x speed.
# Uses only the standard library.
# Run with: python3 compare_runs.py output/test-1x output/test-15x
#
# The runs can split speaker turns differently, so the whole transcript is compared as one
# stream of words. The first run is treated as the reference.
import difflib
import json
import re
import sys

from write_transcript import stamp

A, B = sys.argv[1].rstrip("/"), sys.argv[2].rstrip("/")


def load(d):
    with open(f"{d}/transcript.json", encoding="utf-8") as f:
        return json.load(f)


def words(segs, field):
    # Split on spaces, not \w: Tamil and other Indic vowel signs are not \w characters.
    out = []
    for s in segs:
        for w in (s.get(field) or "").lower().split():
            w = re.sub(r"^[^\w\u0900-\u0DFF]+|[^\w\u0900-\u0DFF]+$", "", w)
            if w:
                out.append(w)
    return out




def wer(ref, hyp):
    """Word error rate of hyp against ref: (substitutions + deletions + insertions) / len(ref)."""
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1] / max(1, len(ref))


a, b = load(A), load(B)
print(f"Reference: {A}   Compared: {B}\n")
print(f"{'':28}{'reference':>12}{'compared':>12}")
print(f"{'speaker turns':28}{len(a):>12}{len(b):>12}")
print(f"{'distinct speakers':28}{len({s['speaker'] for s in a}):>12}{len({s['speaker'] for s in b}):>12}")

for field, label in (("text", "original language"), ("english", "English")):
    wa, wb = words(a, field), words(b, field)
    if not wa and not wb:
        continue
    sm = difflib.SequenceMatcher(None, wa, wb, autojunk=False)
    same = sum(blk.size for blk in sm.get_matching_blocks())
    print(f"\n== {label} ==")
    print(f"{'words':28}{len(wa):>12}{len(wb):>12}")
    print(f"word error rate vs reference: {wer(wa, wb):.1%}")
    print(f"words identical and in order: {same}/{len(wa)} ({same / max(1, len(wa)):.1%})")

    if field == "english":
        print("\nDifferences (reference -> compared), first 40:")
        shown = 0
        for op, i1, i2, j1, j2 in sm.get_opcodes():
            if op == "equal":
                continue
            ctx = " ".join(wa[max(0, i1 - 4):i1])
            print(f"  ...{ctx} [{' '.join(wa[i1:i2]) or '-'}] -> [{' '.join(wb[j1:j2]) or '-'}]")
            shown += 1
            if shown == 40:
                break

# Side by side by time, so you can read the two versions of each moment together.
side = "english" if any(s.get("english") for s in a) else "text"
print(f"\n== Side by side ({side}) ==")
for s in a:
    overlap = [t for t in b if t["start"] < s["end"] and t["end"] > s["start"]]
    print(f"[{stamp(s['start'])}] {s['speaker']}")
    print(f"  ref: {s.get(side, '')}")
    print(f"  cmp: {' | '.join(t.get(side, '') for t in overlap)}")
