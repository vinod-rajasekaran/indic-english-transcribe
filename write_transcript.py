# Writes the readable transcript (.txt) and its JSON twin from segments.json.
# Uses only the standard library, so the default pipeline needs no translation environment.
# Run with: python3 write_transcript.py segments.json transcript.txt
# run_translate.py imports write() and adds an English line to each turn.
import json
import sys


def stamp(sec):
    sec = int(sec)
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def write(segments, out):
    with open(out, "w", encoding="utf-8") as f:
        for s in segments:
            where = f"  ({s['source_file']} at {stamp(s['source_start'])})" if "source_file" in s else ""
            f.write(f"[{stamp(s['start'])}] {s['speaker']}{where}\n")
            f.write(f"  Original: {s['text']}\n")
            if "roman" in s:
                f.write(f"  Roman:    {s['roman']}\n")
            if "english" in s:
                f.write(f"  English:  {s['english']}\n")
            f.write("\n")
    json_out = out[:-4] + ".json" if out.endswith(".txt") else out + ".json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    print(f"Saved {out} and {json_out}")


if __name__ == "__main__":
    src = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "transcript.txt"
    with open(src, encoding="utf-8") as f:
        write(json.load(f), out)
