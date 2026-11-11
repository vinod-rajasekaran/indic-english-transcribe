# Adds source_file and source_start to each segment, so you can find the moment in the
# original recording. Segment start and end stay on the joined timeline.
# Uses only the standard library. Run with: python3 add_sources.py parts.tsv segments.json
import json
import sys

parts_path, segments_path = sys.argv[1], sys.argv[2]

parts, offset = [], 0.0
with open(parts_path, encoding="utf-8") as f:
    for line in f:
        name, dur = line.rstrip("\n").split("\t")
        parts.append((name, offset, offset + float(dur)))
        offset += float(dur)

with open(segments_path, encoding="utf-8") as f:
    segments = json.load(f)

for seg in segments:
    # A segment belongs to the part where it begins.
    for name, begin, end in parts:
        if begin <= seg["start"] < end or (name, begin, end) == parts[-1]:
            seg["source_file"] = name
            seg["source_start"] = round(max(0.0, seg["start"] - begin), 2)
            break

with open(segments_path, "w", encoding="utf-8") as f:
    json.dump(segments, f, ensure_ascii=False, indent=2)

print(f"Mapped {len(segments)} segments across {len(parts)} file(s)")
