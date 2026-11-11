# One-time download. Needs internet and HF_TOKEN. After this, nothing needs the network.
# Run with: ./env-asr/bin/python download_models.py
# Set TRANSLATE=0 to skip the IndicTrans2 translation model.
import os
from huggingface_hub import snapshot_download

# Read HF_TOKEN from a .env file next to this script, unless it is already exported.
env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(env_file):
    with open(env_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.removeprefix("export ").split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip("'\""))

token = os.environ.get("HF_TOKEN")
if not token:
    raise SystemExit("HF_TOKEN not found. Put HF_TOKEN=hf_... in .env or export it.")

models = {
    "pyannote/speaker-diarization-community-1": "models/pyannote-community-1",
    "bodhan-ai/Indic-Transcribe-Flex": "models/indic-transcribe-flex",
}
if os.environ.get("TRANSLATE", "1") != "0":
    models["ai4bharat/indictrans2-indic-en-1B"] = "models/indictrans2-indic-en-1B"

for repo, path in models.items():
    print(f"Downloading {repo} -> {path}")
    snapshot_download(repo, local_dir=path, token=token)

print("All models downloaded.")
