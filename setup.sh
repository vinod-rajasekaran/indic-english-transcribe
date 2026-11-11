#!/usr/bin/env bash
# One-time setup. Needs internet. Run from the repo folder.
# Add TRANSLATE=0 to skip the English translation environment.
set -euo pipefail
cd "$(dirname "$0")"

MAC=0; [ "$(uname)" = "Darwin" ] && MAC=1

if [ "$MAC" = "1" ]; then
  # macOS (Apple silicon): needs Homebrew. Diarization and translation use the GPU, transcription the CPU.
  command -v brew >/dev/null || { echo "Install Homebrew first: https://brew.sh"; exit 1; }
  command -v ffmpeg >/dev/null || brew install ffmpeg
  command -v python3.10 >/dev/null || brew install python@3.10
  # Ollama serves the summary model. Skip with SUMMARIZE=0.
  if [ "${SUMMARIZE:-1}" != "0" ]; then
    command -v ollama >/dev/null || brew install ollama
  fi
else
  sudo apt-get update
  # Python 3.10 is not in the default repos of Ubuntu 24.04 and later; the deadsnakes PPA provides it.
  if ! apt-cache show python3.10 >/dev/null 2>&1; then
    sudo apt-get install -y software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update
  fi
  sudo apt-get install -y ffmpeg git python3.10 python3.10-venv
fi

# Environment 1: diarization and transcription
python3.10 -m venv env-asr
./env-asr/bin/pip install --upgrade pip
./env-asr/bin/pip install torch torchaudio
./env-asr/bin/pip install pyannote.audio transformers sentencepiece soundfile huggingface_hub
# SciPy 1.15.x (the newest for Python 3.10) fails to load on macOS 27 with
#   "section '__DATA/__thread_bss' has a zero-fill section type, but offset field is not zero".
# SciPy 1.14.1 loads fine. pip then warns that pyannote-metrics wants scipy>=1.15.1;
# that warning is harmless, because diarization does not use pyannote-metrics.
if ! ./env-asr/bin/python -c "import scipy.sparse.linalg" 2>/dev/null; then
  echo "SciPy does not load on this system; installing SciPy 1.14.1 instead"
  ./env-asr/bin/pip install "scipy==1.14.1"
fi
./env-asr/bin/python -c "import scipy.sparse.linalg, torch, pyannote.audio, soundfile" \
  || { echo "Environment env-asr does not import cleanly; see Troubleshooting in README.md"; exit 1; }

# Environment 2: translation (skipped with TRANSLATE=0). Kept separate so its transformers pin cannot clash with environment 1.
if [ "${TRANSLATE:-1}" != "0" ]; then
  python3.10 -m venv env-translate
  ./env-translate/bin/pip install --upgrade pip
  ./env-translate/bin/pip install torch sentencepiece sacremoses indictranstoolkit "transformers==4.46.3"
  ./env-translate/bin/python -c "import IndicTransToolkit, transformers" \
    || { echo "Environment env-translate does not import cleanly; see Troubleshooting in README.md"; exit 1; }
fi

mkdir -p models work
echo "Setup done. Next:"
echo "  1. Put your Hugging Face token in .env:   echo 'HF_TOKEN=hf_...' > .env"
echo "  2. Download the models:                    ./env-asr/bin/python download_models.py"
if [ "${SUMMARIZE:-1}" != "0" ]; then
  echo "  3. Start Ollama and get the summary model: brew services start ollama && ollama pull qwen3.5:9b"
  echo "     (on Linux, install Ollama from ollama.com/download first)"
fi
