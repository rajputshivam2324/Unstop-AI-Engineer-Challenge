#!/usr/bin/env bash
# One-shot environment setup (CPU only).
#   bash scripts/setup_env.sh            # uses python3.11
#   PYTHON_BIN=python3.12 bash scripts/setup_env.sh
# System packages needed: ffmpeg (audio extraction). For regenerating the dataset
# only: flite and espeak-ng (TTS).
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python3.11}"

if [ ! -x .venv/bin/python ]; then
  "$PYTHON_BIN" -m venv .venv
fi
.venv/bin/python -m pip install --upgrade pip
# CPU wheel first so no CUDA runtime is pulled in
.venv/bin/pip install "torch==2.15.0" --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/check_env.py
# pre-download the four models (~2.5 GB) into ./cache/hf
.venv/bin/python run_integrity_check.py --input_dir ./sample_submissions \
  --output_dir ./reports --models-only
command -v ffmpeg >/dev/null || echo "WARNING: ffmpeg not found; MP4 files will be reported as errors"
