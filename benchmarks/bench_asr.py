#!/usr/bin/env python
"""ASR word-error-rate benchmark on the corpus' spoken clips.

Re-synthesizes each spoken script with the TTS settings used during development
(espeak-ng at 155 and 270 wpm, flite at natural pace) plus the shipped MP4 audio,
then transcribes with faster-whisper tiny and base (int8, CPU, VAD on - same
settings as ingestion/mp4.py). WER is word-level Levenshtein against the script,
after lowercasing and mapping "%" -> "percent". Numbers and digits are NOT
normalized ("nineteen" vs "19" counts as an error), so WER is pessimistic.

Limitation stated up front: this is synthetic TTS speech, not human or accented
speech; it measures pipeline integration and the TTS-speed failure mode, not
real-world ASR accuracy.
Usage: .venv/bin/python benchmarks/bench_asr.py
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from integrity_engine.corpus_content import DOC09_TEXT, PITCH_SPOKEN_PASSAGE  # noqa: E402

RESULTS = ROOT / "benchmarks" / "results"


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower().replace("%", " percent"))


def wer(ref: str, hyp: str) -> float:
    r, h = words(ref), words(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return d[len(h)] / max(1, len(r))


def duration(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    return float(out.stdout.strip())


def main() -> int:
    from faster_whisper import WhisperModel

    tmp = Path(tempfile.mkdtemp())
    clips: list[tuple[str, str, str, Path]] = []
    scripts = {"08_pitch": (PITCH_SPOKEN_PASSAGE, "slt"), "09_pitch": (DOC09_TEXT, "rms")}
    for clip, (text, voice) in scripts.items():
        mp4 = ROOT / "sample_submissions" / (
            "08_pitch_crossmodal_video.mp4" if clip == "08_pitch" else "09_third_pitch_video.mp4")
        shipped = tmp / f"{clip}_shipped.wav"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(mp4), "-ac", "1", "-ar",
                        "16000", str(shipped)], check=True)
        clips.append((clip, f"shipped MP4 (flite {voice})", text, shipped))
        for rate in (155, 270):
            wav = tmp / f"{clip}_espeak{rate}.wav"
            subprocess.run(["espeak-ng", "-v", "en+f2", "-s", str(rate), "-w", str(wav), text],
                           check=True)
            clips.append((clip, f"espeak-ng {rate} wpm", text, wav))
    rows = []
    for size in ("tiny", "base"):
        model = WhisperModel(size, device="cpu", compute_type="int8",
                             download_root=str(ROOT / "cache" / "hf"))
        for clip, tts, text, wav in clips:
            t0 = time.time()
            segs, _ = model.transcribe(str(wav), language=None, vad_filter=True,
                                       vad_parameters={"min_silence_duration_ms": 500})
            hyp = " ".join(s.text.strip() for s in segs)
            asr_s = time.time() - t0
            audio = duration(wav)
            rows.append({"clip": clip, "tts": tts, "audio_s": audio, "model": size,
                         "wer": wer(text, hyp), "asr_s": asr_s, "rtf": asr_s / audio,
                         "hypothesis": hyp})
            print(f"{size:5s} {clip} {tts:28s} {audio:5.1f}s WER={rows[-1]['wer']:.3f}")
    shutil.rmtree(tmp, ignore_errors=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "asr.json").write_text(json.dumps({"rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
