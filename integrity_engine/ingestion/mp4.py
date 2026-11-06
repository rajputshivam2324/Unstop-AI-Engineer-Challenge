"""<MODULE PURPOSE>
MP4 extractor: ffprobe audio-stream probe (absent -> silent_video), ffmpeg WAV
extraction (16 kHz mono PCM), RMS loudness guard (mean_volume < -45 dB -> fast
silent path, ASR skipped), faster-whisper tiny/int8 transcription with VAD, and
per-segment provenance (one Page per ASR segment, A5). Never raises on file
content; everything degrades to error/silent_video + warnings.

<PUBLIC API> extract(path, config) -> ProcessedDocument, release_asr()
<DEPENDS ON> subprocess/ffmpeg/ffprobe/faster-whisper, ingestion.base, types, config.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from integrity_engine.config import Config
from integrity_engine.ingestion.base import EMPTY_TEXT_THRESHOLD_CH, finish_document
from integrity_engine.text_processing import normalize_text
from integrity_engine.types import Page, ProcessedDocument

logger = logging.getLogger("integrity_engine.ingestion.mp4")

WHISPER_MODEL_SIZE = "tiny"  # spec §4.1: tiny/int8 CPU; base stays an upgrade path
SILENCE_DB_FLOOR = -45.0  # mean_volume below this => effectively silent (E4)
_TRANSCRIPT_JOIN = " "  # A5: segments joined with a single space

_ASR_STATE: dict[str, object] = {"model": None}


def _model_cache_dir(cfg: Config) -> Path:
    base = cfg.runtime.hf_home or cfg.runtime.cache_dir
    return Path(base) / "hf"


def _get_asr(cfg: Config):
    """Lazy process-wide ASR singleton (loaded once per run, released by the CLI)."""
    if _ASR_STATE["model"] is None:
        from faster_whisper import WhisperModel

        download_root = str(_model_cache_dir(cfg))
        _ASR_STATE["model"] = WhisperModel(
            WHISPER_MODEL_SIZE, device="cpu", compute_type="int8",
            download_root=download_root)
        logger.debug("asr: loaded faster-whisper %s int8 (root=%s)",
                     WHISPER_MODEL_SIZE, download_root)
    return _ASR_STATE["model"]


def release_asr() -> None:
    """Drop the process-wide ASR model BEFORE AI models load (RAM discipline §6)."""
    if _ASR_STATE["model"] is not None:
        _ASR_STATE["model"] = None
        gc.collect()
        logger.debug("asr: model released")


def _run(cmd: list[str], timeout_s: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s,
                          check=False)


def _ffprobe_audio(path: Path, timeout_s: int) -> dict | None:
    """First audio-stream metadata dict; {} when there is none; None on probe failure."""
    ffmpeg = shutil.which("ffprobe") or "/usr/bin/ffprobe"
    try:
        proc = _run([
            ffmpeg, "-v", "error", "-select_streams", "a",
            "-show_entries", "stream=codec_name,sample_rate,channels,duration",
            "-of", "json", str(path),
        ], timeout_s)
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.debug("ffprobe failed: %s (%s)", path.name, exc)
        return None
    if proc.returncode != 0:
        return None
    try:
        payload = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        return None
    streams = payload.get("streams") or []
    return streams[0] if streams else {}


def _ffmpeg_to_wav(path: Path, timeout_s: int) -> Path | None:
    """Extract 16 kHz mono PCM WAV; None on failure/timeout."""
    ffmpeg = shutil.which("ffmpeg") or "/usr/bin/ffmpeg"
    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    Path(wav_path).unlink(missing_ok=True)  # clean start (destination, not input)
    try:
        proc = _run([
            ffmpeg, "-y", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000",
            "-c:a", "pcm_s16le", wav_path,
        ], timeout_s)
        if proc.returncode != 0 or not Path(wav_path).is_file():
            logger.debug("ffmpeg wav extraction failed rc=%d %s", proc.returncode,
                         (proc.stderr or "")[-200:])
            Path(wav_path).unlink(missing_ok=True)
            return None
        return Path(wav_path)
    except subprocess.TimeoutExpired:
        Path(wav_path).unlink(missing_ok=True)
        raise
    except OSError:
        Path(wav_path).unlink(missing_ok=True)
        return None


def _mean_volume_db(path: Path, timeout_s: int) -> float | None:
    """ffmpeg volumedetect mean_volume in dB; None when not computable."""
    ffmpeg = shutil.which("ffmpeg") or "/usr/bin/ffmpeg"
    try:
        proc = _run([ffmpeg, "-i", str(path), "-af", "volumedetect", "-f", "null", "-"],
                    timeout_s)
    except (subprocess.TimeoutExpired, OSError):
        return None
    for line in (proc.stderr or "").splitlines():
        if "mean_volume:" in line:
            try:
                return float(line.split("mean_volume:")[1].strip().split(" ")[0])
            except (ValueError, IndexError):
                return None
    return None


def extract(path: Path, cfg: Config) -> ProcessedDocument:
    """Extract an MP4 to a ProcessedDocument with per-segment timestamps; never raises."""
    budget = cfg.runtime.max_ingest_seconds_per_file
    started = time.monotonic()

    try:
        size = path.stat().st_size
    except OSError as exc:
        logger.warning("mp4: unreadable media container: %s (%s: %s)", path.name,
                       type(exc).__name__, exc)
        return finish_document(path, [], [f"unreadable media container ({exc})"],
                               "mp4", cfg, {"_ingestion_status": "error"})
    if size == 0:  # E1 for MP4 too
        logger.warning("file is empty (0 bytes); nothing to analyze: %s", path.name)
        return finish_document(
            path, [], ["file is empty (0 bytes); nothing to analyze"],
            "mp4", cfg, {"_ingestion_status": "error"})

    audio = _ffprobe_audio(path, budget)
    if audio is None:
        logger.warning("mp4: unreadable media container: %s", path.name)
        return finish_document(path, [], ["unreadable media container"],
                               "mp4", cfg, {"_ingestion_status": "error"})
    if not audio:  # E2: no audio stream at all
        logger.warning("mp4: no audio stream in MP4: %s", path.name)
        return finish_document(path, [], ["no audio stream in MP4"],
                               "mp4", cfg, {"_ingestion_status": "silent_video",
                                            "duration_s": None})

    duration_s: float | None
    try:
        duration_s = float(audio.get("duration")) if audio.get("duration") else None
    except (TypeError, ValueError):
        duration_s = None

    if duration_s is not None and duration_s > 60:
        logger.info("mp4: long audio (%.0fs); processing within prototype limits",
                    duration_s)

    # E3/E4 fast path: silent audio detected before ASR (saves wall time)
    mean_db = _mean_volume_db(path, budget)
    if mean_db is not None and mean_db < SILENCE_DB_FLOOR:
        logger.warning("mp4: audio track is effectively silent (mean volume < -45 dB): %s",
                       path.name)
        return finish_document(
            path, [], ["audio track is effectively silent (mean volume < -45 dB)"],
            "mp4", cfg, {"_ingestion_status": "silent_video",
                         "duration_s": duration_s})

    wav_path = _ffmpeg_to_wav(path, budget)
    if wav_path is None:
        logger.warning("mp4: unreadable media container: %s", path.name)
        return finish_document(path, [], ["unreadable media container"],
                               "mp4", cfg, {"_ingestion_status": "error",
                                            "duration_s": duration_s})
    try:
        if time.monotonic() - started > budget:  # E12 time guard
            logger.warning("mp4: asr exceeded per-file time budget: %s", path.name)
            return finish_document(
                path, [], ["asr exceeded per-file time budget"],
                "mp4", cfg, {"_ingestion_status": "error", "duration_s": duration_s})

        warnings: list[str] = []
        try:
            segments_iter, info = _get_asr(cfg).transcribe(
                str(wav_path), language=None, vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500})
        except Exception as exc:  # noqa: BLE001 - ASR failures degrade to error
            summary = f"{type(exc).__name__}: {exc}"[:140]
            logger.warning("mp4: transcription failed: %s (%s)", path.name, summary)
            return finish_document(
                path, [], [f"transcription failed ({summary})"],
                "mp4", cfg, {"_ingestion_status": "error", "duration_s": duration_s})

        language = getattr(info, "language", None) or "unknown"
        if language != "en":  # E5/A4: process anyway, warn
            warning_text = (f"non-english audio detected ({language}); "
                            "transcription may be degraded")
            logger.warning("mp4: non-english audio detected (%s): %s", language,
                           path.name)
            warnings.append(warning_text)

        pages: list[Page] = []
        seg_times: list[tuple[float, float]] = []
        for segment in segments_iter:
            if time.monotonic() - started > budget:  # E12 guard mid-iteration
                logger.warning("mp4: asr exceeded per-file time budget: %s", path.name)
                return finish_document(
                    path, [], ["asr exceeded per-file time budget"],
                    "mp4", cfg, {"_ingestion_status": "error",
                                 "duration_s": duration_s})
            text = normalize_text(segment.text or "").strip()
            if not text:
                continue
            seg_times.append((float(segment.start), float(segment.end)))
            # A5: segment char offsets inside the single-space-joined transcript,
            # computed cumulatively as segments are appended on one space.
            offset0 = pages[-1].char_end + len(_TRANSCRIPT_JOIN) if pages else 0
            pages.append(Page(page=None, text=text, char_start=offset0,
                              char_end=offset0 + len(text)))

        transcript = _TRANSCRIPT_JOIN.join(p.text for p in pages)
        if not seg_times:  # E3: VAD found no speech
            logger.warning("mp4: speech-not-detected by VAD: %s", path.name)
            return finish_document(
                path, [], ["speech-not-detected by VAD; possibly silent or music-only audio"],
                "mp4", cfg, {"_ingestion_status": "silent_video",
                             "duration_s": duration_s})
        if len(transcript.strip()) < EMPTY_TEXT_THRESHOLD_CH:
            logger.info("mp4: transcript too short to analyze: %s", path.name)
            return finish_document(
                path, [], ["transcript too short to analyze"],
                "mp4", cfg, {"_ingestion_status": "empty", "duration_s": duration_s,
                             "asr_language": language, "asr_model": WHISPER_MODEL_SIZE,
                             "segments": seg_times})

        return finish_document(
            path, pages, warnings, "mp4", cfg,
            {"duration_s": duration_s, "asr_language": language,
             "asr_model": WHISPER_MODEL_SIZE, "segments": seg_times},
            separator=_TRANSCRIPT_JOIN)
    finally:
        wav_path.unlink(missing_ok=True)
