"""<MODULE PURPOSE>
Synthetic-dataset generator (§4.10): writes the 14 canonical submission files
(5 PDF / 6 DOCX / 3 MP4) plus dataset_manifest.json into a target directory.
Content constants live in integrity_engine.corpus_content (doc 06 is a verbatim
LLM constant with provenance). MP4s are authored offline via espeak-ng TTS + an
ffmpeg color-track composite; a duration-fit loop keeps spoken clips 10-30 s.
Build outputs are byte-deterministic (bitexact ffmpeg flags, fixed fpdf/docx
metadata).

<PUBLIC API> generate_dataset, write_manifest, build_pdf_from_text,
build_docx_from_text, build_mp4_from_text, build_silent_video

<DEPENDS ON> pathlib, subprocess (ffmpeg/espeak-ng/ffprobe), fpdf2,
python-docx, corpus_content, config.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from integrity_engine.config import Config
from integrity_engine.corpus_content import (CASE_PROMPT, DOC01_PAGES, DOC02_TEXT,
                                             DOC05_PAGES, DOC07_PAGES, DOC09_TEXT,
                                             DOC12_BODY, DOC13_BODY, DOC14_BODY,
                                             LLM_GENERATED_TEXT, PITCH_PASSAGE,
                                             PITCH_SPOKEN_PASSAGE,
                                             build_doc03_text, build_doc04_text)
from integrity_engine.text_processing import normalize_text, token_count

logger = logging.getLogger("integrity_engine.dataset_generator")

DATASET_VERSION = "1.1"
GENERATED_BY = "GLM (LLM), single pass 2026-11-04"
GENERATION_PROMPT_SUMMARY = (
    "~600-word executive case-competition submission analyzing a mid-sized organic "
    "grocery chain called GreenHarvest Foods (root causes, three-phase turnaround "
    "plan, expected margin impact, key risks), written in the register typical of "
    "LLM outputs; provenance of the document 06_llm_generated_summary.pdf only — "
    "the other 13 files are human-authored synthetic strings. Body text embedded "
    "verbatim from docs/03_llm_generated_sample.md as constant LLM_GENERATED_TEXT.")

A6_PROMPT_MIN_TOKENS = 200  # CASE_PROMPT must fill chunk 00 wholesale (dup-template)
MAX_SPEECH_SECONDS = 28.0  # hard 30 s assignment bound minus probe margin
_SPEAKING_WPM_ORDER = (155, 190, 230, 270)  # deterministic search, spec starts at 155


def _run(cmd: list[str], timeout: int = 240) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _pdf_safe(text: str) -> str:
    """Latin-1 projection for core-font PDFs.

    The projection matches the engine's own normalize_text mapping (— / – -> -),
    so extracted+normalized PDF text still equals the normalized source constant;
    the verbatim authoritative text lives in corpus_content/provenance docs.
    """
    replacements = {"\u2014": "-", "\u2013": "-", "\u2015": "-",
                    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
                    "\u2022": "-", "\u00a0": " ", "\ufb01": "fi", "\ufb00": "ff",
                    "\ufb02": "fl"}
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    # final validation: everything else must be latin-1 encodable
    try:
        text.encode("latin-1")
    except UnicodeEncodeError as exc:
        bad = exc.object[exc.start:exc.end]
        raise ValueError(f"non-latin-1 character {bad!r} in dataset text") from exc
    return text


def build_pdf_from_text(text: str, path: Path) -> None:
    """Deterministic multi-page PDF via fpdf2 (page marker: \f\f between pages)."""
    from fpdf import FPDF

    text = _pdf_safe(text)
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12.0)
    pdf.set_creation_date(datetime(2026, 11, 4, 9, 0, 0, tzinfo=timezone.utc))
    for page_text in text.split("\f\f"):
        pdf.add_page()
        pdf.set_font("Helvetica", size=11)
        pdf.multi_cell(0, 5.4, page_text)
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(path))


def build_docx_from_text(text: str, path: Path) -> None:
    """Deterministic DOCX (python-docx; template core.xml dates are static)."""
    import docx

    document = docx.Document()
    for paragraph in text.split("\n"):
        if paragraph.strip():
            document.add_paragraph(paragraph)
    path.parent.mkdir(parents=True, exist_ok=True)
    document.save(str(path))
    _normalize_zip_timestamps(path)


_FIXED_ZIP_TIME = (2026, 11, 4, 9, 0, 0)


def _normalize_zip_timestamps(path: Path) -> None:
    """Rewrite a .docx zip with fixed entry timestamps -> byte-deterministic output."""
    import zipfile

    with zipfile.ZipFile(path) as src:
        entries = [(info.filename, src.read(info.filename)) for info in src.infolist()]
    tmp = path.with_suffix(".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as dst:
        for name, data in entries:
            info = zipfile.ZipInfo(name, date_time=_FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            dst.writestr(info, data)
    tmp.replace(path)


def _espeak_to_wav(text: str, wav_path: Path, voice: str, rate: int) -> bool:
    proc = _run(["espeak-ng", "-v", voice, "-s", str(rate), "-w", str(wav_path), text])
    if proc.returncode == 0 and wav_path.is_file():
        return True
    fallback_voice = voice.lower() if voice != "en" else "en"
    if fallback_voice != voice:
        proc = _run(["espeak-ng", "-v", fallback_voice, "-s", str(rate),
                     "-w", str(wav_path), text])
        return proc.returncode == 0 and wav_path.is_file()
    return False


def _ffprobe_duration(path: Path) -> float | None:
    ffprobe = shutil.which("ffprobe") or "/usr/bin/ffprobe"
    try:
        proc = _run([ffprobe, "-v", "error", "-show_entries", "format=duration",
                     "-of", "json", str(path)])
    except (subprocess.TimeoutExpired, OSError):
        return None
    try:
        return float(json.loads(proc.stdout)["format"]["duration"])
    except (KeyError, ValueError, json.JSONDecodeError):
        return None


def _flite_to_wav(text: str, wav_path: Path, voice: str, stretch: float) -> bool:
    """flite TTS (preferred: clearer than espeak-ng for whisper-tiny, measured)."""
    flite = shutil.which("flite")
    if flite is None:
        return False
    proc = _run([flite, "-voice", voice, "--setf", f"duration_stretch={stretch}",
                 "-t", text, "-o", str(wav_path)])
    return proc.returncode == 0 and wav_path.is_file()


def build_mp4_from_text(text: str, path: Path, espeak_voice: str = "en+f2",
                        espeak_rate: int = 155, flite_voice: str | None = None,
                        flite_stretch: float = 0.85) -> None:
    """TTS narration over a solid-color video track; audio must stay 10-30 s.

    Preferred engine: flite (natural pace, whisper-tiny WER 0.07-0.14 on our clips).
    Fallback: espeak-ng with a deterministic rate search (155->190->230->270 wpm)
    until the audio is <= 28 s. Sped-up espeak audio is a documented failure mode
    (WER 0.66 at 270 wpm, EVALUATION.md failure study 1).
    """
    ffmpeg = shutil.which("ffmpeg") or "/usr/bin/ffmpeg"
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        chosen_rate: int | None = None
        chosen_wav: Path | None = None
        engine = "espeak-ng"
        flite_wav = tmp_dir / "voice_flite.wav"
        if flite_voice and _flite_to_wav(text, flite_wav, flite_voice, flite_stretch):
            duration = _ffprobe_duration(flite_wav)
            if duration is not None and duration <= MAX_SPEECH_SECONDS:
                chosen_wav, engine = flite_wav, f"flite:{flite_voice}"
                logger.info("dataset: %s spoken by flite %s (%.1f s)", path.name,
                            flite_voice, duration)
        candidates = [espeak_rate] + [r for r in _SPEAKING_WPM_ORDER
                                      if r != espeak_rate]
        for rate in ([] if chosen_wav else candidates):
            rate = int(rate)
            wav_path = tmp_dir / f"voice_{rate}.wav"
            if not _espeak_to_wav(text, wav_path, espeak_voice, rate):
                continue
            chosen_rate, chosen_wav = rate, wav_path
            duration = _ffprobe_duration(wav_path)
            if duration is None:
                logger.warning("dataset: ffprobe could not read %s; using rate %d",
                               wav_path.name, rate)
                break
            if duration <= MAX_SPEECH_SECONDS:
                logger.info("dataset: %s spoken at %d wpm (%.1f s)", path.name,
                            rate, duration)
                break
        if chosen_wav is None:
            raise RuntimeError(f"TTS failed for {path.name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("dataset: compose %s (engine=%s, rate=%s)", path.name, engine,
                    chosen_rate or "natural")
        audio_s = _ffprobe_duration(chosen_wav) or MAX_SPEECH_SECONDS
        proc = _run([  # explicit -t: "-shortest" overshoots by ~4 s with lavfi video
            ffmpeg, "-y", "-threads", "1",
            "-f", "lavfi", "-i", "color=c=0x1F3B5C:s=640x360:r=15",
            "-i", str(chosen_wav), "-t", f"{audio_s:.3f}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-ar", "16000", "-ac", "1",
            "-fflags", "+bitexact", "-flags:v", "+bitexact", "-flags:a", "+bitexact",
            "-map_metadata", "-1", str(path),
        ])
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg composite failed for {path.name}: "
                           f"{(proc.stderr or '')[-300:]}")


def build_silent_video(path: Path, seconds: float = 12.0) -> None:
    """Silent clip (anullsrc audio, black track): the Scenario-3 degraded MP4."""
    ffmpeg = shutil.which("ffmpeg") or "/usr/bin/ffmpeg"
    path.parent.mkdir(parents=True, exist_ok=True)
    proc = _run([
        ffmpeg, "-y", "-threads", "1",
        "-f", "lavfi", "-i", "color=c=black:s=640x360:r=10",
        "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", str(seconds),
        "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
        "-ar", "16000", "-ac", "1", str(path),
    ])
    if proc.returncode != 0:
        raise RuntimeError(f"silent-video build failed: {(proc.stderr or '')[-300:]}")


def write_manifest(target_dir: Path) -> None:
    """dataset_manifest.json ground truth (schema §4.10)."""
    manifest = {
        "dataset_version": DATASET_VERSION,
        "generated_with_seed": 42,
        "generated_by": GENERATED_BY,
        "generation_prompt_summary": GENERATION_PROMPT_SUMMARY,
        "ground_truth_pairs": [
            {"query": "01_retail_turnaround_report.pdf",
             "source": "02_retail_turnaround_copy.docx",
             "relation": "exact_copy", "expected_min_peer_similarity": 0.90},
            {"query": "04_original_strategy.docx",
             "source": "03_paraphrased_strategy.docx",
             "relation": "paraphrase", "expected_min_peer_similarity": 0.60,
             "expected_test": ("matched-passage cosine > 0.80 (assignment's "
                               "paraphrase band) while token Jaccard of the "
                               "pair < 0.35")},
            {"query": "07_pitch_crossmodal.pdf",
             "source": "08_pitch_crossmodal_video.mp4",
             "relation": "cross_modal", "expected_min_peer_similarity": 0.70},
        ],
        "expected_integrity_risk": {
            "01_retail_turnaround_report.pdf": "HIGH",
            "02_retail_turnaround_copy.docx": "HIGH",
            "03_paraphrased_strategy.docx": "HIGH",
            "04_original_strategy.docx": "HIGH",
            "05_original_market_pdf.pdf": "LOW",
            "06_llm_generated_summary.pdf": "MEDIUM",
            "07_pitch_crossmodal.pdf": "HIGH",
            "08_pitch_crossmodal_video.mp4": "HIGH",
            "09_third_pitch_video.mp4": "LOW",
            "10_corrupted_empty.pdf": "LOW",
            "11_silent_video.mp4": "LOW",
            "12_case_template_A.docx": "LOW",
            "13_case_template_B.docx": "LOW",
            "14_case_template_C.docx": "LOW",
        },
        "ground_truth_flags": {
            "03_paraphrased_strategy.docx": {
                "external_expected_min": 20.0,
                "note": ("second-generation KB paraphrase: 03 rewords 04, whose 2nd "
                         "paragraph rewords reference_kb KB_SUPPLY_CHAIN")},
            "04_original_strategy.docx": {"external_expected_min": 20.0},
            "06_llm_generated_summary.pdf": {"ai_expected_min_probability": 55.0},
            "05_original_market_pdf.pdf": {"expected_peer_matches": 0},
        },
        "original_false_positive_guard": [
            "05_original_market_pdf.pdf", "09_third_pitch_video.mp4",
            "12_case_template_A.docx", "13_case_template_B.docx",
            "14_case_template_C.docx",
        ],
        "degraded": [
            {"file": "10_corrupted_empty.pdf", "kind": "zero_byte_pdf",
             "expected_status": "error"},
            {"file": "11_silent_video.mp4", "kind": "silent_video",
             "expected_status": "silent_video"},
        ],
        "template_group": {
            "files": ["12_case_template_A.docx", "13_case_template_B.docx",
                      "14_case_template_C.docx"],
            "shared_block": "CASE_PROMPT",
            "expected": ("risk LOW for all three; the trio is all-DOCX (modality "
                         "irrelevant to the >=3-shared-block rule); prompt chunks "
                         "reported as match_type=common_source; no HIGH flag "
                         "(Scenario 4)"),
        },
        "content_authorship_note": (
            "All content is synthetic and authored for this project (doc 06: "
            "generated by GLM/LLM, single pass, disclosed above); no copyrighted "
            "student or vendor data used (assignment §7 compliance)."),
    }
    target = target_dir / "dataset_manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                      encoding="utf-8")
    logger.info("manifest written: %s", target)


def _verify_spoken_mp4s(target_dir: Path) -> dict[str, float]:
    """Gate assertion: the 2 spoken MP4s carry real mono 16 kHz audio, 8-35 s."""
    durations: dict[str, float] = {}
    for name in ("08_pitch_crossmodal_video.mp4", "09_third_pitch_video.mp4"):
        path = target_dir / name
        ffprobe = shutil.which("ffprobe") or "/usr/bin/ffprobe"
        proc = _run([ffprobe, "-v", "error", "-select_streams", "a",
                     "-show_entries", "stream=sample_rate,channels:format=duration",
                     "-of", "json", str(path)])
        payload = json.loads(proc.stdout)
        streams = payload.get("streams") or []
        if not streams:
            raise RuntimeError(f"{name}: no audio stream produced")
        stream = streams[0]
        duration = float(payload["format"]["duration"])
        if not 8.0 <= duration <= 35.0:
            raise RuntimeError(f"{name}: audio duration {duration}s outside 8-35 s")
        if int(stream.get("channels", 0)) != 1:
            raise RuntimeError(f"{name}: expected mono audio")
        if stream.get("sample_rate") != "16000":
            logger.warning("dataset: %s sample_rate=%s (whisper resamples anyway)",
                           name, stream.get("sample_rate"))
        durations[name] = duration
    silent = _ffprobe_duration(target_dir / "11_silent_video.mp4")
    if silent is None or silent < 8.0:
        raise RuntimeError("11_silent_video.mp4 missing/short")
    return durations


def generate_dataset(target_dir: Path, seed: int = 42,
                     cfg: Config | None = None) -> list[Path]:
    """Write all 14 canonical files + manifest; returns paths. Deterministic.

    Overwrites only canonical filenames; any other files in the dir are untouched.
    `seed` is accepted for future stochastic variants; the current corpus is fully
    deterministic (further note FN2). Spec §4.10's list-of-Path return contract is
    honored.
    """
    _ = seed  # accepted for interface stability (FN2)
    prompt_tokens = token_count(normalize_text(CASE_PROMPT))
    if prompt_tokens < A6_PROMPT_MIN_TOKENS:
        raise RuntimeError(f"CASE_PROMPT has {prompt_tokens} tokens; needs >= "
                           f"{A6_PROMPT_MIN_TOKENS} so chunk 00 is the pure prompt")
    align_cfg = (cfg or Config()).alignment
    voice_a = align_cfg.mp4_template_voice_a
    voice_b = align_cfg.mp4_template_voice_b

    target_dir.mkdir(parents=True, exist_ok=True)
    names: list[str] = []

    pdf_jobs = {
        "01_retail_turnaround_report.pdf": "\f\f".join(DOC01_PAGES),
        "05_original_market_pdf.pdf": "\f\f".join(DOC05_PAGES),
        "06_llm_generated_summary.pdf": (
            "Executive Analysis: GreenHarvest Foods (LLM-generated summary)\f\f"
            + LLM_GENERATED_TEXT),
        "07_pitch_crossmodal.pdf": "\f\f".join(DOC07_PAGES),
    }
    for name, text in pdf_jobs.items():
        build_pdf_from_text(text, target_dir / name)
    names.extend(pdf_jobs)

    docx_jobs = {
        "02_retail_turnaround_copy.docx": DOC02_TEXT,
        "03_paraphrased_strategy.docx": build_doc03_text(),
        "04_original_strategy.docx": build_doc04_text(),
        "12_case_template_A.docx": f"{CASE_PROMPT}\n\n{DOC12_BODY}",
        "13_case_template_B.docx": f"{CASE_PROMPT}\n\n{DOC13_BODY}",
        "14_case_template_C.docx": f"{CASE_PROMPT}\n\n{DOC14_BODY}",
    }
    for name, text in docx_jobs.items():
        build_docx_from_text(text, target_dir / name)
    names.extend(docx_jobs)

    build_mp4_from_text(PITCH_SPOKEN_PASSAGE,
                        target_dir / "08_pitch_crossmodal_video.mp4",
                        espeak_voice=voice_a, flite_voice="slt")
    names.append("08_pitch_crossmodal_video.mp4")
    build_mp4_from_text(DOC09_TEXT, target_dir / "09_third_pitch_video.mp4",
                        espeak_voice=voice_b, flite_voice="rms")
    names.append("09_third_pitch_video.mp4")
    build_silent_video(target_dir / "11_silent_video.mp4", seconds=12.0)
    names.append("11_silent_video.mp4")
    (target_dir / "10_corrupted_empty.pdf").touch()  # zero-byte, E1
    names.append("10_corrupted_empty.pdf")

    durations = _verify_spoken_mp4s(target_dir)
    logger.info("spoken clips verified: %s",
                {k: round(v, 1) for k, v in durations.items()})
    write_manifest(target_dir)
    names.append("dataset_manifest.json")
    logger.info("dataset generated in %s", target_dir)
    return [target_dir / name for name in sorted(names)]
