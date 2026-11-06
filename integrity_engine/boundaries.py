"""Alternative matching units with exact offsets into the ORIGINAL full_text.

ASR segment pages remain intact for timestamp mapping. Only matching units are
reconstructed; no punctuation, wording, or timestamp is fabricated.
"""

from __future__ import annotations

import re

from integrity_engine.text_processing import split_sentences, token_count
from integrity_engine.types import ProcessedDocument, Sentence


def reconstructed_sentences(doc: ProcessedDocument, max_gap_s: float = 1.5,
                            max_seconds: float = 45.0,
                            max_tokens: int = 120) -> list[Sentence]:
    """Split PDF headings and join unfinished adjacent ASR segments.

    PDF heading heuristic is intentionally narrow: a short unpunctuated first
    line, followed by an uppercase line. It is kept as its own unit, not deleted.
    Wrapped body lines otherwise stay together. Audio pauses and size limits
    prevent joining unrelated speech across long gaps.
    """
    if doc.modality == "mp4":
        return _speech_sentences(doc, max_gap_s, max_seconds, max_tokens)
    out = []
    for page in doc.pages:
        start, text = page.char_start, page.text
        first, sep, rest = text.partition("\n")
        heading = (doc.modality == "pdf" and sep and 1 <= token_count(first) <= 12
                   and not re.search(r"[.!?;,:-]\s*$", first)
                   and rest.lstrip() and rest.lstrip()[0].isupper())
        parts = [(0, len(first)), (len(first) + 1, len(text))] if heading else [(0, len(text))]
        for a, b in parts:
            for s in split_sentences(text[a:b]):
                out.append(Sentence(s.text, start + a + s.char_start,
                                    start + a + s.char_end, page.page))
    return out


def _speech_sentences(doc, max_gap, max_seconds, max_tokens):
    times = doc.metadata.get("segments", [])
    groups: list[list[int]] = []
    for i, page in enumerate(doc.pages):
        join = False
        if groups and i < len(times):
            first, prev = groups[-1][0], groups[-1][-1]
            if first < len(times) and prev < len(times):
                gap = times[i][0] - times[prev][1]
                text = doc.full_text[doc.pages[first].char_start:page.char_end]
                join = (not re.search(r"[.!?][\"']?\s*$", doc.pages[prev].text)
                        and 0 <= gap <= max_gap
                        and times[i][1] - times[first][0] <= max_seconds
                        and token_count(text) <= max_tokens)
        if join:
            groups[-1].append(i)
        else:
            groups.append([i])
    out = []
    for group in groups:
        start = doc.pages[group[0]].char_start
        end = doc.pages[group[-1]].char_end
        # Punctuation may terminate a sentence inside an individual ASR segment.
        for s in split_sentences(doc.full_text[start:end]):
            out.append(Sentence(s.text, start + s.char_start, start + s.char_end))
    return out


def adjacent_windows(doc: ProcessedDocument, sentences: list[Sentence],
                     max_units: int, max_tokens: int = 120,
                     max_gap_s: float = 1.5, max_seconds: float = 45.0) -> list[Sentence]:
    """Contiguous 1..N-unit candidates; keep page/time boundaries and exact spans.

    These overlapping candidates are NEVER used directly as coverage denominator.
    Evidence coverage is a union of accepted character spans.
    """
    out = []
    for i, first in enumerate(sentences):
        for j in range(i, min(len(sentences), i + max_units)):
            last = sentences[j]
            if first.page != last.page:
                break
            span = doc.full_text[first.char_start:last.char_end]
            if token_count(span) > max_tokens:
                break
            if doc.modality == "mp4":
                times = doc.metadata.get("segments", [])
                covered = [times[k] for k, p in enumerate(doc.pages)
                           if k < len(times) and p.char_start < last.char_end
                           and p.char_end > first.char_start]
                if not covered:
                    break
                if (covered[-1][1] - covered[0][0] > max_seconds
                        or any(b[0] - a[1] > max_gap_s for a, b in zip(covered, covered[1:]))):
                    break
            out.append(Sentence(span, first.char_start, last.char_end, first.page))
    return out
