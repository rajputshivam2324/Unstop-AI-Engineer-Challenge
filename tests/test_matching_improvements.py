"""Mechanism regressions; no global floor changes and exact source provenance."""
import json
from dataclasses import replace
from pathlib import Path

import pytest

from integrity_engine.alignment import DocumentAligner, TemplateRegistry, evidence_pair_scores
from integrity_engine.boundaries import adjacent_windows, reconstructed_sentences
from integrity_engine.config import Config
from integrity_engine.enhanced_alignment import EnhancedDocumentAligner
from integrity_engine.semantic import NLI_REVISION, SemanticScore
from integrity_engine.types import Page, ProcessedDocument

ROOT = Path(__file__).resolve().parents[1]


def speech(parts, times):
    text = ' '.join(parts)
    pages, cursor = [], 0
    for part in parts:
        pages.append(Page(None, part, cursor, cursor + len(part)))
        cursor += len(part) + 1
    return ProcessedDocument('02_test.mp4', Path('02_test.mp4'), 'mp4', 'ok',
                             pages=pages, full_text=text, metadata={'segments': times})


def test_reconstruction_preserves_offsets_and_joins_unfinished_speech():
    doc = speech(['The committee approved a new rail connection',
                  'after the engineers completed the bridge survey.'], [(2, 5), (5.4, 9)])
    units = reconstructed_sentences(doc)
    assert len(units) == 1
    assert units[0].text == doc.full_text[units[0].char_start:units[0].char_end]
    from integrity_engine.alignment import time_span
    assert time_span(doc, units[0].char_start, units[0].char_end) == '00:00:02 - 00:00:09'
    assert len(doc.pages) == 2 and doc.metadata['segments'] == [(2, 5), (5.4, 9)]


def test_pause_and_size_guards_do_not_join_unrelated_speech():
    parts = ['The speaker stopped before finishing the sentence',
             'another speaker began discussing a different subject.']
    doc = speech(parts, [(0, 4), (12, 18)])
    units = reconstructed_sentences(doc)
    assert len(units) == 2
    assert len(adjacent_windows(doc, units, 3)) == 2
    doc.metadata['segments'] = [(0, 4), (4.2, 8)]
    assert len(reconstructed_sentences(doc, max_tokens=10)) == 2


def test_short_pdf_heading_kept_separate_and_body_wrap_preserved():
    text = ('Water system proposal\nThe council repairs damaged pipes before\n'
            'installing a larger tank.')
    doc = ProcessedDocument('01_test.pdf', Path('01_test.pdf'), 'pdf', 'ok',
                             pages=[Page(2, text, 0, len(text))], full_text=text)
    units = reconstructed_sentences(doc)
    assert [u.text for u in units] == ['Water system proposal', text.split('\n', 1)[1]]
    assert all(u.page == 2 and doc.full_text[u.char_start:u.char_end] == u.text
               for u in units)


def test_adjacent_speech_windows_tolerate_premature_asr_punctuation():
    doc = speech(['The council repaired the bridge.',
                  'Before allowing school buses across the river.'], [(1, 4), (4.3, 8)])
    units = reconstructed_sentences(doc)
    assert len(units) == 2
    windows = adjacent_windows(doc, units, 3)
    assert any(w.text == doc.full_text for w in windows)


@pytest.fixture(scope='module')
def embedder():
    from integrity_engine.embeddings import Embedder
    emb = Embedder('sentence-transformers/all-MiniLM-L6-v2', str(ROOT / 'cache/hf'))
    yield emb
    emb.release()


def load_regression(names):
    data = json.loads((ROOT / 'tests/fixtures/matching_regressions.json').read_text())
    return [ProcessedDocument(name, Path(name), Path(name).suffix[1:], 'ok',
                              pages=[Page(**p) for p in data[name]['pages']],
                              full_text=data[name]['full_text'], metadata=data[name]['metadata'])
            for name in names]


def test_confirmed_crossmodal_fix_and_baseline_available(embedder):
    a, b = load_regression(['01_human_speech.pdf', '11_human_crossmodal.mp4'])
    cfg = Config()
    old = DocumentAligner(embedder, TemplateRegistry(), cfg, [])
    assert old.coverage(a, b) == old.coverage(b, a) == 0
    cfg = replace(cfg, matching=replace(cfg.matching, mode='boundary'))
    new = EnhancedDocumentAligner(embedder, TemplateRegistry(), cfg, [])
    pairs = evidence_pair_scores([a, b], new, [], {}, 10)
    assert max(p.score for p in pairs) >= .70
    passages, _ = new.align(a, b)
    assert passages
    assert all(p.query_page == 2 for p in passages)
    assert any(p.matched_timestamp == '00:00:00 - 00:00:15' for p in passages)
    assert all(p.query_passage in a.full_text and p.matched_passage in b.full_text
               for p in passages)
    assert new.coverage(a, b) <= 1 and new.coverage(b, a) <= 1
    assert cfg.alignment.sentence_align_cos_floor == .65


def test_confirmed_paraphrase_retrieved_and_verified_with_real_nli(embedder):
    snapshot = (ROOT / 'cache/hf/models--cross-encoder--nli-deberta-v3-small'
                / 'snapshots' / NLI_REVISION)
    if not snapshot.exists():
        pytest.skip('cache pinned NLI model: --models-only --matching-mode semantic_nli')
    a, b = load_regression(['06_policy_argument.docx', '07_policy_rewording.docx'])
    cfg = Config()
    old = DocumentAligner(embedder, TemplateRegistry(), cfg, [])
    assert old.coverage(a, b) == 0
    cfg = replace(cfg, matching=replace(cfg.matching, mode='semantic_nli'))
    new = EnhancedDocumentAligner(embedder, TemplateRegistry(), cfg, [])
    try:
        pairs = evidence_pair_scores([a, b], new, [], {}, 10)
        assert max(p.score for p in pairs) >= .20
        passages, _ = new.align(a, b)
        assert any('nli verified' in p.note for p in passages)
        assert all(p.query_passage in a.full_text and p.matched_passage in b.full_text
                   for p in passages)
        assert all(p.query_page is None and p.matched_page is None for p in passages)
    finally:
        new.release()


def test_truncated_or_contradictory_verifier_scores_cannot_admit_new_matches(embedder):
    a, b = load_regression(['06_policy_argument.docx', '07_policy_rewording.docx'])
    cfg = Config()
    cfg = replace(cfg, matching=replace(cfg.matching, mode='semantic_nli'))

    class StubVerifier:
        kind = 'nli'

        def __init__(self, value):
            self.value = value

        def score_pairs(self, pairs):
            return [self.value] * len(pairs)

    for score in [SemanticScore(.99, .9), SemanticScore(.99, 0, True)]:
        aligner = EnhancedDocumentAligner(embedder, TemplateRegistry(), cfg, [])
        aligner.verifier = StubVerifier(score)
        assert aligner.coverage(a, b) == 0


def test_hybrid_keeps_nli_veto_and_unions_evidence_spans(embedder):
    a, b = load_regression(['06_policy_argument.docx', '07_policy_rewording.docx'])
    cfg = Config()
    cfg = replace(cfg, matching=replace(cfg.matching, mode='semantic_hybrid'))

    class StubHybrid:
        kind = 'hybrid'

        def __init__(self, contradiction):
            self.contradiction = contradiction

        def score_pairs(self, pairs):
            return [SemanticScore(.9, self.contradiction, False, .01)] * len(pairs)

    for contradiction, expected in [(.9, False), (.01, True)]:
        aligner = EnhancedDocumentAligner(embedder, TemplateRegistry(), cfg, [])
        aligner.verifier = StubHybrid(contradiction)
        coverage = aligner.coverage(a, b)
        assert bool(coverage) == expected
        assert 0 <= coverage <= 1
        passages, _ = aligner.align(a, b)
        for passage in passages:
            assert passage.query_passage in a.full_text
            assert passage.matched_passage in b.full_text


def test_matching_configuration_rejects_invalid_bounds():
    from integrity_engine.config import MatchingConfig
    with pytest.raises(ValueError):
        MatchingConfig(retrieval_topk=0)
    with pytest.raises(ValueError):
        MatchingConfig(nli_contradiction_max=1.1)
