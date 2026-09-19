import json

import pytest

from services.evidence_search import cosine, expand_candidates, fused_candidates


def chunk(ordinal, *, version='v1', section='Repeated heading', parent='parent', vector=None):
    return {'id': f'{version}-{ordinal}', 'version_id': version, 'ordinal': ordinal,
            'section': section, 'section_text': parent, 'content': 'source text', 'context': '',
            'document_id': version, 'embedding': json.dumps(vector or [1., 0.]),
            'card': json.dumps({'title': version, 'summary': 'source text', 'embedding': [1., 0.]})}


def test_repeated_heading_does_not_expand_a_remote_parent():
    chunks = [chunk(i, parent='earlier' if i < 3 else 'separator' if i == 3 else 'later') for i in range(8)]
    expanded = expand_candidates([chunks[6]], chunks, 'question')
    assert [c['ordinal'] for c, _, _ in expanded] == [6, 5, 7, 4]


def test_identical_noncontiguous_parent_is_not_same_section():
    chunks = [chunk(i, parent='separator' if i == 3 else 'identical') for i in range(8)]
    assert {c['ordinal'] for c, _, _ in expand_candidates([chunks[6]], chunks, '')} == {4, 5, 6, 7}


def test_long_parent_does_not_starve_other_selected_document():
    first = [chunk(i) for i in range(100)]
    second = [chunk(i, version='v2') for i in range(3)]
    result = expand_candidates([first[80], second[1]], first + second, '')
    ids = [c['id'] for c, _, _ in result]
    assert ids[:6] == ['v1-80', 'v2-1', 'v1-79', 'v2-0', 'v1-81', 'v2-2']
    assert len(ids) == len(set(ids)) == 103


def test_unmatched_lexical_search_adds_no_arbitrary_id_bonus():
    chunks = [chunk(0, version='a', vector=[0., 1.]), chunk(0, version='z', vector=[1., 0.])]
    ranked = fused_candidates('unmatched', [1., 0.], chunks)
    best = next(c for c in ranked if c['id'] == 'z-0')
    # Dense section rank 1 plus card rank 2; neither lexical route matched.
    assert best['rrf_score'] == pytest.approx(1/61 + 1/62)


@pytest.mark.parametrize('query,vector', [([float('nan')], [1.]), ([[1.]], [[1.]]), ([], []), ([1.], [float('inf')])])
def test_invalid_embedding_fails_before_ranking(query, vector):
    with pytest.raises(ValueError):
        cosine(query, vector)
