"""Reference integrity checks, not legal correctness or model quality scores."""
import ast
import hashlib
import json
import sqlite3
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.build_reference_review import build_bundle, load_originals, FIELDS

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    case = {'id': 'test', 'question': 'q', 'sources': [{'document_id': 1, 'pages': [1, 2]}],
            'review_blockers': ['professional review required']}
    case.update({field: [{'text': 'annotation', 'pages': [1, 2]}] for field in FIELDS})
    sources = {1: {'pages': [{'page_number': 1, 'text': 'exact\n first page'},
                            {'page_number': 2, 'text': 'condition continues here'}],
                   'original_sha256': 'original-version', 'title': 't', 'source_ref': 's'}}
    return {'cases': [case]}, sources


def test_cross_page_evidence_is_literal_and_version_bound():
    spec, sources = fixture()
    result = build_bundle(spec, sources)
    assert not result['is_acceptance_set'] and not result['cases'][0]['release_eligible']
    for span, page in zip(result['evidence'], sources[1]['pages']):
        assert span['quote'] == page['text']
        assert span['quote_sha256'] == hashlib.sha256(page['text'].encode()).hexdigest()
        assert span['char_end'] == len(page['text'])
        assert 'original-version' in span['id']
    for requirement in result['cases'][0]['requirements']:
        assert len(requirement['evidence_ids']) == 2


def test_same_page_in_a_new_version_gets_a_new_id():
    spec, sources = fixture()
    previous = build_bundle(spec, sources)
    sources[1]['original_sha256'] = 'new-version'
    current = build_bundle(spec, sources)
    assert {e['id'] for e in previous['evidence']}.isdisjoint(e['id'] for e in current['evidence'])


def test_corrected_extraction_of_same_original_changes_evidence_identity():
    spec, sources = fixture()
    before = build_bundle(spec, sources)
    sources[1]['pages'][0]['text'] += ' restored equation'
    after = build_bundle(spec, sources)
    assert before['evidence'][0]['id'] != after['evidence'][0]['id']
    assert before['evidence'][1]['id'] == after['evidence'][1]['id']


@pytest.mark.parametrize('mutation', [
    lambda c: c['conditions'][0].update(pages=[3]),
    lambda c: c.update(exceptions=[]),
    lambda c: c['rule'][0].update(text=''),
])
def test_reference_cannot_silently_drop_a_dimension_or_missing_page(mutation):
    spec, sources = fixture()
    mutation(spec['cases'][0])
    with pytest.raises(ValueError):
        build_bundle(spec, sources)


def test_original_checksum_mismatch_stops_before_extraction(tmp_path):
    original = tmp_path/'source.pdf'
    original.write_bytes(b'changed source')
    path = tmp_path/'test.db'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE documents (id INTEGER, original_path TEXT)')
        db.execute('INSERT INTO documents VALUES (1,?)', (str(original),))
    with pytest.raises(ValueError, match='Original version mismatch'):
        load_originals(path, {1: {'sha256': 'expected-original'}}, {1})


def test_diagnostic_questions_are_preserved_and_not_promoted_to_acceptance():
    from scripts.pilot_experiment import CASES
    spec = json.loads((ROOT/'eval/diagnostic-reference-spec.json').read_text(encoding='utf-8'))
    assert {c['id']: c['question'] for c in spec['cases']} == {c[0]: c[1] for c in CASES}
    assert all(c['review_blockers'] for c in spec['cases'])
    assert spec['status'] == 'draft_pending_professional_review'


def test_runtime_has_no_dependency_on_diagnostic_answers():
    # A fixture must never become a retrieval rule or a production answer route.
    for folder in ('services', 'routers', 'models'):
        for path in (ROOT/folder).glob('**/*.py'):
            source = path.read_text(encoding='utf-8')
            assert 'diagnostic-reference-spec' not in source
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    assert not (node.module or '').startswith(('eval.', 'scripts.build_reference_review', 'scripts.pilot_experiment'))
