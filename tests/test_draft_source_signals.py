import json

import pytest

from services.knowledge import prepare_document
from services.evidence_search import public_evidence


@pytest.mark.parametrize('opening', ['טיוטת תקנות', 'טיוטה של חוק', 'סיווג: כללי <טיוטה>'])
def test_draft_marker_survives_configured_type_and_legacy_current(opening):
    text = opening + '\n' + 'הוראות הנוגעות לניהול חשבון עמית בקופת גמל. '*5
    _, card, issues, chunks = prepare_document(text, {
        'id': 1, 'title': 'הוראות קופות גמל', 'document_type': 'חוזר', 'lifecycle_status': 'current'})
    assert 'draft_status_requires_review' in issues
    assert card['draft_markers'] == [opening]
    assert card['lifecycle_status'] == 'unknown' and not card['metadata_verified']
    evidence = public_evidence(dict(chunks[0], id='source', card=json.dumps(card)))
    assert evidence['draft_markers'] == [opening]


def test_discussing_a_draft_does_not_automatically_declare_document_a_draft():
    _, card, issues, _ = prepare_document('לאחר פרסום טיוטה התקבלו הערות.\n'+'הוראות לקופות גמל. '*10,
        {'id': 1, 'title': 'חוזר מחייב', 'lifecycle_status': 'current'})
    assert card['lifecycle_status'] == 'unknown'
    assert 'draft_status_requires_review' in issues
