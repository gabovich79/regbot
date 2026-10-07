import json
import pytest

from config import EMBEDDING_MODEL
from scripts.rebuild_structure import stage_same_source


@pytest.mark.asyncio
async def test_reuse_requires_identical_embedding_input(monkeypatch):
    suffix = 'נושאים (נגזר): \nאוכלוסיות (נגזר): \n'
    old = dict(title='title', topics=[], populations=[], embedding=[1, 2], section_map=[])
    version = dict(id='old', document_id=1, source_hash='hash', embedding_model=EMBEDDING_MODEL,
                   card=json.dumps(old))
    class DB:
        async def execute(self, *args): return self
        async def fetchall(self):
            return [dict(content='same', context='context'+suffix, embedding='[1,2]'),
                    dict(content='changed', context='old'+suffix, embedding='[3,4]')]
    class Gateway:
        async def embed(self, texts):
            assert texts == ['new'+suffix+'changed']
            return [[5, 6]]
    async def stage(db, ident, source_hash, model, card, issues, chunks, vectors):
        assert vectors == [[1, 2], [5, 6]]
        assert card['section_map'] == ['new boundary']
        assert card['embedding'] == [1, 2]
        return 'new'
    monkeypatch.setattr('models.evidence_store.stage', stage)
    card = dict(title='title', source_hash='hash', section_map=['new boundary'], structure_review={})
    chunks = [dict(context='context', content='same'), dict(context='new', content='changed')]
    assert await stage_same_source(DB(), version, card, [], chunks, Gateway()) == ('new', 1)


@pytest.mark.asyncio
async def test_reuse_rejects_changed_source_before_provider_call():
    with pytest.raises(AssertionError):
        await stage_same_source(None, {'source_hash':'old'}, {'source_hash':'new'}, [], [], None)
