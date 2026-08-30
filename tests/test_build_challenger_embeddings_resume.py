import json

import pytest

from scripts.build_challenger_embeddings import (
    BATCH_SIZE,
    _hash,
    embed_missing,
    load_existing,
    node_text,
    profile_text,
    save_progress,
)


@pytest.mark.asyncio
async def test_embed_missing_only_embeds_missing_texts(tmp_path, monkeypatch):
    calls = []

    async def fake_embed_texts(texts):
        calls.append(texts)
        return [[float(i)] * 4 for i in range(len(texts))]

    monkeypatch.setattr("scripts.build_challenger_embeddings.embed_texts", fake_embed_texts)

    texts = ["alpha", "beta"]
    existing = {_hash("alpha"): [0.0, 0.0, 0.0, 0.0]}

    results = await embed_missing(texts, existing, "nodes")

    assert calls == [["beta"]]
    assert len(results) == 2
    assert results[_hash("beta")] == [0.0, 0.0, 0.0, 0.0]


@pytest.mark.asyncio
async def test_embed_missing_processes_in_batches(tmp_path, monkeypatch):
    batch_sizes = []

    async def fake_embed_texts(texts):
        batch_sizes.append(len(texts))
        return [[float(i)] * 4 for i in range(len(texts))]

    monkeypatch.setattr("scripts.build_challenger_embeddings.embed_texts", fake_embed_texts)
    monkeypatch.setattr("scripts.build_challenger_embeddings.BATCH_SIZE", 2)

    texts = ["a", "b", "c", "d", "e"]

    results = await embed_missing(texts, {}, "nodes")

    assert batch_sizes == [2, 2, 1]
    assert len(results) == 5


def test_save_progress_then_load_existing_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.build_challenger_embeddings.CACHE_PATH", tmp_path / "cache.json")

    profile = {"document_id": 1, "canonical_title": "חוק בדיקה", "official_number": None, "topics": [], "profile_summary": ""}
    node = {"id": 1, "document_id": 1, "heading": "סעיף 25", "raw_text": "זכויות עמית אינן ניתנות להעברה."}

    profile_vector = [0.1] * 4
    node_vector = [0.9] * 4

    save_progress(
        [profile],
        [node],
        {_hash(profile_text(profile)): profile_vector},
        {_hash(node_text(node)): node_vector},
    )

    loaded = load_existing()

    assert loaded["profile_hashes"][_hash(profile_text(profile))] == profile_vector
    assert loaded["node_hashes"][_hash(node_text(node))] == node_vector
    assert len(loaded["profiles"]) == 1
    assert len(loaded["nodes"]) == 1


def test_load_existing_ignores_broken_cache(tmp_path, monkeypatch):
    cache = tmp_path / "cache.json"
    cache.write_text("{broken json", encoding="utf-8")
    monkeypatch.setattr("scripts.build_challenger_embeddings.CACHE_PATH", cache)

    loaded = load_existing()

    assert loaded["profiles"] == []
    assert loaded["nodes"] == []
    assert loaded["profile_hashes"] == {}
    assert loaded["node_hashes"] == {}
