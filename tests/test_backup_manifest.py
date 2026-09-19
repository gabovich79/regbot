import json
import sqlite3

import pytest

from scripts.backup_restore import backup, restore


@pytest.fixture
def snapshot(tmp_path):
    source = tmp_path/'source'
    source.mkdir()
    (source/'source.txt').write_text('original source', encoding='utf-8')
    with sqlite3.connect(source/'regbot.db') as db:
        db.execute('CREATE TABLE documents(id INTEGER PRIMARY KEY, text_path TEXT, original_path TEXT)')
        db.execute('INSERT INTO documents VALUES(1,?,NULL)', (str(source/'source.txt'),))
    target = tmp_path/'snapshot'
    backup(source, target)
    return target


def test_missing_database_does_not_create_source_database(tmp_path):
    source = tmp_path/'source'
    source.mkdir()
    with pytest.raises(sqlite3.OperationalError):
        backup(source, tmp_path/'snapshot')
    assert not (source/'regbot.db').exists()
    assert not (tmp_path/'snapshot').exists()


@pytest.mark.parametrize('mutation', ['omitted', 'duplicate', 'wrong_document'])
def test_incomplete_or_ambiguous_manifest_rejected_before_copy(snapshot, tmp_path, mutation):
    manifest = json.loads((snapshot/'manifest.json').read_text())
    if mutation == 'omitted':
        manifest['files'] = []
    elif mutation == 'duplicate':
        manifest['files'] *= 2
    else:
        manifest['files'][0]['document_id'] = 99
    (snapshot/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        restore(snapshot, tmp_path/'restored')
    assert not (tmp_path/'restored').exists()


def test_restore_does_not_recurse_into_snapshot(snapshot):
    with pytest.raises(ValueError, match='outside'):
        restore(snapshot, snapshot/'nested')
    assert not (snapshot/'nested').exists()


def test_unregistered_files_are_not_restored(snapshot, tmp_path):
    (snapshot/'unverified.txt').write_text('not in manifest')
    restore(snapshot, tmp_path/'restored')
    assert not (tmp_path/'restored/unverified.txt').exists()
    assert (tmp_path/'restored/files/1-text_path.txt').read_text() == 'original source'
