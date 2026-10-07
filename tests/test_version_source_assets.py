import json
import sqlite3

import pytest

from services.source_assets import store_asset
from scripts.backup_restore import backup, restore


def source_database(root):
    root.mkdir()
    current = root/'current.txt'; current.write_text('new source')
    with sqlite3.connect(root/'regbot.db') as db:
        db.execute('CREATE TABLE documents(id INTEGER PRIMARY KEY, text_path TEXT, original_path TEXT)')
        db.execute('INSERT INTO documents VALUES(1,?,?)', (str(current),str(current)))
        db.execute('CREATE TABLE evidence_versions(id TEXT PRIMARY KEY, card TEXT)')
        for version in ('old','new'):
            assets = {role:store_asset(root, f'{version} {role}'.encode(), '.txt') for role in ('original','extracted')}
            db.execute('INSERT INTO evidence_versions VALUES(?,?)', (version,json.dumps({'source_assets':assets})))


def test_old_and_new_sources_survive_restore_and_second_backup(tmp_path):
    source = tmp_path/'source'; source_database(source)
    snapshot = tmp_path/'snapshot'; backup(source,snapshot)
    restored = tmp_path/'restored'; restore(snapshot,restored)
    with sqlite3.connect(restored/'regbot.db') as db:
        for version,raw in db.execute('SELECT id,card FROM evidence_versions'):
            for role,asset in json.loads(raw)['source_assets'].items():
                from pathlib import Path
                path = Path(asset['path'])
                assert path.is_relative_to(restored)
                assert path.read_text() == f'{version} {role}'
    backup(restored,tmp_path/'second-snapshot')


@pytest.mark.parametrize('mutation',['omitted','duplicate','checksum','wrong_version'])
def test_incomplete_version_manifest_rejected_before_restore(tmp_path,mutation):
    source = tmp_path/'source'; source_database(source)
    snapshot = tmp_path/'snapshot'; backup(source,snapshot)
    path = snapshot/'manifest.json'; manifest = json.loads(path.read_text())
    files = manifest['version_files']
    if mutation == 'omitted': files.pop()
    elif mutation == 'duplicate': files.append(files[0])
    elif mutation == 'checksum': files[0]['sha256'] = 'bad'
    else: files[0]['version_id'] = 'unknown'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError): restore(snapshot,tmp_path/'restored')
    assert not (tmp_path/'restored').exists()


def test_asset_corruption_is_not_overwritten_or_backed_up(tmp_path):
    source = tmp_path/'source'; source_database(source)
    asset = store_asset(source,b'old original','.txt')
    from pathlib import Path
    Path(asset['path']).write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='checksum'): store_asset(source,b'old original','.txt')
    with pytest.raises(ValueError,match='checksum'): backup(source,tmp_path/'snapshot')


def test_registration_requires_exact_both_hashes_and_rolls_back(tmp_path):
    import hashlib
    from scripts.register_version_sources import register
    root = tmp_path/'sources'; root.mkdir()
    original = root/'source.pdf'; original.write_bytes(b'original')
    extracted = root/'source.txt'; extracted.write_text('exact text',encoding='utf-8')
    database = tmp_path/'regbot.db'
    card = {'original_checksum':hashlib.sha256(original.read_bytes()).hexdigest()}
    with sqlite3.connect(database) as db:
        db.execute('CREATE TABLE evidence_versions(id TEXT PRIMARY KEY,source_hash TEXT,card TEXT)')
        db.execute('INSERT INTO evidence_versions VALUES(?,?,?)',('v',hashlib.sha256(b'exact text').hexdigest(),json.dumps(card)))
    with pytest.raises(ValueError,match='Unknown version'):
        register(database,['v','missing'],[root])
    with sqlite3.connect(database) as db:
        assert 'source_assets' not in json.loads(db.execute('SELECT card FROM evidence_versions').fetchone()[0])
    result = register(database,['v'],[root])
    assert result['activated'] is False
    extracted.write_text('modified text',encoding='utf-8')
    with pytest.raises(ValueError,match='Exact source assets'):
        register(database,['v'],[root])
