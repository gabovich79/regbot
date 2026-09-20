"""Backfill immutable assets only when both existing version hashes match."""
import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

from services.source_assets import store_asset


def register(database, versions, directories):
    originals, texts = {}, {}
    for directory in directories:
        for path in Path(directory).rglob('*'):
            if not path.is_file() or path.suffix.lower() not in ('.pdf','.docx','.txt','.html','.htm'):
                continue
            blob = path.read_bytes()
            originals.setdefault(hashlib.sha256(blob).hexdigest(),path)
            if path.suffix.lower() == '.txt':
                value = path.read_text(encoding='utf-8').encode('utf-8')
                texts.setdefault(hashlib.sha256(value).hexdigest(), (path,value))
    database = Path(database).resolve()
    if not database.is_file():
        raise ValueError('Missing database')
    with sqlite3.connect(database) as db:
        changes = []
        for version in versions:
            row = db.execute('SELECT source_hash,card FROM evidence_versions WHERE id=?',(version,)).fetchone()
            if row is None:
                raise ValueError('Unknown version')
            source_hash, raw = row
            card = json.loads(raw)
            original = originals.get(card.get('original_checksum'))
            extracted = texts.get(source_hash)
            if original is None or extracted is None:
                raise ValueError(f'Exact source assets not found for version {version}')
            card['source_assets'] = {
                'original':store_asset(database.parent,original.read_bytes(),original.suffix.lower()),
                'extracted':store_asset(database.parent,extracted[1],'.txt'),
            }
            if card['source_assets']['original']['sha256'] != card['original_checksum']:
                raise ValueError('Original changed during registration')
            changes.append((json.dumps(card,ensure_ascii=False),version))
        db.executemany('UPDATE evidence_versions SET card=? WHERE id=?',changes)
    return {'registered_versions':list(versions),'activated':False,'reindexed':False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database',required=True)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--sources',nargs='+',required=True)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding='utf-8'))['manifest']
    print(json.dumps(register(args.database,list(manifest.values()),args.sources)))
