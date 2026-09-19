"""Consistent SQLite snapshot + verified source files; restore only to a new directory.

Pause administrative document writes while backing up; concurrent source-file
changes are detected and abort the snapshot. Chat writes use SQLite.backup.
"""
import argparse
import hashlib
import json
import shutil
import sqlite3
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def within(path, root):
    path = path.resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Path escapes the selected data directory')
    return path


def backup(data_dir, destination):
    root, output = Path(data_dir).resolve(), Path(destination).resolve()
    if output.exists() or output.is_relative_to(root):
        raise ValueError('Backup must use a new directory outside the live data directory')
    # Never create an empty database in a mistyped source directory.
    source = sqlite3.connect((root/'regbot.db').as_uri()+'?mode=ro', uri=True)
    output.mkdir(parents=True)
    snapshot = sqlite3.connect(output/'regbot.db')
    try:
        source.backup(snapshot)
        if snapshot.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
            raise ValueError('Database integrity check failed')
        columns = {r[1] for r in snapshot.execute('PRAGMA table_info(documents)')}
        fields = [f for f in ('text_path','original_path') if f in columns]
        rows = snapshot.execute('SELECT '+','.join(['id']+fields)+' FROM documents').fetchall()
        files = []
        for row in rows:
            for field,value in zip(fields,row[1:]):
                if not value:
                    continue
                path = Path(value)
                if not path.is_absolute():
                    path = root/path
                path = within(path,root)
                if not path.is_file():
                    raise ValueError(f'Missing source for document {row[0]} ({field})')
                before = digest(path)
                relative = Path('files')/f'{row[0]}-{field}{path.suffix}'
                target = output/relative
                target.parent.mkdir(exist_ok=True)
                shutil.copyfile(path,target)
                if digest(path) != before or digest(target) != before:
                    raise ValueError('Source changed during snapshot; repeat with admin writes paused')
                files.append({'document_id':row[0], 'field':field, 'path':relative.as_posix(), 'sha256':before})
    finally:
        snapshot.close()
        source.close()
    manifest = {'database_sha256':digest(output/'regbot.db'),'files':files}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return manifest


def restore(snapshot_dir, destination):
    root, output = Path(snapshot_dir).resolve(), Path(destination).resolve()
    if output.exists() or output.is_relative_to(root):
        raise ValueError('Restore destination must be new and outside the snapshot directory')
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    if digest(root/'regbot.db') != manifest['database_sha256']:
        raise ValueError('Database checksum mismatch')
    source_db = sqlite3.connect((root/'regbot.db').as_uri()+'?mode=ro', uri=True)
    try:
        columns = {r[1] for r in source_db.execute('PRAGMA table_info(documents)')}
        fields = [f for f in ('text_path', 'original_path') if f in columns]
        expected = {(row[0], field) for row in source_db.execute('SELECT '+','.join(['id']+fields)+' FROM documents')
                    for field, value in zip(fields, row[1:]) if value}
    finally:
        source_db.close()
    found, paths = set(), set()
    for file in manifest['files']:
        path = within(root/file['path'],root)
        if digest(path) != file['sha256'] or file['field'] not in ('text_path','original_path'):
            raise ValueError('Invalid source checksum or field')
        key = (file['document_id'], file['field'])
        if key in found or path in paths or path == root/'regbot.db' or path == root/'manifest.json':
            raise ValueError('Duplicate or reserved manifest entry')
        found.add(key)
        paths.add(path)
    if found != expected:
        raise ValueError('Source manifest does not match database file references')
    # Copy only verified assets; unrelated files and symlinks are not a backup.
    output.mkdir(parents=True)
    shutil.copyfile(root/'regbot.db',output/'regbot.db')
    shutil.copyfile(root/'manifest.json',output/'manifest.json')
    if digest(output/'regbot.db') != manifest['database_sha256']:
        raise ValueError('Database changed during restore')
    for file in manifest['files']:
        destination_path = within(output/file['path'],output)
        destination_path.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(within(root/file['path'],root),destination_path)
        if digest(destination_path) != file['sha256']:
            raise ValueError('Source changed during restore')
    db = sqlite3.connect(output/'regbot.db')
    try:
        for file in manifest['files']:
            destination_path = within(output/file['path'],output)
            db.execute(f"UPDATE documents SET {file['field']}=? WHERE id=?",(str(destination_path),file['document_id']))
        db.commit()
        if db.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
            raise ValueError('Restored database failed integrity check')
    finally:
        db.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('operation',choices=['backup','restore'])
    parser.add_argument('source')
    parser.add_argument('destination')
    args = parser.parse_args()
    if args.operation == 'backup':
        backup(args.source,args.destination)
    else:
        restore(args.source,args.destination)
    print('Verified',args.operation)
