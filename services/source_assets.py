"""Content-addressed source snapshots, independent of mutable document rows."""
import hashlib
from pathlib import Path


def store_asset(root, content, suffix):
    digest = hashlib.sha256(content).hexdigest()
    target = Path(root)/'evidence_assets'/f'{digest}{suffix}'
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open('xb') as stream:
            stream.write(content)
    except FileExistsError:
        pass
    if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
        raise ValueError('Source asset checksum mismatch')
    return {'path': str(target.resolve()), 'sha256': digest}


async def snapshot_sources(db, original, text):
    databases = await (await db.execute('PRAGMA database_list')).fetchall()
    filename = next((r[2] for r in databases if r[1] == 'main'), '')
    if not filename:
        raise ValueError('Source snapshots require a file-backed database')
    original = Path(original)
    suffix = original.suffix.lower()
    if suffix not in ('.pdf', '.docx', '.txt', '.html', '.htm'):
        suffix = '.bin'
    root = Path(filename).parent
    return {
        'original': store_asset(root, original.read_bytes(), suffix),
        'extracted': store_asset(root, text.encode('utf-8'), '.txt'),
    }
