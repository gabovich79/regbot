"""Audit original-source re-extraction; optionally stage into a restored copy only."""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path


async def stage_same_source(db, version, card, issues, chunks, gateway):
    """Reuse vectors only for identical model, context and original source text."""
    from config import EMBEDDING_MODEL
    from models.evidence_store import stage
    assert version['source_hash'] == card['source_hash']
    assert version['embedding_model'] == EMBEDDING_MODEL
    old_card = json.loads(version['card'])
    assert old_card['title'] == card['title']
    old_card.update(section_map=card['section_map'], structure_review=card['structure_review'])
    rows = await (await db.execute('SELECT content,context,embedding FROM evidence_chunks WHERE version_id=?',
                                  (version['id'],))).fetchall()
    cache = {(r['context'], r['content']): json.loads(r['embedding']) for r in rows}
    for chunk in chunks:
        chunk['context'] += f"נושאים (נגזר): {', '.join(old_card['topics'])}\nאוכלוסיות (נגזר): {', '.join(old_card['populations'])}\n"
    missing = [i for i, chunk in enumerate(chunks) if (chunk['context'], chunk['content']) not in cache]
    vectors = await gateway.embed([chunks[i]['context']+chunks[i]['content'] for i in missing]) if missing else []
    fresh = dict(zip(missing, vectors))
    all_vectors = [fresh[i] if i in fresh else cache[(chunk['context'], chunk['content'])] for i,chunk in enumerate(chunks)]
    new_version = await stage(db, version['document_id'], card['source_hash'], EMBEDDING_MODEL,
                              old_card, issues, chunks, all_vectors)
    return new_version, len(chunks)-len(missing)


async def main(args):
    os.environ['DATA_DIR'] = str(args.data.resolve())
    from models.database import get_db
    from services.document_service import extract_pdf_pages, extract_docx
    from services.knowledge import prepare_document, stage_document, ENC
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))['manifest'].copy()
    if args.output.exists():
        raise ValueError('Refusing to overwrite a completed audit')
    if args.stage:
        if not args.ledger or not args.prices:
            raise ValueError('Staging requires the existing campaign ledger and prices')
        os.environ.update(CAMPAIGN_BUDGET_DB=str(args.ledger.resolve()), CAMPAIGN_BUDGET_USD='20',
                          PROVIDER_PRICES_JSON=args.prices.read_text(encoding='utf-8'))
        import truststore
        truststore.inject_into_ssl()
    db = await get_db()
    results = []
    try:
        for ident in args.documents:
            version = await (await db.execute('SELECT * FROM evidence_versions WHERE id=?', (manifest[str(ident)],))).fetchone()
            old_card = json.loads(version['card'])
            metadata = dict(await (await db.execute('SELECT * FROM documents WHERE id=?', (ident,))).fetchone())
            asset = old_card['source_assets']['original']
            original = Path(asset['path'])
            assert hashlib.sha256(original.read_bytes()).hexdigest() == asset['sha256']
            metadata.update(original_path=str(original), title=old_card['title'])
            pages = extract_pdf_pages(str(original)) if original.suffix.lower()=='.pdf' else None
            text = '\n\n'.join(p['text'] for p in pages) if pages else extract_docx(str(original))
            source_hash, card, issues, chunks = prepare_document(text, metadata, pages)
            parents = [chunks[s['first_chunk']]['section_text'] for s in card['section_map']]
            cursor = 0
            for parent in parents:
                position = text.find(parent, cursor)
                assert position >= cursor and not text[cursor:position].strip(), 'Section cuts lost source content'
                cursor = position + len(parent)
            assert not text[cursor:].strip(), 'Section cuts lost source tail'
            for start, section in enumerate(card['section_map']):
                lo = section['first_chunk']
                hi = card['section_map'][start+1]['first_chunk'] if start+1<len(card['section_map']) else len(chunks)
                parent = chunks[lo]['section_text']
                covered = 0
                for chunk in chunks[lo:hi]:
                    position = parent.find(chunk['content'], max(0, covered-len(chunk['content'])))
                    assert 0 <= position <= covered, 'Gap in source chunk coverage'
                    covered = max(covered, position+len(chunk['content']))
                assert covered == len(parent), 'Missing source tail'
            sizes = [len(ENC.encode(p)) for p in parents]
            result = dict(document_id=ident, previous_version=version['id'], original_sha256=asset['sha256'],
                          text_unchanged=source_hash==version['source_hash'], source_hash=source_hash,
                          sections=len(parents), chunks=len(chunks), max_section_tokens=max(sizes),
                          oversized_sections=sum(n>24000 for n in sizes), issues=issues,
                          lossless_section_and_chunk_coverage=True, activated=False)
            if args.stage:
                from services.providers import Gateway
                gateway=Gateway(purpose='structure_rebuild', limit=1)
                try:
                    if args.reuse_embeddings:
                        new_version, reused = await stage_same_source(db, version, card, issues, chunks, gateway)
                        result['reused_identical_vectors'] = reused
                    else:
                        new_version, _ = await stage_document(db, metadata, text, pages, gateway=gateway)
                    manifest[str(ident)] = new_version
                    result['new_version'] = new_version
                except Exception as exc:
                    result['error_type'] = type(exc).__name__
                result['cost'] = gateway.spent
            results.append(result)
            args.output.write_text(json.dumps(dict(manifest=manifest, results=results, acceptance=False),
                                             ensure_ascii=False, indent=2), encoding='utf-8')
            print(json.dumps(result, ensure_ascii=False), flush=True)
    finally:
        await db.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('data','manifest','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--documents', type=int, nargs='+', required=True)
    parser.add_argument('--stage', action='store_true')
    parser.add_argument('--reuse-embeddings', action='store_true')
    parser.add_argument('--ledger', type=Path)
    parser.add_argument('--prices', type=Path)
    asyncio.run(main(parser.parse_args()))
