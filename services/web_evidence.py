"""Allowlisted fetched evidence. Search-generated prose is never evidence."""
import asyncio
import hashlib
import ipaddress
import socket
from datetime import datetime, timezone
from urllib.parse import urlparse, urljoin, urlunparse

import httpx
from bs4 import BeautifulSoup

PRIMARY = ('gov.il', 'knesset.gov.il')
SECONDARY = ('kolzchut.org.il', 'harel-group.co.il', 'menoramivt.co.il', 'fnx.co.il', 'analyst.co.il')
MAX_BYTES = 12 * 1024 * 1024


def read_docx_source(payload):
    """Bound decompression before parsing a Word source, preserving tables."""
    import io
    import zipfile
    from services.document_service import extract_docx_bytes
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        entries = archive.infolist()
        if len(entries) > 4096 or sum(e.file_size for e in entries) > 4 * MAX_BYTES:
            raise ValueError('Expanded Word source exceeds size limit')
        if any(e.flag_bits & 1 for e in entries) or 'word/document.xml' not in archive.namelist():
            raise ValueError('Source is not an unencrypted Word document')
    return extract_docx_bytes(payload)


def is_discovery_redirect(url):
    parsed = urlparse(url)
    return (parsed.scheme == 'https' and parsed.hostname == 'vertexaisearch.cloud.google.com'
            and parsed.path.startswith('/grounding-api-redirect/') and not parsed.username
            and not parsed.password and parsed.port in (None,443))


def source_kind(url):
    parsed = urlparse(url)
    host = (parsed.hostname or '').lower().rstrip('.')
    if parsed.scheme != 'https' or parsed.username or parsed.password or parsed.port not in (None,443):
        raise ValueError('Only HTTPS allowlisted sources are permitted')
    for kind, roots in (('official_web', PRIMARY), ('secondary_web', SECONDARY)):
        if any(host == d or host.endswith('.' + d) for d in roots):
            return kind
    raise ValueError('Source is not allowlisted')


async def validate_dns(url, allow_discovery=False):
    if not (allow_discovery and is_discovery_redirect(url)):
        source_kind(url)
    host = urlparse(url).hostname
    records = await asyncio.to_thread(socket.getaddrinfo, host, 443, type=socket.SOCK_STREAM)
    if not records or any(not ipaddress.ip_address(r[4][0]).is_global for r in records):
        raise ValueError('Private or non-global destination rejected')
    return records[0][4][0]


async def fetch_source(url):
    # Strict host allowlist plus DNS checks on every redirect. trust_env=False
    # avoids inheriting an arbitrary HTTP proxy from untrusted runtime input.
    async with httpx.AsyncClient(timeout=10, follow_redirects=False, trust_env=False) as client:
        for _ in range(4):
            address = await validate_dns(url, allow_discovery=True)
            parsed = urlparse(url)
            literal = f'[{address}]' if ':' in address else address
            pinned_url = urlunparse(parsed._replace(netloc=literal))
            # Pin the vetted address while preserving TLS SNI and Host. There is
            # no second DNS lookup that could rebind to an internal address.
            async with client.stream('GET', pinned_url, headers={'User-Agent':'RegBot/2 evidence reader','Host':parsed.hostname},
                                     extensions={'sni_hostname':parsed.hostname}) as response:
                if response.is_redirect:
                    url = urljoin(url, response.headers['location'])
                    continue
                response.raise_for_status()
                # Google's intermediary may locate a page, but is never itself
                # an allowed evidence publisher.
                source_kind(url)
                if int(response.headers.get('content-length', '0')) > MAX_BYTES:
                    raise ValueError('Source exceeds size limit')
                payload = bytearray()
                async for block in response.aiter_bytes():
                    payload.extend(block)
                    if len(payload) > MAX_BYTES:
                        raise ValueError('Source exceeds size limit')
                content_type = response.headers.get('content-type','').lower()
            if 'pdf' in content_type or bytes(payload[:5]) == b'%PDF-':
                import fitz
                with fitz.open(stream=bytes(payload), filetype='pdf') as doc:
                    pages = [{'page': i, 'content': p.get_text(sort=True)} for i,p in enumerate(doc,1)]
                title = urlparse(url).path.rsplit('/',1)[-1]
            elif 'wordprocessingml' in content_type or bytes(payload[:4]) == b'PK\x03\x04':
                pages = [{'page': None, 'content': read_docx_source(bytes(payload))}]
                title = urlparse(url).path.rsplit('/',1)[-1]
            elif 'html' in content_type or 'text/plain' in content_type:
                soup = BeautifulSoup(bytes(payload), 'html.parser')
                title = soup.title.get_text(' ', strip=True) if soup.title else url
                for tag in soup(['script','style','nav','header','footer']):
                    tag.decompose()
                pages = [{'page':None, 'content':soup.get_text('\n',strip=True)}]
            else:
                raise ValueError('Unsupported source media type')
            if not any(p['content'].strip() for p in pages):
                raise ValueError('Source has no readable text')
            return {'url':url, 'title':title, 'kind':source_kind(url), 'pages':pages,
                    'source_hash':hashlib.sha256(payload).hexdigest(),
                    'retrieved_at':datetime.now(timezone.utc).isoformat()}
    raise ValueError('Too many redirects')


def sanitized_search(plan):
    # Only bounded domain vocabulary leaves the private conversation. Arbitrary
    # LLM-generated queries, names, account IDs and dates are never forwarded.
    vocabulary = {
        'product': {'קרן השתלמות','קופת גמל','קופת גמל להשקעה','קרן פנסיה','ביטוח מנהלים'},
        'operation': {'משיכה','הפקדה','מיסוי','הלוואה','דמי ניהול','ניוד','מסלול השקעה','זכויות עמיתים','פיצויים','דיווח'},
        'population': {'שכיר','עצמאי','מעסיק','מוטב','יורש','גוף מוסדי'},
    }
    parts = [plan[k] for k,allowed in vocabulary.items() if plan.get(k) in allowed]
    year = plan.get('tax_year')
    if isinstance(year,int) and 1900 <= year <= 2100:
        parts.append(str(year))
    return ' '.join(parts) + ' הוראות רגולציה ישראל' if parts else None


async def supplement(plan, gateway, existing, trace):
    from services.knowledge import split_spans, ENC
    from services.evidence_search import bm25
    query = sanitized_search(plan)
    if not query:
        trace.setdefault('web_errors',[]).append('No safe domain query available')
        return []
    trace['web_query'] = query
    seen, evidence = set(), []
    attempts = 0
    for suffix in ('', ' מקור רשמי חוק תקנות'):
        links = await gateway.discover(query + suffix)
        for item in links:
            url = item['url']
            if url in seen:
                continue
            seen.add(url)
            try:
                if not is_discovery_redirect(url):
                    source_kind(url)
            except ValueError:
                continue
            if attempts >= 6:
                break
            attempts += 1
            try:
                source = await fetch_source(url)
                candidates = []
                for page in source['pages']:
                    for left,right in split_spans(page['content']):
                        text = page['content'][left:right]
                        span_hash = hashlib.sha256(text.encode()).hexdigest()
                        candidates.append({'id':f"W{source['source_hash'][:16]}-{span_hash[:16]}",
                            'content':text, 'title':source['title'], 'url':source['url'], 'kind':source['kind'],
                            'section':None, 'page_start':page['page'], 'page_end':page['page'],
                            'source_hash':source['source_hash'], 'span_hash':span_hash,
                            'retrieved_at':source['retrieved_at'], 'effective_date':None,
                            'metadata_verified':False})
                scores = bm25(plan['standalone_question'], [c['content'] for c in candidates])
                evidence.extend(candidates[i] for i in sorted(range(len(scores)),key=lambda i:-scores[i])[:3])
            except (ValueError, httpx.HTTPError, OSError) as error:
                trace.setdefault('web_errors',[]).append({'url':url, 'error':type(error).__name__})
        if evidence or attempts >= 6:
            break
    used = sum(len(ENC.encode(__import__('json').dumps(e,ensure_ascii=False))) for e in existing)
    selected = []
    for e in evidence:
        size = len(ENC.encode(__import__('json').dumps(e,ensure_ascii=False)))
        if used + size <= 24000:
            selected.append(e)
            used += size
    trace['web_evidence'] = selected
    return selected
