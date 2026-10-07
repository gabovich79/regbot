from services.section_context import pack_sections


def chunk(ordinal, section, parent, content=None, version='v1'):
    return dict(id=f'{version}-{ordinal}',version_id=version,ordinal=ordinal,
                section=section,section_text=parent,content=content or parent)


def public(c):
    return {'id':c['id'],'content':c['content']}


def test_continuation_kept_even_when_only_last_chunk_ranks():
    chunks=[chunk(0,'1. כלל','full','beginning'),chunk(1,'1. כלל','full','exception')]
    evidence,trace=pack_sections([chunks[1]],chunks,public)
    assert [e['id'] for e in evidence]==['v1-0','v1-1']
    assert trace['complete_section_count']==1


def test_no_partial_section_when_budget_is_too_small():
    chunks=[chunk(0,'1. כלל','full','x '*100),chunk(1,'1. כלל','full','y '*100)]
    evidence,trace=pack_sections([chunks[0]],chunks,public,budget=150)
    assert evidence==[]
    assert trace['omitted_section_seeds']==['v1-0']


def test_oversized_legacy_parent_is_explicitly_partial_not_empty():
    chunks=[chunk(i,'פרק','large parent','x '*50) for i in range(20)]
    evidence,trace=pack_sections([chunks[10]],chunks,public,budget=300)
    assert [e['id'] for e in evidence]==['v1-9','v1-10','v1-11']
    assert trace['partial_section_seeds']==['v1-0']
    assert trace['complete_section_count']==0


def test_governing_and_explicit_references_accompany_rule():
    chunks=[chunk(0,'1. זכאות','לפי סעיף 2 להלן'),chunk(1,'2. חריג','exception'),chunk(2,'3. תחולה','scope')]
    evidence,_=pack_sections([chunks[0]],chunks,public)
    assert {e['id'] for e in evidence}=={'v1-0','v1-1','v1-2'}


def test_external_reference_does_not_bind_local_same_number():
    chunks=[chunk(0,'1. זכאות','לפי סעיף 2 לחוק'),chunk(1,'2. אחר','unrelated')]
    evidence,trace=pack_sections([chunks[0]],chunks,public)
    assert [e['id'] for e in evidence]==['v1-0']
    assert trace['unresolved_section_references']


def test_reference_to_this_circular_remains_local():
    chunks=[chunk(0,'1. זכאות','לפי סעיף 2 לחוזר זה'),chunk(1,'2. חריג','exception')]
    evidence,trace=pack_sections([chunks[0]],chunks,public)
    assert len(evidence)==2
    assert not trace['unresolved_section_references']


def test_repeated_heading_does_not_merge_distant_sections():
    chunks=[chunk(0,'טופס','same'),chunk(1,'אחר','other'),chunk(2,'טופס','same')]
    evidence,_=pack_sections([chunks[2]],chunks,public)
    assert [e['id'] for e in evidence]==['v1-2']
