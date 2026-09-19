import json
from docx import Document
from scripts.inventory_local_sources import inventory
from services.document_integrity_service import _official_numbers
from services.document_profile_service import _extract_official_number


def test_wrapped_regulatory_number_is_not_replaced_by_older_reference():
    text = 'חוזר גופים מוסדיים 2021-9-\n5\nסיווג: כללי\nתיקון חוזר 2016-9-29'
    assert _official_numbers(text) == {'2021-9-5', '2016-9-29'}
    assert _extract_official_number('2021-9-5.pdf', text) == '2021-9-5'


def test_inventory_matches_content_and_refuses_to_overwrite(tmp_path):
    source, reference, output = (tmp_path/name for name in ('source','reference','output'))
    source.mkdir(); reference.mkdir()
    doc = Document(); doc.add_paragraph('תוכן מקור לבדיקה של התאמה מדויקת')
    doc.save(source/'different-name.docx')
    (reference/'1.txt').write_text('תוכן מקור לבדיקה של התאמה מדויקת',encoding='utf-8')
    (reference/'manifest.json').write_text(json.dumps([{'id':1,'title':'old name'}]),encoding='utf-8')
    result=inventory(source,reference,output)
    assert len(result['files'])==1
    row=result['files'][0]
    assert 'error' not in row
    assert row['candidates'][0]['exact_normalized_text']
    assert row['candidates'][0]['id']==1
    assert not row['candidates'][0]['original_already_retained']
    import pytest
    with pytest.raises(FileExistsError):
        inventory(source,reference,output)
