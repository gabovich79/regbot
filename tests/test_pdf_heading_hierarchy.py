from services.document_service import _page_record
from services.knowledge import prepare_document


def test_pdf_line_alignment_identifies_bold_numeric_markers_without_changing_text():
    class Page:
        def get_text(self, mode=None, **kwargs):
            if mode is None: return '13\n.\nהודעה\n2\n.\nפרט נדרש'
            return {'blocks':[{'lines':[
                {'spans':[{'text':text,'flags':flags}]} for text,flags in
                [('13',20),('.',20),('הודעה',16),('2',4),('.',4),('פרט נדרש',4)]]}]}
    result=_page_record(Page(),1)
    assert result['text']==Page().get_text()
    assert result['bold_numbered_starts']==[0]


def test_nested_plain_item_keeps_its_full_parent_across_pages():
    first='1. פתיחה\nכלל\n2. הליך\nתנאי\n13. הודעה\nחובה לשלוח'
    second='2. פרט נדרש\nחריג ותוכן ההודעה'
    pages=[{'page_number':1,'text':first,'bold_numbered_starts':[first.index(x) for x in ['1.','2.','13.']]},
           {'page_number':2,'text':second,'bold_numbered_starts':[]}]
    _,card,issues,chunks=prepare_document('',{'id':1,'title':'מסמך'},pages)
    parent=[c for c in chunks if '2. פרט נדרש' in c['content']]
    assert parent and all(c['section']=='13. הודעה' for c in parent)
    assert all('חובה לשלוח' in c['section_text'] for c in parent)
    assert card['structure_review']['suppressed_numbered_boundaries']
    assert 'pdf_hierarchy_inferred_from_typography_requires_review' in issues


def test_unaligned_page_disables_typographic_inference_for_whole_document():
    text='1. ראשי\n2. שני\n3. שלישי\n4. רביעי'
    pages=[{'page_number':1,'text':text,'bold_numbered_starts':[0,8,16]},
           {'page_number':2,'text':'המשך','bold_numbered_starts':None}]
    _,card,_,_=prepare_document('',{'id':1,'title':'מסמך'},pages)
    assert card['structure_review']['mode']=='text_patterns'
    assert any(s['section']=='4. רביעי' for s in card['section_map'])
