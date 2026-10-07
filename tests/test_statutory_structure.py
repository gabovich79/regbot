from services.document_structure import statutory_caption, boundaries
from services.knowledge import SECTION
from services.document_service import _page_record


def test_caption_is_not_amendment_metadata():
    assert statutory_caption('משיכת תשלומי מעביד מקרן השתלמות (תיקון מס 60)')
    assert statutory_caption('1984-) תשמד60 תשלום ריבית והפרשי הצמדה (תיקון מס')
    assert not statutory_caption('2002-) תשסב132 (תיקון מס')
    assert not statutory_caption('(תיקון מס 132) הוראת שעה')
    assert not statutory_caption('תשלום ריבית רגיל ללא עיצוב')
    assert statutory_caption('(תיקון מס 108) משיכה מקרן השתלמות לעצמאים (תיקון מס')
    assert statutory_caption('1993-) תשנ"ג95 הגדרות (תיקון מס')
    assert not statutory_caption('2005-) תשס"ה147 (תיקון מס2002-) תשס"ב132 (תיקון מס')
    assert statutory_caption('תשלומים (תיקון מס 95)')


def test_annex_parts_preserve_table_boundaries():
    text="נספח א\nחלק 1\nbody\n[תחילת טבלה במקור: 1]\nחלק 2\ncell\n[סוף טבלה במקור: 1]\nחלק 3\nlast"
    labels=list(boundaries(text,SECTION).values())
    assert 'חלק 1' in labels and 'חלק 3' in labels
    assert 'חלק 2' not in labels


class Page:
    def __init__(self, text, lines):
        self.text, self.lines = text, lines

    def get_text(self, mode=None, **kwargs):
        if mode == 'dict':
            return {'blocks': [{'lines': [{'spans': [{'text': text, 'flags': flags}]} for text, flags in self.lines]}]}
        return self.text


def test_caption_alignment_survives_unrelated_unmatched_line():
    title = '1993-) תשנ"ג95 הגדרות (תיקון מס'
    text = title+'\n.1 הגדרות גוף הטקסט'
    record = _page_record(Page(text, [('unmatched formula', 4), (title, 20)]), 1)
    assert record['text'] == text
    assert record['bold_numbered_starts'] is None
    assert record['bold_caption_starts'] == [0]


def test_plain_caption_requires_bold_and_numbered_following_clause():
    title = 'חובת האפוטרופוס לשלם'
    text = title+'\nתוכן הסעיף .6\n'
    assert _page_record(Page(text, [(title, 20)]), 1)['bold_caption_starts'] == [0]
    assert not _page_record(Page(text, [(title, 4)]), 1)['bold_caption_starts']
    assert not _page_record(Page(title+'\nטקסט ללא מספר', [(title, 20)]), 1)['bold_caption_starts']
    for metadata in ('2004-תשס"ה', '1979-ת"ט תשל"ט'):
        assert not _page_record(Page(metadata+'\nתוכן .6', [(metadata, 20)]), 1)['bold_caption_starts']


def test_repeated_caption_is_not_bound_to_guessed_occurrence():
    title = 'הגדרות (תיקון מס 95)'
    record = _page_record(Page(title+'\nbody\n'+title, [(title, 20)]), 1)
    assert not record['bold_caption_starts']
