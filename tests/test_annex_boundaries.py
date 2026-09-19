import pytest

from services.knowledge import prepare_document


@pytest.mark.parametrize('heading',["נספח א'", "'נספח א", '  נספח א – טופס', 'נספח-תיקון הוראות'])
def test_annex_cannot_inherit_previous_operative_section(heading):
    text='19. ביטול חוזרים\nהחוזר הקודם בטל.\n'+heading+'\nפרטי הטופס\n1. שדה חובה\nשם\n'
    _,card,_,chunks=prepare_document(text,{'id':1,'title':'מסמך בדיקה'})
    cancellation=next(c for c in chunks if c['section'].startswith('19.'))
    assert 'פרטי הטופס' not in cancellation['section_text']
    form=next(c for c in chunks if 'פרטי הטופס' in c['content'])
    assert form['section'].startswith('נספח')
    nested=next(c for c in chunks if c['content'].startswith('1.'))
    assert nested['section'].startswith('נספח') and 'ביטול חוזרים' not in nested['section']
    assert ''.join(c['content'] for c in chunks)==text


def test_inline_annex_reference_is_not_a_new_section():
    text="4. הצטרפות\nיש למלא טופס בנספח א'.\nנספחים נלווים יישלחו בנפרד.\n"
    _,card,_,chunks=prepare_document(text,{'id':1,'title':'בדיקה'})
    assert len(card['section_map'])==1
    assert ''.join(c['content'] for c in chunks)==text


def test_pdf_line_break_between_number_dot_and_title_preserves_section():
    text='מבוא\n3\n.\n צירוף עמית\nהוראה מהותית\n4\n . חריגים\nחריג\n'
    _,card,_,chunks=prepare_document(text,{'id':1,'title':'בדיקה'})
    assert len(card['section_map'])==3
    assert 'צירוף עמית' in chunks[1]['section']
    assert 'חריגים' in chunks[2]['section']
    assert ''.join(c['content'] for c in chunks)==text


def test_year_followed_by_period_is_not_numbered_heading():
    text='מבוא\n1964\n.\nתוכן ממשיך\n'
    _,card,_,chunks=prepare_document(text,{'id':1,'title':'בדיקה'})
    assert len(card['section_map'])==1
