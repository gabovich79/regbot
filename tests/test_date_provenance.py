from services.date_provenance import date_present
from services.evidence_contract import bind_units
import pytest


def test_regulatory_identifier_is_not_a_date():
    assert not date_present('2021-09-05','חוזר 2021-9-5')
    assert not date_present('2021-09-05','חוזר 2021-09-05')


def test_supported_formats_and_mismatch():
    assert date_present('2021-09-01','תחילה ביום 1 בספטמבר 2021')
    assert date_present('2021-09-01','2021 בספטמבר 1')
    assert date_present('2021-09-01','תחילתן 01.09.2021')
    assert date_present('2021-09-01','effective from 2021-09-01')
    assert not date_present('2021-09-05','תחילה ביום 1 בספטמבר 2021')


def test_unattributed_date_never_enters_renderable_components():
    part=lambda text:dict(text=text,source_ids=['E1'])
    unit=dict(rule=part('rule'),scope=[],conditions=[],exceptions=[],requirements=[],
              period=dict(**part('תחילה 2021-09-05'),start_date='2021-09-05',end_date=None))
    units,gaps=bind_units({'units':[unit],'missing':[]},[{'id':'E1','content':'חוזר 2021-9-5'}])
    assert not units[0]['period_known'] and gaps
    assert [c['kind'] for c in units[0]['components']]==['rule']


@pytest.mark.parametrize('date_id,accepted', [
    ('D1-Vversion1-C2', True),
    ('D2-Vversion1-C2', False),
    ('D1-Vversion2-C2', False),
])
def test_period_must_belong_to_rule_document_version(date_id, accepted):
    rule_id='D1-Vversion1-C1'
    unit=dict(rule={'text':'rule','source_ids':[rule_id]},scope=[],conditions=[],exceptions=[],requirements=[],
              period={'text':'תחילה ביום 1 בספטמבר 2021','source_ids':[date_id],
                      'start_date':'2021-09-01','end_date':None})
    evidence=[{'id':rule_id,'content':'rule'}, {'id':date_id,'content':'תחילה ביום 1 בספטמבר 2021'}]
    units,gaps=bind_units({'units':[unit],'missing':[]}, evidence)
    assert units[0]['period_known'] is accepted
    assert bool(gaps) is not accepted
    if not accepted:
        assert all(c['kind']!='period' for c in units[0]['components'])


def test_extra_rule_pointer_cannot_launder_date_from_other_document():
    unit=dict(rule={'text':'rule','source_ids':['D1-Vv-C1']},scope=[],conditions=[],exceptions=[],requirements=[],
              period={'text':'תחילה ביום 1 בספטמבר 2021','source_ids':['D1-Vv-C1','D2-Vv-C1'],
                      'start_date':'2021-09-01','end_date':None})
    evidence=[{'id':'D1-Vv-C1','content':'rule'}, {'id':'D2-Vv-C1','content':'תחילה ביום 1 בספטמבר 2021'}]
    units,gaps=bind_units({'units':[unit],'missing':[]}, evidence)
    assert not units[0]['period_known'] and gaps
