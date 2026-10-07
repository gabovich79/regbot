from services.date_provenance import date_present
from services.evidence_contract import bind_units


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
