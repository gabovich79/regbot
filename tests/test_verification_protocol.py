from services.verification_protocol import verification_schema,decode_verification
from services.evidence_pipeline import verification_result

def test_schema_keys_match_actual_indices_not_positions():
    schema=verification_schema([{'index':3}],[],['issue'])
    assert schema['properties']['claims']['required']==['3']
    assert schema['properties']['issues']['required']==['0']
    assert schema['properties']['issues']['additionalProperties'] is False

def test_decoding_never_invents_approval_for_missing_or_extra_checks():
    raw={'claims':{'3':{'supported':True}},'units':{},'components':{},
         'issues':{'0':{'status':'covered','claim_indices':[3]}},'missing':[],'conflicts':[]}
    checked=decode_verification(raw)
    assert verification_result(checked,[{'index':3}],['a'])[3]
    raw['issues']['1']={'status':'covered','claim_indices':[3]}
    assert not verification_result(decode_verification(raw),[{'index':3}],['a'])[3]
    assert decode_verification({'claims':{}})=={}
