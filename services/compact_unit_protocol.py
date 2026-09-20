"""EXPERIMENTAL shared-component protocol; not used by the runtime pipeline.

Live probes failed pointer/completeness validation. Keep it isolated until it
passes source coverage and complete-answer checks; tests only prove decoding.
"""
from copy import deepcopy
import re

from services.evidence_contract import FIELDS, MAX_COMPONENTS, MAX_UNITS, UNIT_SEMANTICS

TASK = (
    'Extract evidence units in Hebrew from original evidence, NOT an answer. '
    'Return parts [{id,text,source_ids}] and units [{rule,scope,conditions,exceptions,requirements,period}], missing. '
    'Give each distinct component a unique id such as P1. Unit.rule is a part id; '
    'scope/conditions/exceptions/requirements are arrays of part ids, including explicit empty arrays. '
    'Reuse a part ONLY when its complete wording AND source IDs AND meaning are identical. '
    'A shared part applies only to units that explicitly reference it. Never assume global applicability. '
    'Period remains an inline {text,source_ids,start_date,end_date} object or null. '
    'Every part must be referenced; never leave discovered required details outside units. '
    'ALL entries in unit.scope/conditions/exceptions/requirements MUST be P-number IDs, NEVER Hebrew text. '
    'Put qualification text into parts with its own supporting source_ids BEFORE referencing it. '
    + UNIT_SEMANTICS
)


def schema():
    pointer = {'type':'string','pattern':r'^P[1-9][0-9]*$'}
    part = {'type':'object','properties':{
        'id':pointer,'text':{'type':'string'},
        'source_ids':{'type':'array','items':{'type':'string'}}},
        'required':['id','text','source_ids'],'additionalProperties':False}
    period = {'type':'object','properties':{
        'text':{'type':'string'},'source_ids':{'type':'array','items':{'type':'string'}},
        'start_date':{'type':['string','null']},'end_date':{'type':['string','null']}},
        'required':['text','source_ids','start_date','end_date'],'additionalProperties':False}
    unit = {'type':'object','properties':{
        'rule':pointer,
        **{field:{'type':'array','items':pointer} for field in FIELDS},
        'period':{'anyOf':[period,{'type':'null'}]}},
        'required':['rule',*FIELDS,'period'],'additionalProperties':False}
    return {'type':'object','properties':{
        'parts':{'type':'array','items':part},'units':{'type':'array','items':unit},
        'missing':{'type':'array','items':{'type':'string'}}},
        'required':['parts','units','missing'],'additionalProperties':False}


def expand(payload):
    """Reject dangling, duplicate or orphan parts rather than losing details.

    The existing bind_units still enforces source IDs, text and total expanded
    component budgets. Sharing does not create a route around those bounds.
    """
    if not isinstance(payload,dict) or not isinstance(payload.get('parts'),list) or not isinstance(payload.get('units'),list):
        raise ValueError('Invalid compact evidence payload')
    if len(payload['parts']) > MAX_COMPONENTS or len(payload['units']) > MAX_UNITS:
        raise ValueError('Compact evidence budget exceeded')
    parts, used = {}, set()
    for part in payload['parts']:
        if not isinstance(part,dict) or not isinstance(part.get('id'),str) or not re.fullmatch(r'P[1-9][0-9]*',part['id']) or part['id'] in parts:
            raise ValueError('Invalid or duplicate component id')
        parts[part['id']]={key:deepcopy(value) for key,value in part.items() if key!='id'}
    def resolve(ident):
        if not isinstance(ident,str) or ident not in parts:
            raise ValueError('Unknown component id')
        used.add(ident)
        return deepcopy(parts[ident])
    units=[]
    for unit in payload['units']:
        if not isinstance(unit,dict) or 'period' not in unit:
            raise ValueError('Invalid compact unit')
        result={'rule':resolve(unit.get('rule')),'period':deepcopy(unit['period'])}
        for field in FIELDS:
            if not isinstance(unit.get(field),list):
                raise ValueError('Missing compact qualification list')
            result[field]=[resolve(ident) for ident in unit[field]]
        units.append(result)
    if used != set(parts):
        raise ValueError('Unreferenced evidence components')
    return {'units':units,'missing':deepcopy(payload.get('missing'))}
