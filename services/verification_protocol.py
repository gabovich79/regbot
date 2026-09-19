"""Restrict check IDs; application validation enforces complete unique sets."""

def obj(properties):
    return {'type':'object','properties':properties,'required':list(properties),'additionalProperties':False}

def verification_schema(claims, units, issues):
    boolean={'type':'boolean'}
    claim_ids=[c['index'] for c in claims]
    claim_index={'type':'integer','enum':claim_ids or [-1]}
    claim=obj({'index':claim_index,**{k:boolean for k in ('supported','scope_preserved','period_consistent','qualifications_preserved')}})
    issue=obj({'issue_index':{'type':'integer','enum':list(range(len(issues))) or [-1]},
               'status':{'type':'string','enum':['covered','missing','conflict']},
               'claim_indices':{'type':'array','items':claim_index}})
    strings={'type':'array','items':{'type':'string'}}
    return obj({'checks':{'type':'array','items':claim},
                'unit_checks':{'type':'array','items':obj({'unit_id':{'type':'string','enum':[u['id'] for u in units] or ['NONE']},'complete':boolean})},
                'component_checks':{'type':'array','items':obj({'component_id':{'type':'string','enum':[c['id'] for u in units for c in u['components']] or ['NONE']},'supported':boolean})},
                'issue_checks':{'type':'array','items':issue},
                'missing':strings,'conflicts':strings})

def decode_verification(raw):
    if not isinstance(raw,dict):return {}
    if 'checks' in raw:return raw
    try:
        if any(not isinstance(raw[k],dict) for k in ('claims','units','components','issues')):return {}
        if any(not isinstance(k,str) or not k.isdigit() for k in list(raw['claims'])+list(raw['issues'])):return {}
        return {'checks':[dict(v,index=int(k)) for k,v in raw['claims'].items()],
                'unit_checks':[{'unit_id':k,'complete':v} for k,v in raw['units'].items()],
                'component_checks':[{'component_id':k,'supported':v} for k,v in raw['components'].items()],
                'issue_checks':[dict(v,issue_index=int(k)) for k,v in raw['issues'].items()],
                'missing':raw['missing'],'conflicts':raw['conflicts']}
    except (KeyError,TypeError,ValueError):return {}
