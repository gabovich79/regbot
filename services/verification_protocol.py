"""Exact check keys prevent model-added or missing verifier indices."""

def obj(properties):
    return {'type':'object','properties':properties,'required':list(properties),'additionalProperties':False}

def verification_schema(claims, units, issues):
    boolean={'type':'boolean'}
    claim=obj({k:boolean for k in ('supported','scope_preserved','period_consistent','qualifications_preserved')})
    issue=obj({'status':{'type':'string','enum':['covered','missing','conflict']},
               'claim_indices':{'type':'array','items':{'type':'integer'}}})
    strings={'type':'array','items':{'type':'string'}}
    return obj({'claims':obj({str(c['index']):claim for c in claims}),
                'units':obj({u['id']:boolean for u in units}),
                'components':obj({c['id']:boolean for u in units for c in u['components']}),
                'issues':obj({str(i):issue for i in range(len(issues))}),
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
