"""Compact request-local source IDs, resolved back before evidence is bound."""
from copy import deepcopy


def compact_sources(evidence):
    lookup={f'S{i+1}':e['id'] for i,e in enumerate(evidence)}
    return [dict(e,id=alias) for alias,e in zip(lookup,evidence)],lookup


def restore_source_ids(payload, lookup):
    """Only source_ids are translated; source text is never rewritten."""
    result=deepcopy(payload)
    lookup={**{v:v for v in lookup.values()},**lookup}
    def visit(value):
        if isinstance(value,dict):
            for key,child in value.items():
                if key=='source_ids':
                    if not isinstance(child,list) or any(not isinstance(i,str) or i not in lookup for i in child):
                        raise ValueError('Unknown request-local source pointer')
                    value[key]=[lookup[i] for i in child]
                else:visit(child)
        elif isinstance(value,list):
            for child in value:visit(child)
    visit(result)
    return result


def coverage_schema(source_ids):
    aspect={'type':'object','properties':{'issue':{'type':'string','maxLength':240},
            'source_ids':{'type':'array','maxItems':3,'items':{'type':'string','enum':list(source_ids) or ['NO_SOURCE']}}},
            'required':['issue','source_ids'],'additionalProperties':False}
    messages={'type':'array','maxItems':12,'items':{'type':'string','maxLength':240}}
    return {'type':'object','properties':{
        'covered':{'type':'array','maxItems':20,'items':aspect},
        'aspects':{'type':'array','maxItems':18,'items':aspect},
        'missing':messages,'conflicts':messages,'needs_web':{'type':'boolean'}},
        'required':['covered','aspects','missing','conflicts','needs_web'],'additionalProperties':False}


def extraction_schema(source_ids):
    part={'type':'object','properties':{'text':{'type':'string','maxLength':1800},
          'source_ids':{'type':'array','minItems':1,'maxItems':4,'items':{'type':'string','enum':list(source_ids) or ['NO_SOURCE']}}},
          'required':['text','source_ids'],'additionalProperties':False}
    parts={'type':'array','maxItems':12,'items':part}
    unit={'type':'object','properties':{'rule':part,'scope':parts,'conditions':parts,'exceptions':parts,
          'period':{'anyOf':[part,{'type':'null'}]}},
          'required':['rule','scope','conditions','exceptions','period'],'additionalProperties':False}
    return {'type':'object','properties':{'units':{'type':'array','maxItems':20,'items':unit},
            'missing':{'type':'array','items':{'type':'string'}}},'required':['units','missing'],'additionalProperties':False}
