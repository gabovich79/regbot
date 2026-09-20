"""Require an explicit relevance judgment for every original candidate."""


def ranking_schema(ids):
    item={'type':'object','properties':{
        'id':{'type':'string','enum':list(ids)},
        'score':{'type':'integer','enum':[0,1,2,3]},
        'reason':{'type':'string'}},'required':['id','score','reason'],'additionalProperties':False}
    return {'type':'object','properties':{'ratings':{'type':'array','items':item}},
            'required':['ratings'],'additionalProperties':False}


def ranked_ids(raw, lookup, limit=20):
    ratings=raw.get('ratings') if isinstance(raw,dict) else None
    if not isinstance(ratings,list) or len(ratings)!=len(lookup):
        raise ValueError('Incomplete candidate ratings')
    seen=set()
    for row in ratings:
        if (not isinstance(row,dict) or not isinstance(row.get('id'),str) or
                row['id'] not in lookup or row['id'] in seen or
                type(row.get('score')) is not int or row['score'] not in (0,1,2,3) or
                not isinstance(row.get('reason'),str) or not row['reason'].strip()):
            raise ValueError('Invalid candidate rating')
        seen.add(row['id'])
    positions={key:i for i,key in enumerate(lookup)}
    # The provider's response order is not a ranking. Stable ties only use the
    # prior retrieval order; no topic-specific bonuses or source-label boosts.
    return [r['id'] for r in sorted(ratings,key=lambda r:(-r['score'],positions[r['id']]))
            if r['score']>=2][:limit]
