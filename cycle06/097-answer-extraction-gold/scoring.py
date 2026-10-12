"""Frozen automatic rules, not human semantic scoring."""
import json,re
LABELS=('negative','positive')
def unique_object(pairs):
    if len({k for k,v in pairs})!=len(pairs):raise ValueError('duplicate key')
    return dict(pairs)
def score(text,gold):
    canonical=json.dumps({'sentiment':LABELS[gold]})
    try:
        obj=json.loads(text,object_pairs_hook=unique_object)
        valid=isinstance(obj,dict) and set(obj)=={'sentiment'} and obj['sentiment'] in LABELS
    except (ValueError,TypeError):valid=False;obj=None
    hits=re.findall(r'(?i)\b(positive|negative)\b',text)
    first=hits[0].lower() if hits else None
    unique=set(x.lower() for x in hits)
    return {'exact':text.strip()==canonical,'json_format':bool(valid),'json_correct':bool(valid and obj['sentiment']==LABELS[gold]),'first_label':first,'regex_correct':first==LABELS[gold], 'unique_label':next(iter(unique)) if len(unique)==1 else None,'hit_count':len(hits),'distinct_labels':sorted(unique)}
def aggregate(rows):
    keys=['exact','json_format','json_correct','regex_correct']
    out={'n':len(rows),'counts':{k:sum(r['scores'][k] for r in rows) for k in keys},'paired':{'regex_gain':[r['idx'] for r in rows if r['scores']['regex_correct'] and not r['scores']['exact']],'regex_loss':[r['idx'] for r in rows if r['scores']['exact'] and not r['scores']['regex_correct']]},'length_stops':sum(r['stop']=='length' for r in rows),'tokens':{'input':sum(r['input_length'] for r in rows),'generated_including_eos':sum(r['generated_length'] for r in rows)},'human':{'status':'pending','n_completed':0,'error_rates':None},'confirmation_inference':False}
    return out
