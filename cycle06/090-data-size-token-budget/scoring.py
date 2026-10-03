"""Independent teaching evaluator: explicit label proxy is NOT semantic human scoring."""
import json, re
LABELS=('negative','positive')

def unique_object(pairs):
    if len({k for k,v in pairs}) != len(pairs): raise ValueError('duplicate key')
    return dict(pairs)

def score(text, contract, gold):
    labels=set(re.findall(r'\b(?:positive|negative)\b',text.lower()))
    extracted=next(iter(labels)) if len(labels)==1 else None
    fmt=False
    if contract=='plain': fmt=text.strip() in LABELS
    elif contract=='json':
        try:
            obj=json.loads(text,object_pairs_hook=unique_object)
            fmt=isinstance(obj,dict) and set(obj)=={'sentiment'} and obj['sentiment'] in LABELS
        except (ValueError,TypeError): pass
    else: raise ValueError(contract)
    correct=extracted==LABELS[gold]
    return {'extracted':extracted,'label_state':'unique' if len(labels)==1 else ('missing' if not labels else 'ambiguous'),
            'format':bool(fmt),'content_proxy':correct,'joint':bool(fmt and correct),'bucket':f'F{int(fmt)}C{int(correct)}'}

def aggregate(rows):
    result={}
    for c in ('plain','json'):
        rr=[r for r in rows if r['contract']==c];n=len(rr);f=sum(r['format'] for r in rr);j=sum(r['joint'] for r in rr)
        result[c]={'n':n,'format_count':f,'correct_count':sum(r['content_proxy'] for r in rr),'joint_count':j,
            'format_rate':f/n,'content_proxy_accuracy':sum(r['content_proxy'] for r in rr)/n,'joint_accuracy':j/n,
            'correct_given_format':j/f if f else None,'conditional_denominator':f,
            'buckets':{b:sum(r['bucket']==b for r in rr) for b in ('F0C0','F0C1','F1C0','F1C1')},
            'label_states':{s:sum(r['label_state']==s for r in rr) for s in ('unique','missing','ambiguous')},
            'per_class':{LABELS[g]:{'n':sum(r['gold']==g for r in rr),'correct':sum(r['content_proxy'] and r['gold']==g for r in rr)} for g in (0,1)},
            'input_tokens':sum(r['input_length'] for r in rr),'generated_tokens_including_eos':sum(len(r['generated_ids']) for r in rr),
            'limit_stops':sum(r['stop']=='length' for r in rr)}
    a={r['idx']:r for r in rows if r['contract']=='plain'};b={r['idx']:r for r in rows if r['contract']=='json'}
    result['paired']={key:{'json_gain':sum(b[i][key]>a[i][key] for i in a),'tie':sum(b[i][key]==a[i][key] for i in a),'json_loss':sum(b[i][key]<a[i][key] for i in a)} for key in ('format','content_proxy','joint')}
    result['constant_baselines']={LABELS[g]:sum(r['gold']==g for r in a.values())/len(a) for g in (0,1)}
    return result
