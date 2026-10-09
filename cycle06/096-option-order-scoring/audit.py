"""Independent arithmetic replay. No imports from the inference implementation."""
import argparse, hashlib, json, math
from pathlib import Path

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def check(folder, root):
    records=json.loads((folder/'records.json').read_text())
    identity=json.loads((folder/'identity.json').read_text())
    plan=json.loads((root/'plan.json').read_text())
    assert identity['plan_sha256']==digest(root/'plan.json')
    assert identity['protocol_sha256']==digest(root/'PROTOCOL.md')
    assert not set(identity['selected'])&set(plan['reserved'])
    expected={(n,i) for n in plan['conditions'] for i in identity['selected']}
    assert {(r['condition'],r['idx']) for r in records}==expected and len(records)==len(expected)
    counts={n:{m:0 for m in ['sum','token_mean','character_mean']} for n in plan['conditions']}
    lookup={};max_error=0
    for r in records:
        name=r['condition'];expected_map=[1,0] if name.endswith('reverse') else [0,1]
        assert r['mapping']==expected_map and r['gold'] in [0,1]
        assert r['position_gold']==expected_map.index(r['gold'])
        names=list('AB') if 'letter' in name else [['negative','positive'][i] for i in expected_map]
        vals={m:[] for m in counts[name]}
        for j,c in enumerate(r['candidates']):
            assert c['choice']==names[j] and len(c['continuation_ids'])==len(c['token_logprobs'])>0
            assert all(math.isfinite(x) and x<=0 for x in c['token_logprobs'])
            summed=math.fsum(c['token_logprobs']);assert abs(summed-c['score'])<1e-7
            error=abs(summed-r['official_scores'][j]);assert error<=2e-5;max_error=max(max_error,error)
            assert c['input_shape']==[1,c['input_tokens']]
            assert c['logits_shape']==[1,c['input_tokens'],49152]
            assert c['input_tokens']==r['context_tokens']+len(c['continuation_ids'])-1
            vals['sum'].append(summed);vals['token_mean'].append(summed/len(c['continuation_ids']));vals['character_mean'].append(summed/len(c['choice']))
        for m,v in vals.items():
            pos=0 if v[0]>=v[1] else 1;semantic=expected_map[pos]
            got=r['predictions'][m]
            assert got['position']==pos and got['semantic']==semantic
            assert all(abs(a-b)<1e-7 for a,b in zip(v,got['scores']))
            counts[name][m]+=int(semantic==r['gold'])
        assert r['official_metrics']['acc']==int(r['predictions']['sum']['semantic']==r['gold'])
        assert r['official_metrics']['acc_norm']==int(r['predictions']['character_mean']['semantic']==r['gold'])
        lookup[name,r['idx']]=r
    summary=json.loads((folder/'summary.json').read_text())
    for name,count in counts.items():
        assert summary['conditions'][name]['correct']==count
        rows=[r for r in records if r['condition']==name]
        d=summary['conditions'][name]
        assert d['n']==len(rows)
        assert d['positive']==sum(r['predictions']['sum']['semantic'] for r in rows)
        assert d['first_position']==sum(r['predictions']['sum']['position']==0 for r in rows)
        assert d['candidate_input_tokens']==sum(c['input_tokens'] for r in rows for c in r['candidates'])
    for pair,entry in summary['pairs'].items():
        b,a=pair.split(' minus ');gains=[];losses=[];flips=[]
        for i in identity['selected']:
            x=lookup[a,i];y=lookup[b,i];assert x['gold']==y['gold'] and x['sentence_sha256']==y['sentence_sha256']
            xp=x['predictions']['sum']['semantic'];yp=y['predictions']['sum']['semantic']
            if xp!=yp:flips.append(i)
            if xp!=x['gold'] and yp==y['gold']:gains.append(i)
            if xp==x['gold'] and yp!=y['gold']:losses.append(i)
        assert entry=={'gains':gains,'losses':losses,'flips':flips,'delta':(len(gains)-len(losses))/len(identity['selected'])}
    for i in identity['selected']:
        for order in ['forward','reverse']:
            a=lookup['listed_text_'+order,i];b=lookup['listed_letter_'+order,i]
            assert a['context_sha256']==b['context_sha256'] and a['context_tokens']==b['context_tokens']
    assert summary['candidate_scores']==2*len(records)
    return {'status':'passed','rows':len(records),'questions':len(identity['selected']),'candidate_scores':len(records)*2,'counts':counts,'max_official_error':max_error,'input_sha256':{n:digest(folder/n) for n in ['records.json','summary.json','identity.json']},'script_sha256':digest(Path(__file__)),'human_review':False}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args()
    result=check(a.folder,Path(__file__).resolve().parent)
    (a.folder/'independent_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
