"""Independent standard-library arithmetic audit; no model/scoring/stat imports."""
import argparse,hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def check_rows(rows,names,ids):
    groups={n:{} for n in names}
    for r in rows:
        n=r['condition'];i=r['idx'];assert n in groups and i in ids and i not in groups[n]
        scores=[]
        assert len(r['candidates'])==2 and sorted(r['mapping'])==[0,1] and r['gold'] in [0,1]
        for c,official in zip(r['candidates'],r['official_scores']):
            assert len(c['continuation_ids'])==len(c['token_logprobs'])>0
            score=math.fsum(c['token_logprobs']);assert math.isfinite(score) and abs(score-c['score'])<1e-6 and abs(score-official)<=2e-5;scores.append(score)
        pos=0 if scores[0]>=scores[1] else 1;pred=r['mapping'][pos];correct=int(pred==r['gold'])
        assert pred==r['prediction'] and correct==r['correct']==r['official_accuracy']
        assert len(r['sentence_sha256'])==len(r['context_sha256'])==64
        groups[n][i]=r
    assert all(set(g)==set(ids) for g in groups.values())
    for i in ids:
        rs=[g[i] for g in groups.values()];assert len({(r['gold'],r['sentence_sha256']) for r in rs})==1
    return groups

def exact_interval(w,l,n):
    # Exact empirical bootstrap law: n independent draws from {-1,0,1}.
    pmf={0:1.0};probs={-1:l/n,0:1-(w+l)/n,1:w/n}
    for _ in range(n):
        new={}
        for s,p in pmf.items():
            for step,q in probs.items():new[s+step]=new.get(s+step,0)+p*q
        pmf=new
    assert abs(sum(pmf.values())-1)<1e-12
    result=[]
    for quantile in [.025,.975]:
        total=0
        for s,p in sorted(pmf.items()):
            total+=p
            if total>=quantile:result.append(s/n);break
    return result

def audit(root):
    cfg=read(root/'config.json');plan=read(root/'plan.json');sel=read(root/'selection.json')
    dev=check_rows(read(root/'development/records.json'),cfg['conditions'],plan['development'])
    counts={n:sum(r['correct'] for r in dev[n].values()) for n in cfg['conditions']}
    winner=sorted(cfg['conditions'],key=lambda n:(-counts[n],cfg['conditions'].index(n)))[0]
    assert winner==sel['winner'] and all(sel['all_attempts'][n]['correct']==counts[n] for n in counts)
    assert sha(root/'development/records.json')==sel['development_sha256']
    assert not set(plan['development'])&set(plan['reserved']) and len(plan['reserved'])==128
    for f,h in sel['frozen_inputs'].items():assert sha(root/f)==h
    names=list(dict.fromkeys([sel['baseline'],winner]));test=check_rows(read(root/'confirmation/records.json'),names,plan['reserved']);ident=read(root/'confirmation/identity.json')
    assert ident['conditions']==names and ident['selected']==plan['reserved'] and ident['started_at']>sel['frozen_at']
    assert ident['selection_sha256']==sha(root/'selection.json') and ident['frozen_inputs']==sel['frozen_inputs']
    x=test[sel['baseline']];y=test[winner];g=[i for i in sorted(x) if y[i]['correct']>x[i]['correct']];l=[i for i in sorted(x) if y[i]['correct']<x[i]['correct']]
    n=len(x);a=read(root/'confirmation/analysis.json');ci=exact_interval(len(g),len(l),n)
    assert a['input_sha256']==sha(root/'confirmation/records.json') and a['gains']==g and a['losses']==l
    assert a['baseline_correct']==sum(r['correct'] for r in x.values()) and a['winner_correct']==sum(r['correct'] for r in y.values())
    assert a['delta']==(len(g)-len(l))/n and sum(v['count'] for v in a['histogram'])==cfg['resamples']
    assert all(abs(v-q)<=1/n for v,q in zip(ci,a['paired_percentile_95']))
    return {'passed':True,'development_counts':counts,'winner':winner,'n':n,'gains':len(g),'losses':len(l),'delta':a['delta'],'exact_empirical_bootstrap_95':ci,'numerical_interval_tolerance':1/n,'input_hashes':{str(p.relative_to(root)):sha(p) for p in [root/'development/records.json',root/'selection.json',root/'confirmation/records.json',root/'confirmation/analysis.json']},'no_human_semantic_judgment':True}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--out',type=Path);a=p.parse_args();r=audit(a.root)
    if a.out:a.out.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r,indent=2))
