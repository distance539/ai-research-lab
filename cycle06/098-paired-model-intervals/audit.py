"""Independent standard-library audit: no import of run/scoring/numpy/scipy."""
import argparse, collections, hashlib, json, math, pathlib
ROOT=pathlib.Path(__file__).resolve().parent
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def audit(predictions,summary):
    rr=json.loads(predictions.read_text());s=json.loads(summary.read_text());maps=[]
    def pairs(items):
        assert len(items)==len(dict(items)), 'duplicate JSON key'
        return dict(items)
    for arm in ['shuffle93','shuffle94']:
        rows=[r for r in rr if r['stage']==arm];assert len(rows)==len({r['idx'] for r in rows})
        maps.append({r['idx']:r for r in rows})
        for r in rows:
            try:o=json.loads(r['output'],object_pairs_hook=pairs);valid=isinstance(o,dict) and set(o)=={'sentiment'} and o['sentiment'] in ['negative','positive']
            except (ValueError,AssertionError):valid=False;o={}
            correct=valid and o['sentiment']==['negative','positive'][r['gold']]
            assert r['format']==valid and r['joint']==correct
    a,b=maps;assert a.keys()==b.keys() and len(a)==s['n']
    assert all(a[i]['gold']==b[i]['gold'] and a[i]['prompt_sha256']==b[i]['prompt_sha256'] for i in a)
    ds=[int(b[i]['joint'])-int(a[i]['joint']) for i in sorted(a)];n=len(ds)
    assert sum(ds)/n==s['difference']
    assert sum(a[i]['joint'] for i in a)==s['A_correct'] and sum(b[i]['joint'] for i in b)==s['B_correct']
    cells=collections.Counter(f"A{int(a[i]['joint'])}B{int(b[i]['joint'])}" for i in a)
    assert all(s['cells'][key]==cells[key] for key in s['cells'])
    win=ds.count(1);loss=ds.count(-1);m=win+loss
    exact=min(1.,2*sum(math.comb(m,k) for k in range(min(win,loss)+1))/2**m) if m else 1.
    assert exact==s['exact_discordant_binomial_p']
    # Analytic empirical bootstrap for sparse one-direction discordance: K~Binomial(n,m/n).
    analytic=None
    if m and (win==0 or loss==0):
        pmf=[math.comb(n,k)*(m/n)**k*(1-m/n)**(n-k) for k in range(n+1)]
        points=sorted(((k/n)*(1 if win else -1),v) for k,v in enumerate(pmf))
        def quantile(q):
            cumulative=0
            for value,prob in points:
                cumulative+=prob
                if cumulative>=q:return value
        analytic=[quantile(.025),quantile(.975)]
        assert analytic==s['paired_percentile_interval'], 'bootstrap differs from analytic quantiles'
    assert digest(predictions)==s['input_sha256']
    return {'status':'passed','n':n,'input_sha256':digest(predictions),'summary_sha256':digest(summary),'correct':[s['A_correct'],s['B_correct']],'independent_exact_p':exact,'analytic_empirical_bootstrap_interval':analytic,'human_semantic_review':False}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--predictions',type=pathlib.Path,default=ROOT/'cached_predictions.json');p.add_argument('--summary',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,required=True);a=p.parse_args();r=audit(a.predictions,a.summary);a.out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
