"""Repeat hash partitions without replacement; no selection using either evaluation set."""
from pathlib import Path
import hashlib, json, math, statistics
from fractions import Fraction
from collections import Counter
P=Path(__file__).resolve().parent
def read(n): return json.loads((P/n).read_bytes())
def save(n,x): (P/n).write_text(json.dumps(x,separators=(',',':'))+'\n')
def key(seed,i): return hashlib.sha256(f'{seed}:{i}'.encode()).digest(),i
def scores(rows):
    out={}
    for r in rows:
        z=r['logits']; w=[math.exp(x-max(z)) for x in z]
        out[r['idx']]=1-w[r['label']]/sum(w)
    return out
def main():
    cfg=read('protocol.json'); a=Fraction(cfg['alpha']); n=cfg['calibration']; k=math.ceil((n+1)*(1-a))
    c=scores(read('calibration_predictions.json')); t=scores(read('run_predictions.json'))
    m=read('split_manifest.json')
    assert list(c)==m['calibration'] and list(t)==m['protocol_audit'] and not set(c)&set(t)
    assert len(c)==2*n==256 and len(t)==360 and k==cfg['fixed_rank']
    rows=[]; by={i:[0,0] for i in c}
    for s in range(cfg['seed_first'],cfg['seed_first']+cfg['repeats']):
        ids=sorted(c,key=lambda i:key(s,i)); fit,ev=ids[:n],ids[n:]
        q=sorted(c[i] for i in fit)[k-1]; missed=[i for i in ev if c[i]>q]
        for i in ev:by[i][0]+=1;by[i][1]+=int(c[i]<=q)
        rows.append(dict(seed=s,fit=fit,test=ev,q=q,missed=missed,hits=n-len(missed),audit_missed=[i for i in t if t[i]>q]))
    def stats(hits,den):
        return dict(den=den,mean=sum(hits)/(len(hits)*den),min=min(hits),max=max(hits),below=sum(x<den*(1-a) for x in hits),counts=dict(sorted(Counter(hits).items())))
    out=dict(held=stats([x['hits'] for x in rows],n),fixed=stats([len(t)-len(x['audit_missed']) for x in rows],len(t)),rank=k,always=sum(h==m for m,h in by.values()),never=sum(h==0 for m,h in by.values()),distinct_scores=len(set(c.values())))
    save('splits.json',rows);save('by_id.json',by);save('results.json',out)
    inputs='config.json protocol.json resources.lock.json split_manifest.json baseline.py model_run.py run.py audit.py calibration_predictions.json run_predictions.json audit_predictions.json'.split()
    save('lock.json',{f:hashlib.sha256((P/f).read_bytes()).hexdigest() for f in inputs})
    print(json.dumps(out))
if __name__=='__main__':main()
