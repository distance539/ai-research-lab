"""Official SciPy paired bootstrap, with a direct shared-index cross-check."""
import argparse,time,platform
from pathlib import Path
from common import ROOT,load,save,digest

def analyze(rows,selection,cfg):
    import numpy as np
    from scipy.stats import bootstrap
    a={r['idx']:r for r in rows if r['condition']==selection['baseline']};b={r['idx']:r for r in rows if r['condition']==selection['winner']}
    assert a.keys()==b.keys() and set(a)==set(selection['confirmation_ids'])
    ids=sorted(a);x=np.array([a[i]['correct'] for i in ids]);y=np.array([b[i]['correct'] for i in ids]);n=len(ids)
    stat=lambda u,v,axis=-1:np.mean(v-u,axis=axis)
    result=bootstrap((x,y),stat,paired=True,vectorized=True,method='percentile',n_resamples=cfg['resamples'],random_state=np.random.default_rng(cfg['bootstrap_seed']))
    ix=np.random.default_rng(cfg['bootstrap_seed']).integers(0,n,size=(cfg['resamples'],n));direct=(y[ix]-x[ix]).mean(axis=1)
    assert np.array_equal(result.bootstrap_distribution,direct)
    wins=[i for i in ids if b[i]['correct']>a[i]['correct']];losses=[i for i in ids if b[i]['correct']<a[i]['correct']]
    values,counts=np.unique(direct,return_counts=True)
    return {'n':n,'baseline':selection['baseline'],'winner':selection['winner'],'baseline_correct':int(x.sum()),'winner_correct':int(y.sum()),'delta':float((y-x).mean()),'gains':wins,'losses':losses,'ties':n-len(wins)-len(losses),'paired_percentile_95':[float(result.confidence_interval.low),float(result.confidence_interval.high)],'official_direct_all_resamples_equal':True,'resamples':cfg['resamples'],'seed':cfg['bootstrap_seed'],'histogram':[{'delta':float(v),'count':int(c)} for v,c in zip(values,counts)],'constant_positive_correct':sum(a[i]['gold']==1 for i in ids),'interpretation':'fixed selected pipeline, row-conditional; not a search-wide interval or bias decomposition'}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--records',type=Path,default=ROOT/'confirmation/records.json');p.add_argument('--out',type=Path,default=ROOT/'confirmation/analysis.json');a=p.parse_args();t=time.perf_counter()
    result=analyze(load(a.records),load(ROOT/'selection.json'),load(ROOT/'config.json'));result['input_sha256']=digest(a.records);save(a.out,result);print(result)
