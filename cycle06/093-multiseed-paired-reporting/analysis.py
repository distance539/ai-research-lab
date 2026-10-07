"""Independent descriptive pairing; no pooled seed/sample inference or selection."""
import json,pathlib,argparse
import numpy as np

def analyze(rows,c):
    result={};groups={}
    for contract in ['json','plain','letter']:
        base=sorted([r for r in rows if r['stage']=='before' and r['contract']==contract],key=lambda r:r['idx'])
        ids=[r['idx'] for r in base];n=len(ids);rng=np.random.default_rng(c['bootstrap_seed'])
        if contract=='letter':
            inds=[np.where(np.array([r['gold'] for r in base])==g)[0] for g in range(4)]
            boot=np.concatenate([rng.choice(ii,size=(c['bootstrap_replicates'],len(ii)),replace=True) for ii in inds],axis=1)
        else:boot=rng.integers(0,n,size=(c['bootstrap_replicates'],n))
        result[contract]={}
        for metric in ['format','content_proxy','joint']:
            out={};b=np.array([r[metric] for r in base],dtype=int)
            for arm in c['arms']:
                rr=sorted([r for r in rows if r['stage']==arm and r['contract']==contract],key=lambda r:r['idx']);assert [r['idx'] for r in rr]==ids
                a=np.array([r[metric] for r in rr],dtype=int);d=a-b;reps=d[boot].mean(axis=1)
                out[arm]={'n':n,'before_count':int(b.sum()),'after_count':int(a.sum()),'delta':float(d.mean()),'win':int((d>0).sum()),'tie':int((d==0).sum()),'loss':int((d<0).sum()),'conditional_paired_percentile95':np.quantile(reps,[.025,.975],method='linear').tolist()}
            vals=np.array([out[f'shuffle{s}']['delta'] for s in [93,94,95]])
            result[contract][metric]={'arms':out,'three_shuffle_seeds':{'mean_delta':float(vals.mean()),'sample_sd_delta':float(vals.std(ddof=1)),'min_delta':float(vals.min()),'max_delta':float(vals.max()),'seeds':[93,94,95]}}
    return {'bootstrap_replicates':c['bootstrap_replicates'],'bootstrap_seed':c['bootstrap_seed'],'interval_scope':'conditional fixed checkpoint, descriptive development sensitivity, iid SST rows; class-stratified news; no selection correction','results':result}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=pathlib.Path,required=True);a=p.parse_args();c=json.loads((pathlib.Path(__file__).parent/'config.json').read_text());rows=[json.loads(x) for x in (a.out/'predictions.jsonl').read_text().splitlines()];(a.out/'statistics.json').write_text(json.dumps(analyze(rows,c),indent=2)+'\n')
