"""098: replay paired statistics, or regenerate two real checkpoint outputs."""
import argparse, gc, hashlib, json, pathlib, platform, resource, sys, time
from prepare import ROOT, digest, prepare
from common import message, sha, save
from scoring import score

def load_rows(path):
    rows=json.loads(path.read_text()); cfg=json.loads((ROOT/'config.json').read_text())
    maps=[]
    for arm in (cfg['A'],cfg['B']):
        subset=[r for r in rows if r['stage']==arm]
        assert len(subset)==len({r['idx'] for r in subset}), 'duplicate ID'
        maps.append({r['idx']:r for r in subset})
    a,b=maps; assert a and a.keys()==b.keys(), 'unequal paired IDs'
    for i in a:
        assert a[i]['gold']==b[i]['gold'] and a[i]['prompt_sha256']==b[i]['prompt_sha256'], 'unaligned identity'
        for r in (a[i],b[i]):
            assert r['contract']=='json' and r['task']=='target'
            assert all(r[k]==v for k,v in score(r['output'],'json',r['gold']).items()), 'score mismatch'
    return a,b

def statistics(path,out):
    import numpy as np, scipy
    from scipy.stats import bootstrap, binomtest
    started=time.perf_counter(); c=json.loads((ROOT/'config.json').read_text()); a,b=load_rows(path)
    ids=sorted(a); x=np.array([int(a[i]['joint']) for i in ids]); y=np.array([int(b[i]['joint']) for i in ids]); d=y-x
    def diff(a,b,axis=-1): return np.mean(b,axis=axis)-np.mean(a,axis=axis)
    args=dict(n_resamples=c['bootstrap_replicates'],confidence_level=c['confidence_level'],method=c['method'],vectorized=True)
    paired=bootstrap((x,y),diff,paired=True,random_state=np.random.default_rng(c['bootstrap_seed']),**args)
    unpaired=bootstrap((x,y),diff,paired=False,random_state=np.random.default_rng(c['bootstrap_seed']),**args)
    # Separate direct indexing implementation must exactly reproduce official distribution.
    ix=np.random.default_rng(c['bootstrap_seed']).integers(0,len(d),size=(c['bootstrap_replicates'],len(d)))
    independent=d[ix].mean(axis=1); assert np.array_equal(independent,paired.bootstrap_distribution)
    wins=int((d>0).sum()); losses=int((d<0).sum()); discord=wins+losses
    cells={f'A{i}B{j}':int(((x==i)&(y==j)).sum()) for i in (0,1) for j in (0,1)}
    result={'n':len(ids),'A_correct':int(x.sum()),'B_correct':int(y.sum()),'difference':float(d.mean()),'cells':cells,'B_wins':wins,'B_losses':losses,'ties':int((d==0).sum()),'discordant_ids':[i for i,v in zip(ids,d) if v],
      'paired_percentile_interval':list(map(float,paired.confidence_interval)),
      'unpaired_WRONG_FOR_THIS_DESIGN_interval':list(map(float,unpaired.confidence_interval)),
      'paired_bootstrap_se':float(paired.standard_error),'unpaired_bootstrap_se':float(unpaired.standard_error),
      'exact_discordant_binomial_p':float(binomtest(wins,discord,.5).pvalue) if discord else 1.,
      'paired_distribution_exactly_matches_direct_indexing':True,
      'replicates':c['bootstrap_replicates'],'seed':c['bootstrap_seed'],'method':c['method'],'input_sha256':digest(path),
      'covariance_sample':float(np.cov(x,y,ddof=1)[0,1]),'paired_variance_mean':float(d.var(ddof=1)/len(d)),
      'independent_variance_mean':float((x.var(ddof=1)+y.var(ddof=1))/len(d)),
      'interpretation':'Exploratory row-conditional sensitivity only; not blind confirmation, equivalence, or seed population inference.'}
    save(out/'summary.json',result)
    save(out/'paired_rows.json',[{'idx':i,'gold':a[i]['gold'],'prompt_sha256':a[i]['prompt_sha256'],'A':int(x[k]),'B':int(y[k]),'difference':int(d[k])} for k,i in enumerate(ids)])
    values,counts=np.unique(paired.bootstrap_distribution,return_counts=True)
    save(out/'bootstrap_histogram.json',[{'difference':float(v),'count':int(n)} for v,n in zip(values,counts)])
    save(out/'stat_environment.json',{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'seconds':time.perf_counter()-started})
    print(json.dumps(result),flush=True)

def infer(args):
    import torch, transformers, pyarrow.parquet as pq
    from transformers import AutoTokenizer,AutoModelForCausalLM,GenerationConfig
    started=time.perf_counter();c=json.loads((ROOT/'inference_config.json').read_text());torch.set_num_threads(c['threads']);torch.manual_seed(98)
    lock=prepare(args.cache);cache_rows=json.loads((ROOT/'cached_predictions.json').read_text());ck=json.loads((ROOT/'checkpoint_lock.json').read_text())
    stages={'download':'passed: pinned model/data/source hashes checked','preprocessing':'running','training':'not run; frozen 093 checkpoints reused','inference':'pending','evaluation':'pending'};save(args.out/'stages.json',stages)
    validation=next(k for k in lock['files'] if k.endswith('.parquet') and 'validation-' in k)
    source=pq.read_table(args.cache/validation).to_pylist();byid={r['idx']:r for r in source}
    ids=json.loads((ROOT/'split.json').read_text())['target_dev'];ids=ids[:2] if args.smoke else ids
    reserved=json.loads((ROOT/'split.json').read_text())['target_reserved'];assert not set(ids)&set(reserved)
    md=args.cache/'models'/lock['model']['revision'];tok=AutoTokenizer.from_pretrained(md,local_files_only=True)
    gen=GenerationConfig(**c['decoding'],eos_token_id=tok.eos_token_id,pad_token_id=tok.pad_token_id,bos_token_id=tok.bos_token_id)
    identity={'model':lock['model'],'resources_sha256':digest(ROOT/'resources.lock.json'),'data_file_sha256':digest(args.cache/validation),'config':c,'ids':ids,'checkpoints':ck,'chat_template_sha256':sha(tok.chat_template),'code':{p.name:digest(p) for p in ROOT.glob('*.py')},'stats_config_sha256':digest(ROOT/'config.json'),'protocol_sha256':digest(ROOT/'PROTOCOL.md')}
    identity['cache_key']=sha(json.dumps(identity,sort_keys=True));save(args.out/'identity.json',identity)
    records=[];shape=None;stages['preprocessing']='passed: same source IDs and untouched confirmation set';save(args.out/'stages.json',stages)
    for spec in ck:
        checkpoint=args.checkpoints/spec['filename'];assert digest(checkpoint)==spec['sha256'], 'checkpoint mismatch'
        model=AutoModelForCausalLM.from_pretrained(md,local_files_only=True,torch_dtype=torch.float32,attn_implementation='eager')
        model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True);model.eval()
        with torch.inference_mode():
            for i in ids:
                row=byid[i];text=tok.apply_chat_template(message(c,'json',row['sentence']),tokenize=False,add_generation_prompt=True)
                x=tok(text,add_special_tokens=False,return_tensors='pt');assert x.input_ids.shape[1]<=512
                if shape is None:shape={'input':list(x.input_ids.shape),'logits':list(model(**x).logits.shape)}
                t=time.perf_counter();y=model.generate(**x,generation_config=gen)[0,x.input_ids.shape[1]:].tolist();output=tok.decode(y,skip_special_tokens=True)
                rec={'stage':spec['arm'],'idx':i,'gold':row['label'],'contract':'json','task':'target','prompt_sha256':sha(text),'input_length':x.input_ids.shape[1],'generated_ids':y,'output':output,'stop':'eos' if y[-1]==tok.eos_token_id else 'length','seconds':time.perf_counter()-t,**score(output,'json',row['label'])}
                old=next(r for r in cache_rows if r['stage']==spec['arm'] and r['idx']==i)
                assert all(v==old[k] for k,v in rec.items() if k!='seconds'), 'prediction differs from frozen093'
                records.append(rec);print(json.dumps({'stage':spec['arm'],'idx':i,'joint':rec['joint'],'input_tokens':rec['input_length'],'generated_tokens':len(y),'seconds':rec['seconds']}),flush=True)
        del model;gc.collect()
    save(args.out/'records.json',records);statistics(args.out/'records.json',args.out)
    sec=time.perf_counter()-started;rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
    assert sec<1200 and rss<6442450944
    save(args.out/'environment.json',{'os':platform.platform(),'python':platform.python_version(),'torch':torch.__version__,'transformers':transformers.__version__,'device':'cpu','dtype':'float32','threads':4,'seconds_after_imports':sec,'peak_rss_bytes':rss,'gpu_memory':'not measured; CPU','shape':shape,'outputs':len(records),'all_equal_to_frozen093_except_seconds':True,'input_tokens':sum(r['input_length'] for r in records),'generated_tokens_including_eos':sum(len(r['generated_ids']) for r in records)})
    stages.update(inference='passed: real checkpoint generation',evaluation='passed: exact cached tokens and scores, paired statistics audited',exit_code=0);save(args.out/'stages.json',stages)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=pathlib.Path,required=True);p.add_argument('--infer',action='store_true');p.add_argument('--smoke',action='store_true');p.add_argument('--cache',type=pathlib.Path);p.add_argument('--checkpoints',type=pathlib.Path);p.add_argument('--input',type=pathlib.Path,default=ROOT/'cached_predictions.json');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    if a.infer:
        assert a.cache and a.checkpoints,'--infer requires --cache and --checkpoints';infer(a)
    else:statistics(a.input,a.out)
