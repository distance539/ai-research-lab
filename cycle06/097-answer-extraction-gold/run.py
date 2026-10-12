"""Real frozen model generation plus official harness extraction. No human labels."""
import argparse,hashlib,json,platform,resource,sys,time,traceback
from pathlib import Path
from prepare import prepare,digest,ROOT
from scoring import score,aggregate

def sha(x):return hashlib.sha256(x.encode()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main(a):
    import torch,transformers,pyarrow.parquet as pq
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from lm_eval.filters.extraction import RegexFilter
    from lm_eval.filters.selection import TakeFirstFilter
    from importlib.metadata import version
    start=time.perf_counter();cfg=json.loads((ROOT/'config.json').read_text());lock=prepare(a.cache)
    plan=json.loads((ROOT/'plan.json').read_text());sample=json.loads((ROOT/'review_sample.json').read_text())
    assert sample['protocol_sha256']==digest(ROOT/'PROTOCOL.md')
    source=a.cache/'glue'/lock['glue']['revision']/'sst2/validation-00000-of-00001.parquet'
    data=pq.read_table(source).to_pylist();lookup={r['idx']:r for r in data}
    ids=plan['development'][:2] if a.smoke else plan['development'];assert not set(ids)&set(plan['reserved'])
    a.out.mkdir(parents=True,exist_ok=False)
    identity={'created_at':__import__('datetime').datetime.now().astimezone().isoformat(),'config':cfg,'source_hashes':{p.name:digest(p) for p in ROOT.glob('*.py')},'protocol_sha256':digest(ROOT/'PROTOCOL.md'),'sample_sha256':digest(ROOT/'review_sample.json'),'resources':{k:v['sha256'] for k,v in lock['files'].items()},'selected':ids,'inference_cache':'disabled'}
    identity['cache_key']=sha(json.dumps(identity,sort_keys=True));save(a.out/'identity.json',identity)
    save(a.out/'stages.json',{'download':'verified cache','preprocess':'passed','inference':'started','automatic_evaluation':'pending','human_evaluation':'pending'})
    torch.set_num_threads(4);torch.manual_seed(cfg['seed'])
    tok=AutoTokenizer.from_pretrained(a.cache/'models'/lock['model']['revision'],local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(a.cache/'models'/lock['model']['revision'],local_files_only=True,torch_dtype=torch.float32,attn_implementation='eager').eval()
    setup=time.perf_counter()-start;rows=[];times=[]
    for idx in ids:
        r=lookup[idx];messages=[{'role':'system','content':cfg['system']},{'role':'user','content':cfg['task']+'\n'+cfg['contracts']['json']+'\nReview: '+r['sentence']}]
        prompt=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        inputs=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_tensors='pt',return_dict=True)
        n=inputs['input_ids'].shape[1];assert n<=512
        begin=time.perf_counter()
        with torch.inference_mode():
            shape=list(model(**inputs).logits.shape) if not rows else None
            output=model.generate(**inputs,**cfg['decoding'],pad_token_id=tok.eos_token_id)
        dt=time.perf_counter()-begin;times.append(dt)
        generated=output[0,n:].tolist();text=tok.decode(generated,skip_special_tokens=True)
        official=list(TakeFirstFilter().apply(RegexFilter(regex_pattern=cfg['regex']).apply([[text]],None),None))[0]
        scores=score(text,r['label']);assert (official.lower() if official!='[invalid]' else None)==scores['first_label']
        rows.append({'idx':idx,'gold':r['label'],'sentence_sha256':sha(r['sentence']),'prompt_sha256':sha(prompt),'input_length':n,'input_shape':list(inputs['input_ids'].shape),'first_logits_shape':shape,'output':text,'output_sha256':sha(text),'generated_length':len(generated),'stop':'eos' if generated and generated[-1]==tok.eos_token_id else 'length','scores':scores,'official_extracted':official,'seconds':dt})
        save(a.out/'records.json',rows);print(json.dumps({'idx':idx,'seconds':dt,'generated':len(generated)}),flush=True)
    elapsed=time.perf_counter()-start;rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
    projection=setup+sum(times)*64 if a.smoke else None
    save(a.out/'summary.json',aggregate(rows));save(a.out/'environment.json',{'os':platform.platform(),'python':platform.python_version(),'torch':torch.__version__,'transformers':transformers.__version__,'harness':version('lm_eval'),'device':'cpu','dtype':'float32','threads':4,'elapsed_after_import_seconds':elapsed,'inference_seconds':sum(times),'peak_rss_bytes':rss,'projected_seconds':projection,'gpu_memory':'not measured; CPU'})
    assert elapsed<cfg['walltime_budget_seconds'] and rss<cfg['rss_budget_bytes']
    if a.smoke:assert projection<cfg['walltime_budget_seconds']
    save(a.out/'stages.json',{'download':'verified12 cached files','preprocess':'passed; no truncation','inference':'passed','automatic_evaluation':'passed','human_evaluation':'pending','confirmation_inference':False,'exit_code':0})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    try:main(a)
    except Exception as e:
        if a.out.is_dir():save(a.out/'failure.json',{'type':type(e).__name__,'error':str(e)})
        traceback.print_exc();sys.exit(1)
