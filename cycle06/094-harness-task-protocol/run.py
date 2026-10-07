"""Real SST2 likelihood audit and unmodified lm-eval official API evaluation."""
import argparse, datetime, hashlib, importlib.metadata, json, os, pathlib, platform, resource, sys, time, traceback
from prepare import ROOT,prepare,digest
def sha(x):return hashlib.sha256(x.encode()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str)+'\n')
def load_data(cache,lock):
    import pyarrow.parquet as pq
    base=cache/'glue'/lock['glue']['revision']/'sst2'
    return {s:pq.read_table(base/f'{s}-00000-of-00001.parquet').to_pylist() for s in ['train','validation']}
def split(rows):
    order=sorted(rows,key=lambda r:sha('88:'+str(r['idx'])))
    return order[:64],order[64:192]
def context(row):return row['sentence']+'\nQuestion: Is this sentence positive or negative?\nAnswer:'
def independent(model,tok,rows,maxlen,private):
    import torch
    records=[];traces=[]
    for row in rows:
        ctx=context(row);ci=tok.encode(ctx,add_special_tokens=False);candidates=[]
        for choice in ['negative','positive']:
            cont=' '+choice;ids=tok.encode(ctx+cont,add_special_tokens=False)
            assert ids[:len(ci)]==ci and len(ids)>len(ci)
            assert len(ids)<=maxlen,'refuse silent truncation'
            target=ids[len(ci):];inputs=torch.tensor([ids[:-1]])
            with torch.inference_mode():logits=model(inputs).logits;lp=logits[0].log_softmax(-1)
            values=[float(lp[len(ci)-1+j,t]) for j,t in enumerate(target)]
            candidates.append({'choice':choice,'continuation_ids':target,'token_logprobs':values,'sum_logprob':sum(values),'input_shape':list(inputs.shape),'logits_shape':list(logits.shape)})
        pred=max(range(2),key=lambda j:candidates[j]['sum_logprob'])
        records.append({'idx':row['idx'],'gold':row['label'],'context_sha256':sha(ctx),'context_tokens':len(ci),'candidates':candidates,'pred':pred,'acc':int(pred==row['label'])})
        traces.append({'idx':row['idx'],'sentence':row['sentence'],'context':ctx,'context_ids':ci})
    save(private/'input_traces.json',traces)
    return records
def main(a):
    import torch,transformers
    from transformers import AutoModelForCausalLM,AutoTokenizer
    start=time.perf_counter();cfg=json.loads((ROOT/'config.json').read_text());torch.set_num_threads(cfg['threads']);torch.manual_seed(cfg['seed'])
    lock=prepare(a.cache);save(a.out/'stages.json',{'download':'hash-verified cached pinned files','preprocess':'started','inference':'pending','evaluation':'pending'})
    data=load_data(a.cache,lock);dev,reserved=split(data['validation']);selected=dev[:4] if a.mode=='manual' else dev[:2] if a.smoke else dev
    save(a.out/'split.json',{'mode':a.mode,'smoke':a.smoke,'dev':[r['idx'] for r in dev],'selected':[r['idx'] for r in selected],'reserved':[r['idx'] for r in reserved],'confirmation_inference':False,'train_rows':len(data['train']),'validation_rows':len(data['validation'])})
    assert not set(r['idx'] for r in selected)&set(r['idx'] for r in reserved)
    identity={'resources':{n:v['sha256'] for n,v in lock['files'].items()},'harness':lock['harness'],'config':cfg,'task_yaml_sha256':digest(ROOT/'official_sst2.yaml'),'code_hashes':{n:digest(ROOT/n) for n in ['run.py','prepare.py','config.json','official_sst2.yaml','PROTOCOL.md']},'inference_cache':'disabled; key saved for future use'}
    identity['key']=sha(json.dumps(identity,sort_keys=True));save(a.out/'identity.json',identity)
    tok=AutoTokenizer.from_pretrained(a.cache/'models'/lock['model']['revision'],local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(a.cache/'models'/lock['model']['revision'],local_files_only=True,torch_dtype=torch.float32,attn_implementation='eager').eval()
    ts=time.perf_counter();manual=independent(model,tok,selected,cfg['max_length'],a.private);independent_seconds=time.perf_counter()-ts;save(a.out/'independent.json',manual)
    if a.mode=='official':
        import lm_eval,yaml,datasets
        from lm_eval.api.task import ConfigurableTask
        from lm_eval.models.huggingface import HFLM
        def local_dataset(**kwargs):return datasets.DatasetDict({s:datasets.Dataset.from_list(rows) for s,rows in data.items()})
        task_config=yaml.safe_load((ROOT/'official_sst2.yaml').read_text());task_config['custom_dataset']=local_dataset
        task=ConfigurableTask(config=task_config)
        backend=HFLM(pretrained=model,tokenizer=tok,batch_size=1,max_length=cfg['max_length'],add_bos_token=False)
        ts=time.perf_counter()
        result=lm_eval.simple_evaluate(model=backend,tasks=[task],num_fewshot=0,samples={'sst2':sorted(r['idx'] for r in selected)},bootstrap_iters=0,log_samples=True,apply_chat_template=False,random_seed=94,numpy_random_seed=94,torch_random_seed=94,fewshot_random_seed=94)
        official_seconds=time.perf_counter()-ts;save(a.private/'official_raw.json',result)
        samples=result['samples']['sst2'];byid={r['idx']:r for r in manual};pub=[]
        for r in samples:
            idx=r['doc']['idx'];ref=byid[idx];scores=[float(x[0][0]) for x in r['resps']]
            assert r['doc_id']==idx and r['target']==ref['gold']
            assert all(arg[0]==context(r['doc']) and arg[1]==' '+['negative','positive'][j] for j,arg in enumerate(r['arguments']))
            errors=[abs(scores[j]-ref['candidates'][j]['sum_logprob']) for j in range(2)];assert max(errors)<=2e-5
            assert r['acc']==ref['acc']
            pub.append({'idx':idx,'gold':r['target'],'scores':scores,'is_greedy':[bool(x[0][1]) for x in r['resps']],'acc':r['acc'],'context_sha256':ref['context_sha256'],'max_score_error':max(errors)})
        assert len(pub)==len(selected)
        aggregate=result['results']['sst2']['acc,none'];assert aggregate==sum(r['acc'] for r in pub)/len(pub)
        save(a.out/'official_samples.json',pub);save(a.out/'official_metrics.json',result['results'])
        save(a.out/'audit.json',{'status':'passed','rows':len(pub),'candidate_requests':2*len(pub),'max_score_error':max(r['max_score_error'] for r in pub),'aggregate_recomputed':aggregate,'correct':sum(r['acc'] for r in pub),'no_confirmation':True,'official_raw_sha256':digest(a.private/'official_raw.json'),'raw_distribution':'original review text, prompt and context IDs retained private, public derived scores only'})
        save(a.out/'effective_protocol.json',{'original_yaml':task_config|{'custom_dataset':'local_dataset: offline transport only'},'num_fewshot':task.config.num_fewshot,'target_delimiter':task.config.target_delimiter,'fewshot_split':task.config.fewshot_split,'training_split':task.config.training_split,'fewshot_actual_samples':0,'apply_chat_template':False,'add_bos_token':False,'output_type':task.OUTPUT_TYPE,'metric_aggregation':{k:getattr(v,'__name__',str(v)) for k,v in task.aggregation().items()},'filter':'none/take_first','length_normalization':False,'eos_scored':False,'generation':False,'max_length':512,'batch_size':1,'task_version':1.0})
    else:official_seconds=None
    elapsed=time.perf_counter()-start;rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
    assert elapsed<cfg['walltime_budget_seconds'] and rss<cfg['rss_budget_bytes']
    save(a.out/'environment.json',{'os':platform.platform(),'python':platform.python_version(),'torch':torch.__version__,'transformers':transformers.__version__,'lm_eval':importlib.metadata.version('lm_eval'),'device':'cpu','dtype':'float32','threads':cfg['threads'],'seconds_after_imports':elapsed,'independent_seconds':independent_seconds,'official_seconds':official_seconds,'peak_rss_bytes':rss,'gpu_memory':'not measured; CPU only'})
    save(a.out/'stages.json',{'download':'hash-verified cached pinned files','preprocess':'passed; no truncation','inference':'passed; real pretrained model','evaluation':'independent worksheet ready' if a.mode=='manual' else 'official entry and independent comparison passed','exit_code':0});print(json.dumps({'mode':a.mode,'rows':len(selected),'seconds':elapsed,'rss':rss,'status':'passed'}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,required=True);p.add_argument('--private',type=pathlib.Path,required=True);p.add_argument('--mode',choices=['manual','official'],default='official');p.add_argument('--smoke',action='store_true');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);a.private.mkdir(parents=True,exist_ok=False)
    try:main(a)
    except Exception as e:save(a.out/'failure.json',{'error':str(e),'type':type(e).__name__,'exit_code':1});traceback.print_exc();sys.exit(1)
