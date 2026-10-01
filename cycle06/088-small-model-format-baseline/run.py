"""Real frozen SmolLM2 generation on a preregistered public SST-2 development subset."""
import argparse, hashlib, json, os, pathlib, platform, resource, sys, time, traceback
from prepare import ROOT, digest, prepare
from scoring import score, aggregate

def save(p, obj): p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def load_rows(cache, lock):
    import pyarrow.parquet as pq
    return pq.read_table(cache/'sst2'/lock['data']['revision']/'data/validation-00000-of-00001.parquet').to_pylist()
def ordered_rows(rows, seed):
    return sorted(rows,key=lambda r:hashlib.sha256(f'{seed}:{r["idx"]}'.encode()).hexdigest())
def message(cfg,contract,sentence):
    return [{'role':'system','content':cfg['system']},{'role':'user','content':cfg['task']+'\n'+cfg['contracts'][contract]+'\nReview: '+sentence}]

def main(a):
    import torch, transformers, numpy, pyarrow
    from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
    t0=time.perf_counter(); cfg=json.loads((ROOT/'config.json').read_text());torch.set_num_threads(cfg['threads']);torch.manual_seed(cfg['seed'])
    lock=prepare(a.cache);save(a.out/'stages.json',{'download':'verified existing hash-locked cache','preprocess':'running','inference':'pending','evaluation':'pending'})
    rows=load_rows(a.cache,lock);assert len(rows)==872
    rows=ordered_rows(rows,cfg['seed']);dev=rows[:cfg['count']];reserved=rows[cfg['count']:cfg['count']+cfg['confirmation_count']]
    split={'development':[r['idx'] for r in dev],'confirmation_reserved':[r['idx'] for r in reserved],'unused_count':len(rows)-len(dev)-len(reserved)}
    assert not set(split['development']) & set(split['confirmation_reserved']);save(a.out/'split.json',split)
    if a.smoke: dev=dev[:2]
    modeldir=a.cache/'models'/lock['model']['revision']
    tok=AutoTokenizer.from_pretrained(modeldir,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(modeldir,local_files_only=True,torch_dtype=torch.float32,attn_implementation='eager').eval()
    gen=GenerationConfig(**cfg['decoding'],eos_token_id=model.config.eos_token_id,pad_token_id=model.config.pad_token_id,bos_token_id=model.config.bos_token_id)
    identity={'config':cfg,'resources':lock,'selected_ids':[r['idx'] for r in dev],'smoke':a.smoke,
        'code_hashes':{n:digest(ROOT/n) for n in ['run.py','prepare.py','scoring.py','config.json','resources.lock.json','PROTOCOL.md']},
        'chat_template_sha256':hashlib.sha256(tok.chat_template.encode()).hexdigest(),'generation_config':gen.to_dict()}
    identity['cache_key']=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest();save(a.out/'identity.json',identity)
    modelinfo={'parameters':sum(p.numel() for p in model.parameters()),'config':model.config.to_dict(),'dtype':str(next(model.parameters()).dtype)}
    modelinfo['config'].pop('_name_or_path',None);save(a.out/'model.json',modelinfo)
    records=[];inference_start=time.perf_counter();shape=None
    with torch.inference_mode(),(a.out/'predictions.jsonl').open('w') as f:
        for row in dev:
            for contract in cfg['contracts']:
                msgs=message(cfg,contract,row['sentence']);text=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
                inputs=tok.apply_chat_template(msgs,tokenize=True,add_generation_prompt=True,return_tensors='pt',return_dict=True)
                assert tok.encode(text,add_special_tokens=False)==inputs['input_ids'][0].tolist()
                assert inputs['input_ids'].shape[1]<=cfg['max_input_tokens']
                if shape is None:
                    probe=model(**inputs,output_hidden_states=True,use_cache=False)
                    shape={'input_ids':list(inputs['input_ids'].shape),'last_hidden':list(probe.hidden_states[-1].shape),'logits':list(probe.logits.shape),'hidden_states_count':len(probe.hidden_states),'first_next_token_argmax':int(probe.logits[0,-1].argmax())}
                    del probe
                start=time.perf_counter();out=model.generate(**inputs,generation_config=gen);elapsed=time.perf_counter()-start
                ids=out[0,inputs['input_ids'].shape[1]:].tolist();response=tok.decode(ids,skip_special_tokens=True)
                if not records: assert ids[0]==shape['first_next_token_argmax']
                rec={'idx':row['idx'],'gold':row['label'],'contract':contract,'sentence_sha256':hashlib.sha256(row['sentence'].encode()).hexdigest(),
                    'prompt_sha256':hashlib.sha256(text.encode()).hexdigest(),'input_length':inputs['input_ids'].shape[1],
                    'generated_ids':ids,'output':response,'output_with_special_tokens':tok.decode(ids,skip_special_tokens=False),
                    'stop':'eos' if ids[-1]==gen.eos_token_id else 'length','seconds':elapsed,**score(response,contract,row['label'])}
                records.append(rec);f.write(json.dumps(rec,ensure_ascii=False)+'\n');f.flush()
                print(json.dumps({'idx':row['idx'],'contract':contract,'tokens':len(ids),'seconds':round(elapsed,3),'format':rec['format'],'correct':rec['content_proxy']}),flush=True)
                if time.perf_counter()-t0>cfg['walltime_budget_seconds']:raise TimeoutError('predeclared runtime budget')
    save(a.out/'shapes.json',shape);save(a.out/'summary.json',aggregate(records))
    cases={c:{b:min((r for r in records if r['contract']==c and r['bucket']==b),key=lambda r:r['idx'],default=None) for b in ['F0C0','F0C1','F1C0','F1C1']} for c in cfg['contracts']};save(a.out/'cases.json',cases)
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform!='darwin':rss*=1024
    save(a.out/'environment.json',{'os':platform.platform(),'machine':platform.machine(),'python':platform.python_version(),'torch':torch.__version__,'transformers':transformers.__version__,'numpy':numpy.__version__,'pyarrow':pyarrow.__version__,'device':'cpu','threads':torch.get_num_threads(),'total_seconds':time.perf_counter()-t0,'inference_seconds':time.perf_counter()-inference_start,'peak_rss_bytes':rss,'gpu_memory':'not measured; CPU only'})
    assert rss<cfg['rss_budget_bytes'];save(a.out/'stages.json',{'download':'passed; hash-verified cache','preprocess':'passed; no truncation','inference':'passed','evaluation':'passed','exit_code':0})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--cache',type=pathlib.Path,required=True);ap.add_argument('--out',type=pathlib.Path,required=True);ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    try:main(args)
    except Exception as e:
        save(args.out/'failure.json',{'type':type(e).__name__,'message':str(e),'exit_code':1});traceback.print_exc();sys.exit(1)
