"""Controlled real SFT: nested data sizes at equal supervised-token budgets."""
import argparse, gc, json, pathlib, platform, resource, sys, time, traceback
from prepare import ROOT, digest, prepare
from common import save, sha, ordered, build, collate, evaluate
from scoring import aggregate

def select(c,cache,lock):
    import pyarrow.parquet as pq
    dp=cache/'sst2'/lock['data']['revision']/'data'
    train=pq.read_table(dp/'train-00000-of-00001.parquet').to_pylist()
    groups=[ordered([r for r in train if r['label']==g],c['train_seed'])[:16] for g in (0,1)]
    pool=[r for pair in zip(*groups) for r in pair]
    assert set(r['idx'] for r in pool[:8])==set(r['idx'] for r in ordered(train,89)[:8])
    val=ordered(pq.read_table(dp/'validation-00000-of-00001.parquet').to_pylist(),c['split_seed'])
    dev=val[:c['eval_count']];reserved=val[c['count']:c['count']+c['confirmation_count']]
    norm=lambda s:' '.join(s.lower().split())
    texts={norm(r['sentence']) for r in pool};assert len(texts)==len(pool)
    assert not texts&{norm(r['sentence']) for r in dev+reserved}
    return pool,dev,reserved

def nll(model,tok,rows):
    import torch
    losses=[]
    with torch.inference_mode():
        for r in rows:
            b=collate([r],'assistant',tok.pad_token_id);o=model(**b,use_cache=False)
            ce=torch.nn.functional.cross_entropy(o.logits[:,:-1].reshape(-1,o.logits.shape[-1]),b['labels'][:,1:].reshape(-1),ignore_index=-100,reduction='none')
            vals=ce[(b['labels'][:,1:]!=-100).reshape(-1)].tolist();assert len(vals)==8
            losses.append({'idx':r['idx'],'gold':r['gold'],'token_nll':vals,'mean':sum(vals)/len(vals)})
    return losses

def main(a):
    import torch, transformers
    from transformers import AutoTokenizer,AutoModelForCausalLM
    start=time.perf_counter();c=json.loads((ROOT/'config.json').read_text());torch.set_num_threads(c['threads']);torch.manual_seed(c['seed']);lock=prepare(a.cache)
    stages={'download':'passed;17 pinned resources hash verified','preprocessing':'running','training':'pending','evaluation':'pending'};save(a.out/'stages.json',stages)
    pool,dev,reserved=select(c,a.cache,lock)
    if a.smoke:dev=dev[:2]
    split={'training_pool':[r['idx'] for r in pool],'small_nested':[r['idx'] for r in pool[:8]],'executed_development':[r['idx'] for r in dev],'confirmation_reserved':[r['idx'] for r in reserved],'train_validation_normalized_overlap':0,'smoke':a.smoke};save(a.out/'split.json',split)
    md=a.cache/'models'/lock['model']['revision'];tok=AutoTokenizer.from_pretrained(md,local_files_only=True);rows=[build(tok,c,r) for r in pool]
    assert all(len(r['ids'])-r['prefix']==8 for r in rows)
    schedule={name:(rows[:spec['unique']]*spec['epochs'])[:4 if a.smoke else spec['unique']*spec['epochs']] for name,spec in c['arms'].items()}
    identity={'config':c,'resources':lock,'split':split,'chat_template_sha256':sha(tok.chat_template),'code_hashes':{n:digest(ROOT/n) for n in ['run.py','common.py','prepare.py','scoring.py','config.json','resources.lock.json','PROTOCOL.md']}}
    identity['cache_key']=sha(json.dumps(identity,sort_keys=True));save(a.out/'identity.json',identity)
    save(a.out/'rows.json',[{k:v for k,v in r.items() if k!='ids'}|{'length':len(r['ids']),'answer_tokens':len(r['ids'])-r['prefix']} for r in rows])
    a.private.mkdir(parents=True,exist_ok=True);save(a.private/(a.out.name+'-token_trace.json'),rows)
    stages['preprocessing']='passed; nested classes, no overlap,8answer targets, no truncation';save(a.out/'stages.json',stages)
    predictions=[];steps=[];checkpoints=[];summary={};train_nll={}
    for arm,exposures in schedule.items():
        torch.manual_seed(c['seed']);model=AutoModelForCausalLM.from_pretrained(md,local_files_only=True,torch_dtype=torch.float32,attn_implementation='eager');model.config.use_cache=False;model.eval()
        if not predictions:
            before=evaluate(model,tok,c,dev,'before');predictions+=before;summary['before']=aggregate(before);train_nll['before']=nll(model,tok,rows)
        opt=torch.optim.SGD(model.parameters(),lr=c['lr'],momentum=0,weight_decay=0);probe=model.model.layers[0].self_attn.q_proj.weight.detach().clone();ts=time.perf_counter()
        for i in range(0,len(exposures),2):
            rr=exposures[i:i+2];assert [r['gold'] for r in rr]==[0,1];b=collate(rr,'assistant',tok.pad_token_id);opt.zero_grad(set_to_none=True)
            out=model(**b,use_cache=False);valid=b['labels'][:,1:]!=-100;assert valid.sum().item()==16
            manual=torch.nn.functional.cross_entropy(out.logits[:,:-1].reshape(-1,out.logits.shape[-1]),b['labels'][:,1:].reshape(-1),ignore_index=-100);err=abs(manual.item()-out.loss.item());assert err<2e-6
            out.loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),c['clip_norm']).item();assert 0<norm<float('inf');opt.step()
            rec={'arm':arm,'step':i//2+1,'idx':[r['idx'] for r in rr],'loss':out.loss.item(),'ce_error':err,'supervised_tokens':16,'input_tokens':int(b['attention_mask'].sum()),'padded_tokens':b['input_ids'].numel(),'attention_cells':2*b['input_ids'].shape[1]**2,'gradient_norm':norm,'input_shape':list(b['input_ids'].shape),'logits_shape':list(out.logits.shape)};steps.append(rec);print(json.dumps(rec),flush=True)
            del out,manual,b
            if time.perf_counter()-start>c['walltime_budget_seconds']:raise TimeoutError('budget exceeded')
        sec=time.perf_counter()-ts;delta=(model.model.layers[0].self_attn.q_proj.weight-probe).abs().max().item();assert delta>0
        ck=a.private/(a.out.name+'-'+arm+'-state.pt');torch.save(model.state_dict(),ck)
        checkpoints.append({'arm':arm,'filename':ck.name,'sha256':digest(ck),'bytes':ck.stat().st_size,'q_proj_max_abs_change':delta,'training_seconds':sec,'parameters':sum(p.numel() for p in model.parameters())})
        model.zero_grad(set_to_none=True);pred=evaluate(model,tok,c,dev,arm);predictions+=pred;summary[arm]=aggregate(pred);train_nll[arm]=nll(model,tok,rows)
        del model,opt,probe;gc.collect()
    (a.out/'predictions.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in predictions));save(a.out/'steps.json',steps);save(a.out/'summary.json',summary);save(a.out/'checkpoints.json',checkpoints);save(a.out/'train_nll.json',train_nll)
    budget={arm:{'unique_rows':len(set(r['idx'] for r in exp)),'exposures':len(exp),'steps':len(exp)//2,**{k:sum(r[k] for r in steps if r['arm']==arm) for k in ['supervised_tokens','input_tokens','padded_tokens','attention_cells']}} for arm,exp in schedule.items()};save(a.out/'budget.json',budget)
    assert budget['repeat8']['supervised_tokens']==budget['unique32']['supervised_tokens']
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024);assert rss<c['rss_budget_bytes']
    save(a.out/'environment.json',{'os':platform.platform(),'python':platform.python_version(),'torch':torch.__version__,'transformers':transformers.__version__,'device':'cpu','dtype':'float32','threads':torch.get_num_threads(),'seconds_after_imports':time.perf_counter()-start,'peak_rss_bytes':rss,'gpu_memory':'not measured; CPU only'})
    stages.update(training='passed;three arms complete',evaluation='passed;development only, no confirmation',exit_code=0);save(a.out/'stages.json',stages)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--cache',type=pathlib.Path,required=True);ap.add_argument('--out',type=pathlib.Path,required=True);ap.add_argument('--private',type=pathlib.Path,required=True);ap.add_argument('--smoke',action='store_true');a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    try:main(a)
    except Exception as e:save(a.out/'failure.json',{'error':str(e),'type':type(e).__name__,'exit_code':1});traceback.print_exc();sys.exit(1)
