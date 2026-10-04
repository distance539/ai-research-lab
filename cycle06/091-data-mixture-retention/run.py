"""Real four-arm SFT with fixed supervised budget and independent retention task."""
import argparse,gc,json,pathlib,platform,resource,sys,time,traceback
from prepare import ROOT,digest,prepare
from common import save,sha,build,collate,evaluate
from mixture import select,news_build,schedule,news_evaluate,summarize,paired

def main(a):
    import torch,transformers
    from transformers import AutoTokenizer,AutoModelForCausalLM
    start=time.perf_counter();c=json.loads((ROOT/'config.json').read_text());torch.set_num_threads(c['threads']);torch.manual_seed(c['seed']);lock=prepare(a.cache)
    stages={'download':'passed; all locked resources hash verified','preprocessing':'running','training':'pending','evaluation':'pending'};save(a.out/'stages.json',stages)
    tr,dev,reserved,news,ndev,nreserve=select(c,a.cache,lock)
    if a.smoke:dev=dev[:2];ndev=ndev[:4]
    split={'target_train':[r['idx'] for r in tr],'target_dev':[r['idx'] for r in dev],'target_reserved':[r['idx'] for r in reserved],'news_train':[r['idx'] for r in news],'news_dev':[r['idx'] for r in ndev],'news_reserved':[r['idx'] for r in nreserve],'smoke':a.smoke,'normalized_train_held_overlap':0};save(a.out/'split.json',split)
    md=a.cache/'models'/lock['model']['revision'];tok=AutoTokenizer.from_pretrained(md,local_files_only=True)
    tt=[dict(build(tok,c,r),task='target') for r in tr];nn=[news_build(tok,c,r) for r in news];assert all(len(r['ids'])-r['prefix']==8 for r in tt)
    arms=schedule(c,tt,nn,a.smoke);rows=tt+nn
    identity={'config':c,'resources':lock,'split':split,'chat_template_sha256':sha(tok.chat_template),'code_hashes':{n:digest(ROOT/n) for n in ['run.py','common.py','mixture.py','scoring.py','prepare.py','config.json','resources.lock.json','PROTOCOL.md']}}
    identity['cache_key']=sha(json.dumps(identity,sort_keys=True));save(a.out/'identity.json',identity)
    save(a.out/'rows.json',[{k:v for k,v in r.items() if k!='ids'}|{'length':len(r['ids']),'answer_tokens':len(r['ids'])-r['prefix']} for r in rows]);a.private.mkdir(parents=True,exist_ok=True);save(a.private/(a.out.name+'-token-trace.json'),rows)
    stages['preprocessing']='passed; disjoint rows, target8/news2 answer tokens, no truncation';save(a.out/'stages.json',stages)
    predictions=[];steps=[];checkpoints=[];summary={};budget={}
    def ev(model,stage):
        pred=[dict(r,task='target') for r in evaluate(model,tok,c,dev,stage)]+news_evaluate(model,tok,c,ndev,stage)
        predictions.extend(pred);summary[stage]=summarize(pred)
        (a.out/'predictions.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in predictions));save(a.out/'summary.json',summary)
    for arm,batches in arms.items():
        torch.manual_seed(c['seed']);model=AutoModelForCausalLM.from_pretrained(md,local_files_only=True,torch_dtype=torch.float32,attn_implementation='eager');model.config.use_cache=False;model.eval()
        if not predictions:ev(model,'before')
        opt=torch.optim.SGD(model.parameters(),lr=c['lr'],momentum=0,weight_decay=0);probe=model.model.layers[0].self_attn.q_proj.weight.detach().clone();ts=time.perf_counter()
        for i,(task,rr) in enumerate(batches):
            opt.zero_grad(set_to_none=True);micro=[];weighted_loss=0
            for j in range(0,len(rr),2):
                b=collate(rr[j:j+2],'assistant',tok.pad_token_id);out=model(**b,use_cache=False);n=int((b['labels'][:,1:]!=-100).sum())
                manual=torch.nn.functional.cross_entropy(out.logits[:,:-1].reshape(-1,out.logits.shape[-1]),b['labels'][:,1:].reshape(-1),ignore_index=-100);err=abs(manual.item()-out.loss.item());assert err<2e-6
                loss=out.loss*n/16;weighted_loss+=loss.item();loss.backward()
                micro.append({'idx':[r['idx'] for r in rr[j:j+2]],'loss':out.loss.item(),'ce_error':err,'supervised_tokens':n,'input_tokens':int(b['attention_mask'].sum()),'padded_tokens':b['input_ids'].numel(),'attention_cells':2*b['input_ids'].shape[1]**2,'input_shape':list(b['input_ids'].shape),'logits_shape':list(out.logits.shape)})
                del out,manual,loss,b
            assert sum(m['supervised_tokens'] for m in micro)==16
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),c['clip_norm']).item();assert 0<norm<float('inf');opt.step()
            rec={'arm':arm,'step':i+1,'task':task,'idx':[r['idx'] for r in rr],'loss':weighted_loss,'gradient_norm':norm,'microbatches':micro,**{k:sum(m[k] for m in micro) for k in ['supervised_tokens','input_tokens','padded_tokens','attention_cells']}};steps.append(rec);print(json.dumps(rec),flush=True)
            if time.perf_counter()-start>c['walltime_budget_seconds']:raise TimeoutError('budget exceeded')
        sec=time.perf_counter()-ts;delta=(model.model.layers[0].self_attn.q_proj.weight-probe).abs().max().item();assert delta>0
        ck=a.private/(a.out.name+'-'+arm+'-state.pt');torch.save(model.state_dict(),ck)
        checkpoints.append({'arm':arm,'filename':ck.name,'sha256':digest(ck),'bytes':ck.stat().st_size,'q_proj_max_abs_change':delta,'training_seconds':sec,'parameters':sum(p.numel() for p in model.parameters())})
        ss=[r for r in steps if r['arm']==arm];budget[arm]={'steps':len(ss),**{k:sum(r[k] for r in ss) for k in ['supervised_tokens','input_tokens','padded_tokens','attention_cells']},'by_task':{t:{'steps':sum(r['task']==t for r in ss),'exposures':sum(len(r['idx']) for r in ss if r['task']==t),'supervised_tokens':sum(r['supervised_tokens'] for r in ss if r['task']==t)} for t in ['target','news']}}
        model.zero_grad(set_to_none=True);ev(model,arm);save(a.out/'steps.json',steps);save(a.out/'budget.json',budget);save(a.out/'checkpoints.json',checkpoints)
        del model,opt,probe;gc.collect()
    save(a.out/'paired.json',paired(predictions))
    eligible=[arm for arm in ['target_only','replay25','replay50'] if summary[arm]['news']['joint_count']>=summary['before']['news']['joint_count']-1]
    chosen=max(eligible,key=lambda arm:summary[arm]['json']['joint_count']) if eligible else None
    save(a.out/'selection.json',{'rule':'news joint >= baseline-1; then max target JSON joint; ties lower replay fraction','eligible':eligible,'selected_development_only':chosen,'confirmation_inference':False})
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024);assert rss<c['rss_budget_bytes']
    save(a.out/'environment.json',{'os':platform.platform(),'python':platform.python_version(),'torch':torch.__version__,'transformers':transformers.__version__,'device':'cpu','dtype':'float32','threads':torch.get_num_threads(),'seconds_after_imports':time.perf_counter()-start,'peak_rss_bytes':rss,'gpu_memory':'not measured; CPU only'})
    stages.update(training='passed;four arms complete',evaluation='passed;development only, no confirmation',exit_code=0);save(a.out/'stages.json',stages)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--cache',type=pathlib.Path,required=True);ap.add_argument('--out',type=pathlib.Path,required=True);ap.add_argument('--private',type=pathlib.Path,required=True);ap.add_argument('--smoke',action='store_true');a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    try:main(a)
    except Exception as e:save(a.out/'failure.json',{'error':str(e),'type':type(e).__name__,'exit_code':1});traceback.print_exc();sys.exit(1)
