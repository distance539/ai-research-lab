"""092 real fixed-checkpoint, cross-template paired experiment."""
import argparse,gc,json,pathlib,platform,resource,sys,time,traceback,re
from prepare import ROOT,prepare,digest
from common import save,sha,build,collate
from mixture import select,news_score
from scoring import score
from templates import messages,contract

def state_hash(state):
    import hashlib
    h=hashlib.sha256()
    for k,v in sorted(state.items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def summarize(rr):
    out={}
    for stage in dict.fromkeys(r['stage'] for r in rr):
        out[stage]={}
        for name in dict.fromkeys(r['template'] for r in rr):
            rows=[r for r in rr if (r['stage'],r['template'])==(stage,name)];n=len(rows)
            out[stage][name]={'n':n,**{k:sum(r[k] for r in rows) for k in ['format','content_proxy','joint']},'input_tokens':sum(r['input_length'] for r in rows),'generated_tokens':sum(r['generated_length'] for r in rows),'length_stops':sum(r['stop']=='length' for r in rows)}
    return out

def pairs(rr):
    ix={(r['stage'],r['template'],r['idx']):r for r in rr};out={}
    contrasts=[('before',n,'after',n) for n in dict.fromkeys(r['template'] for r in rr)]+[(s,'json_original',s,n) for s in ['before','after'] for n in ['json_paraphrase','json_reordered','plain_original']]+[(s,'news_original',s,'news_paraphrase') for s in ['before','after']]
    for sa,ta,sb,tb in contrasts:
        aa=[r for r in rr if (r['stage'],r['template'])==(sa,ta)];bb=[ix[sb,tb,r['idx']] for r in aa]
        out[f'{sa}/{ta} -> {sb}/{tb}']={k:{'win':sum(b[k]>a[k] for a,b in zip(aa,bb)),'tie':sum(b[k]==a[k] for a,b in zip(aa,bb)),'loss':sum(b[k]<a[k] for a,b in zip(aa,bb)),'delta_count':sum(int(b[k])-int(a[k]) for a,b in zip(aa,bb))} for k in ['format','content_proxy','joint']}
    return out

def main(a):
    import torch,transformers
    from transformers import AutoTokenizer,AutoModelForCausalLM,GenerationConfig
    start=time.perf_counter();c=json.loads((ROOT/'config.json').read_text());torch.set_num_threads(c['threads']);torch.manual_seed(c['seed']);lock=prepare(a.cache)
    stages={'download':'passed;20resource hashes verified','preprocessing':'running','training':'pending','inference':'pending','evaluation':'pending'};save(a.out/'stages.json',stages)
    tr,dev,reserved,_,ndev,nreserve=select(c,a.cache,lock)
    if a.smoke:dev=dev[:2];ndev=ndev[:4]
    a.private.mkdir(parents=True,exist_ok=True);md=a.cache/'models'/lock['model']['revision'];tok=AutoTokenizer.from_pretrained(md,local_files_only=True)
    split={'train':[r['idx'] for r in tr],'dev':[r['idx'] for r in dev],'reserved':[r['idx'] for r in reserved],'news_dev':[r['idx'] for r in ndev],'news_reserved':[r['idx'] for r in nreserve],'news_training':[],'smoke':a.smoke};save(a.out/'split.json',split)
    prompts=[]
    for name in c['templates']:
        news=name.startswith('news');rows=ndev if news else dev
        for r in rows:
            source=r['text' if news else 'sentence'];msgs=messages(c,name,source);assert msgs[-1]['content'].endswith(source)
            text=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True);ids=tok(text,add_special_tokens=False)['input_ids'];assert len(ids)<=c['max_input_tokens']
            prompts.append({'template':name,'task':'news' if news else 'sst2','idx':r['idx'],'gold':r['label'],'source':source,'messages':msgs,'text':text,'ids':ids,'source_sha256':sha(source),'prompt_sha256':sha(text)})
    review=[{k:v for k,v in r.items() if k not in ['ids','text']}|{'human_decision':'','reviewer':'','reviewed_at':'','note':''} for r in prompts]
    save(a.private/(a.out.name+'-human-review.json'),review)
    public_review=[{k:r[k] for k in ['template','task','idx','gold','source_sha256','prompt_sha256']}|{'human_decision':'','reviewer':'','reviewed_at':''} for r in review];save(a.out/'human_review_pending.json',public_review)
    save(a.out/'prompt_audit.json',[{k:r[k] for k in ['template','task','idx','gold','source_sha256','prompt_sha256']}|{'input_length':len(r['ids'])} for r in prompts])
    stages['preprocessing']='passed;original text/labels held constant;human semantic review pending';save(a.out/'stages.json',stages)
    torch.manual_seed(c['checkpoint_training_seed']);model=AutoModelForCausalLM.from_pretrained(md,local_files_only=True,torch_dtype=torch.float32,attn_implementation='eager');model.eval();model.config.use_cache=False
    rows=[build(tok,c,r) for r in tr];gen=GenerationConfig(**c['decoding'],eos_token_id=tok.eos_token_id,pad_token_id=tok.pad_token_id,bos_token_id=tok.bos_token_id)
    preds=[];shapes=[]
    def ev(stage):
        with torch.inference_mode():
            for r in prompts:
                x=torch.tensor([r['ids']]);t=time.perf_counter()
                if not shapes:
                    y=model(input_ids=x,attention_mask=torch.ones_like(x),output_hidden_states=True);shapes.append({'input':list(x.shape),'hidden':list(y.hidden_states[-1].shape),'logits':list(y.logits.shape),'first_argmax':int(y.logits[0,-1].argmax())});del y
                y=model.generate(input_ids=x,attention_mask=torch.ones_like(x),generation_config=gen)[0,len(r['ids']):].tolist();txt=tok.decode(y,skip_special_tokens=True)
                if not preds:assert y[0]==shapes[0]['first_argmax']
                scorer=news_score(txt,r['gold']) if r['task']=='news' else score(txt,contract(r['template']),r['gold'])
                preds.append({k:r[k] for k in ['template','task','idx','gold','source_sha256','prompt_sha256']}|{'stage':stage,'input_length':len(r['ids']),'generated_ids':y,'generated_length':len(y),'output':txt,'output_sha256':sha(txt),'stop':'eos' if y[-1]==tok.eos_token_id else 'length','seconds':time.perf_counter()-t,**scorer})
        print(json.dumps({'stage':stage,'predictions':len(preds),'summary':summarize(preds)[stage]}),flush=True)
    ev('before');steps=[];ts=time.perf_counter()
    if a.checkpoint:
        assert digest(a.checkpoint)==c['inherited_checkpoint_sha256'];state=torch.load(a.checkpoint,weights_only=True,map_location='cpu');model.load_state_dict(state);del state
        training={'mode':'reuse091target_only','file_sha256':digest(a.checkpoint),'steps_this_run':0}
    else:
        opt=torch.optim.SGD(model.parameters(),lr=c['lr'],momentum=0,weight_decay=0)
        for i in range(64):
            batch=rows[(i*2)%32:(i*2)%32+2];b=collate(batch,'assistant',tok.pad_token_id);opt.zero_grad(set_to_none=True);o=model(**b,use_cache=False)
            manual=torch.nn.functional.cross_entropy(o.logits[:,:-1].reshape(-1,o.logits.shape[-1]),b['labels'][:,1:].reshape(-1),ignore_index=-100);err=abs(manual.item()-o.loss.item());assert err<2e-6
            n=int((b['labels'][:,1:]!=-100).sum());assert n==16;o.loss.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0).item();opt.step()
            steps.append({'step':i+1,'idx':[r['idx'] for r in batch],'loss':o.loss.item(),'gradient_norm':gn,'supervised_tokens':n,'input_tokens':int(b['attention_mask'].sum()),'shape':list(b['input_ids'].shape),'ce_error':err});print(json.dumps(steps[-1]),flush=True);del o,b,manual
        model.zero_grad(set_to_none=True);del opt
        ck=a.private/(a.out.name+'-target_only-state.pt');torch.save(model.state_dict(),ck);training={'mode':'regenerated091target_only','file_sha256':digest(ck),'checkpoint_filename':ck.name,'steps_this_run':64}
    training.update(seconds=time.perf_counter()-ts,state_sha256=state_hash(model.state_dict()));save(a.out/'training.json',training);save(a.out/'steps.json',steps)
    stages['training']='passed;'+training['mode'];ev('after');save(a.out/'shapes.json',shapes)
    raw=a.private/(a.out.name+'-raw-predictions.jsonl');raw.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in preds))
    public=[];redacted=[]
    for r in preds:
        source=next(x['source'] for x in prompts if (x['template'],x['idx'])==(r['template'],r['idx']));words=re.findall(r'\w+',source.lower());output=' '.join(re.findall(r'\w+',r['output'].lower()));echo=any(' '.join(words[i:i+8]) in output for i in range(len(words)-7))
        x=dict(r)
        if echo:
            x.pop('output');x.pop('generated_ids');x['redacted']='8-word source overlap; raw retained privately';redacted.append({k:r[k] for k in ['stage','template','idx']})
        public.append(x)
    (a.out/'predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in public));save(a.out/'summary.json',summarize(preds));save(a.out/'paired.json',pairs(preds));save(a.out/'distribution.json',{'raw_sha256':digest(raw),'raw_count':len(preds),'public_full_count':len(preds)-len(redacted),'redactions':redacted,'rule':'8consecutive source words; excludes raw input/source/tokenIDs for inputs'})
    identity={'config':c,'resources':lock,'split':split,'training':training,'chat_template_sha256':sha(tok.chat_template),'code_hashes':{n:digest(ROOT/n) for n in ['run.py','templates.py','common.py','mixture.py','scoring.py','prepare.py','config.json','resources.lock.json','PROTOCOL.md']}}
    identity['cache_key']=sha(json.dumps(identity,sort_keys=True));save(a.out/'identity.json',identity)
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024);seconds=time.perf_counter()-start;assert rss<c['rss_budget_bytes'] and seconds<c['walltime_budget_seconds']
    save(a.out/'environment.json',{'os':platform.platform(),'python':platform.python_version(),'torch':torch.__version__,'transformers':transformers.__version__,'device':'cpu','dtype':'float32','threads':4,'seconds_after_imports':seconds,'peak_rss_bytes':rss,'gpu_memory':'not measured'})
    stages.update(inference='passed;'+str(len(preds))+'generations',evaluation='automated F/C/J passed;human review pending',exit_code=0);save(a.out/'stages.json',stages)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,required=True);p.add_argument('--private',type=pathlib.Path,required=True);p.add_argument('--checkpoint',type=pathlib.Path);p.add_argument('--smoke',action='store_true');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    try:main(a)
    except Exception as e:save(a.out/'failure.json',{'error':str(e),'type':type(e).__name__,'exit_code':1});traceback.print_exc();sys.exit(1)
