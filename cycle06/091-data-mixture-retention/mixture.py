"""Independent task selection, token-budget scheduling and retention scoring."""
import json,re,time
from common import ordered,sha,build,collate
from scoring import aggregate

def news_message(c,text):
    return [{'role':'system','content':c['system']},{'role':'user','content':c['news_task']+'\n'+c['news_contract']+'\nArticle: '+text}]

def news_build(tok,c,row):
    msgs=news_message(c,row['text']);prefix=tok.apply_chat_template(msgs,tokenize=True,add_generation_prompt=True)
    target='ABCD'[row['label']];full=tok.apply_chat_template(msgs+[{'role':'assistant','content':target}],tokenize=True,add_generation_prompt=False)
    assert full[:len(prefix)]==prefix
    eos=full.index(tok.eos_token_id,len(prefix));ids=full[:eos+1]
    assert tok.decode(ids[len(prefix):eos])==target and len(ids)-len(prefix)==2 and len(ids)<=c['news_max_train_tokens']
    return {'task':'news','idx':row['idx'],'gold':row['label'],'ids':ids,'prefix':len(prefix),'eos':eos,'text_sha256':sha(row['text']),'template_tail_removed':full[eos+1:]}

def select(c,cache,lock):
    import pyarrow.parquet as pq
    dp=cache/'sst2'/lock['data']['revision']/'data';tr=pq.read_table(dp/'train-00000-of-00001.parquet').to_pylist()
    groups=[ordered([r for r in tr if r['label']==g],c['train_seed'])[:16] for g in (0,1)]
    target=[r for pair in zip(*groups) for r in pair]
    va=ordered(pq.read_table(dp/'validation-00000-of-00001.parquet').to_pylist(),c['split_seed']);dev=va[:64];reserved=va[64:192]
    nd=cache/'ag_news'/lock['news']['revision']/'data'
    def interleave(rows,n,start=0):
        groups=[ordered([r for r in rows if r['label']==g],c['news_seed'])[start:start+n] for g in range(4)]
        return [r for group in zip(*groups) for r in group]
    raw={s:[dict(r,idx=i) for i,r in enumerate(pq.read_table(nd/f'{s}-00000-of-00001.parquet').to_pylist())] for s in ('train','test')}
    news=interleave(raw['train'],8);ndev=interleave(raw['test'],8);nreserve=interleave(raw['test'],16,8)
    norm=lambda s:' '.join(s.lower().split())
    for pool,held,key in [(target,dev+reserved,'sentence'),(news,ndev+nreserve,'text')]:
        texts={norm(r[key]) for r in pool};assert len(texts)==len(pool);assert not texts&{norm(r[key]) for r in held}
    assert len({norm(r['text']) for r in ndev+nreserve})==96
    return target,dev,reserved,news,ndev,nreserve

def schedule(c,target,news,smoke):
    result={}
    for arm,spec in c['arms'].items():
        cursors={'target':0,'news':0};batches=[];steps=(2 if arm=='target_half' else 4) if smoke else spec['steps']
        for i in range(steps):
            task=spec['pattern'][i%len(spec['pattern'])];pool=target if task=='target' else news;n=2 if task=='target' else 8
            rr=[pool[(cursors[task]+j)%len(pool)] for j in range(n)];cursors[task]+=n
            assert sum(len(r['ids'])-r['prefix'] for r in rr)==16
            batches.append((task,rr))
        result[arm]=batches
    return result

def news_score(text,gold):
    labels=set(re.findall(r'\b[A-D]\b',text));ext=next(iter(labels)) if len(labels)==1 else None
    fmt=text.strip() in 'ABCD' and len(text.strip())==1;cor=ext=='ABCD'[gold]
    return {'extracted':ext,'format':fmt,'content_proxy':cor,'joint':fmt and cor}

def news_evaluate(model,tok,c,rows,stage):
    import torch
    from transformers import GenerationConfig
    gen=GenerationConfig(**c['decoding'],eos_token_id=tok.eos_token_id,pad_token_id=tok.pad_token_id,bos_token_id=tok.bos_token_id)
    records=[]
    with torch.inference_mode():
        for r in rows:
            text=tok.apply_chat_template(news_message(c,r['text']),tokenize=False,add_generation_prompt=True)
            x=tok(text,add_special_tokens=False,return_tensors='pt');assert x.input_ids.shape[1]<=c['max_input_tokens']
            t=time.perf_counter();y=model.generate(**x,generation_config=gen)[0,x.input_ids.shape[1]:].tolist();out=tok.decode(y,skip_special_tokens=True)
            records.append({'stage':stage,'task':'news','idx':r['idx'],'gold':r['label'],'contract':'letter','prompt_sha256':sha(text),'input_length':x.input_ids.shape[1],'generated_ids':y,'output':out,'stop':'eos' if y[-1]==tok.eos_token_id else 'length','seconds':time.perf_counter()-t,**news_score(out,r['label'])})
    return records

def summarize(rows):
    s=aggregate([r for r in rows if r['task']=='target']);nr=[r for r in rows if r['task']=='news'];n=len(nr)
    s['news']={'n':n,**{k+'_count':sum(r[k] for r in nr) for k in ('format','content_proxy','joint')},'input_tokens':sum(r['input_length'] for r in nr),'generated_tokens':sum(len(r['generated_ids']) for r in nr)}
    return s

def paired(allrows):
    out={};before={(r['task'],r['contract'],r['idx']):r for r in allrows if r['stage']=='before'}
    for stage in dict.fromkeys(r['stage'] for r in allrows if r['stage']!='before'):
        out[stage]={}
        for task,contract in [('target','plain'),('target','json'),('news','letter')]:
            rr=[r for r in allrows if (r['stage'],r['task'],r['contract'])==(stage,task,contract)]
            out[stage][task+'_'+contract]={k:{'win':sum(r[k]>before[(task,contract,r['idx'])][k] for r in rr),'tie':sum(r[k]==before[(task,contract,r['idx'])][k] for r in rr),'loss':sum(r[k]<before[(task,contract,r['idx'])][k] for r in rr)} for k in ('format','content_proxy','joint')}
    return out
