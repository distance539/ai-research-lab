"""Frozen answer-space/order experiment; real HF and official harness scoring."""
import argparse, hashlib, json, platform, resource, sys, time, traceback
from pathlib import Path
from prepare import ROOT, digest, prepare

def save(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str)+'\n')

def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()

def spec(name, row):
    mapping = [1,0] if name.endswith('reverse') else [0,1]
    words = ['negative','positive']
    ctx = row['sentence']+'\nQuestion: Is this sentence positive or negative?\n'
    if name != 'official_text':
        ctx += '\n'.join(f'{letter}. {words[y]}' for letter,y in zip('AB',mapping))+'\n'
    ctx += 'Answer:'
    choices = list('AB') if 'letter' in name else [words[y] for y in mapping]
    return ctx, choices, mapping

def score_direct(model, tok, ctx, choices):
    import torch
    prefix = tok.encode(ctx,add_special_tokens=False)
    out=[]
    for choice in choices:
        ids=tok.encode(ctx+' '+choice,add_special_tokens=False)
        assert ids[:len(prefix)]==prefix and len(ids)>len(prefix)
        target=ids[len(prefix):]
        with torch.inference_mode():
            logits=model(torch.tensor([ids[:-1]])).logits
            lp=logits[0].log_softmax(-1)
        values=[float(lp[len(prefix)-1+j,t]) for j,t in enumerate(target)]
        out.append({'choice':choice,'continuation_ids':target,'token_logprobs':values,'score':sum(values),'input_tokens':len(ids)-1,'input_shape':[1,len(ids)-1],'logits_shape':list(logits.shape)})
    return out

def predictions(candidates, mapping):
    values={'sum':[c['score'] for c in candidates],
            'token_mean':[c['score']/len(c['continuation_ids']) for c in candidates],
            'character_mean':[c['score']/len(c['choice']) for c in candidates]}
    positions={k:max(range(2),key=lambda j:v[j]) for k,v in values.items()}
    return {k:{'scores':values[k],'position':p,'semantic':mapping[p]} for k,p in positions.items()}

def summarize(rows):
    groups={n:{r['idx']:r for r in rows if r['condition']==n} for n in dict.fromkeys(r['condition'] for r in rows)}
    conditions={}
    for n,g in groups.items():
        conditions[n]={'n':len(g),'correct':{m:sum(r['predictions'][m]['semantic']==r['gold'] for r in g.values()) for m in ['sum','token_mean','character_mean']},'positive':sum(r['predictions']['sum']['semantic'] for r in g.values()),'first_position':sum(r['predictions']['sum']['position']==0 for r in g.values()),'context_tokens_range':[min(r['context_tokens'] for r in g.values()),max(r['context_tokens'] for r in g.values())],'candidate_input_tokens':sum(c['input_tokens'] for r in g.values() for c in r['candidates']),'continuation_lengths':sorted(set(len(c['continuation_ids']) for r in g.values() for c in r['candidates']))}
    pairs={}
    contrasts=[('listed_text_forward','listed_text_reverse'),('listed_letter_forward','listed_letter_reverse'),('listed_text_forward','listed_letter_forward'),('listed_text_reverse','listed_letter_reverse'),('official_text','listed_text_forward')]
    for a,b in contrasts:
        assert groups[a].keys()==groups[b].keys()
        gains=[];losses=[];flips=[]
        for i,x in groups[a].items():
            y=groups[b][i];xp=x['predictions']['sum']['semantic'];yp=y['predictions']['sum']['semantic'];gold=x['gold']
            if xp!=yp: flips.append(i)
            if xp!=gold and yp==gold:gains.append(i)
            if xp==gold and yp!=gold:losses.append(i)
        pairs[b+' minus '+a]={'gains':gains,'losses':losses,'flips':flips,'delta':(len(gains)-len(losses))/len(groups[a])}
    return {'conditions':conditions,'pairs':pairs,'independent_questions':len(next(iter(groups.values()))),'rows':len(rows),'candidate_scores':2*len(rows),'confirmation_inference':False}

def main(a):
    import torch, transformers, pyarrow.parquet as pq, datasets, yaml
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from lm_eval.api.instance import Instance
    from lm_eval.api.task import ConfigurableTask
    from lm_eval.models.huggingface import HFLM
    from importlib.metadata import version
    started=time.perf_counter()
    cfg=json.loads((ROOT/'config.json').read_text());lock=prepare(a.cache)
    plan=json.loads((ROOT/'plan.json').read_text())
    base=a.cache/'glue'/lock['glue']['revision']/'sst2'
    data={s:pq.read_table(base/f'{s}-00000-of-00001.parquet').to_pylist() for s in ['train','validation']}
    ordered=sorted(data['validation'],key=lambda r:sha('88:'+str(r['idx'])))
    assert [r['idx'] for r in ordered[:64]]==plan['development']
    assert [r['idx'] for r in ordered[64:192]]==plan['reserved']
    selected=plan['development'][:2] if a.smoke else plan['development']
    assert not set(selected)&set(plan['reserved'])
    a.out.mkdir(parents=True,exist_ok=False)
    identity={'config':cfg,'plan_sha256':digest(ROOT/'plan.json'),'protocol_sha256':digest(ROOT/'PROTOCOL.md'),'code':{p.name:digest(p) for p in ROOT.glob('*.py')},'resources':{k:v['sha256'] for k,v in lock['files'].items()},'selected':selected,'harness_commit':lock['harness']['commit'],'inference_cache':'disabled','decode':'none; teacher-forced candidate likelihood'}
    identity['cache_key']=sha(json.dumps(identity,sort_keys=True));save(a.out/'identity.json',identity)
    save(a.out/'stages.json',{'download':'cached resources verified','preprocess':'passed','inference':'started','evaluation':'pending'})
    torch.set_num_threads(cfg['threads']);torch.manual_seed(cfg['seed'])
    tok=AutoTokenizer.from_pretrained(a.cache/'models'/lock['model']['revision'],local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(a.cache/'models'/lock['model']['revision'],local_files_only=True,torch_dtype=torch.float32,attn_implementation='eager').eval()
    backend=HFLM(pretrained=model,tokenizer=tok,batch_size=1,max_length=cfg['max_length'],add_bos_token=False)
    def local_dataset(**kwargs):
        return datasets.DatasetDict({s:datasets.Dataset.from_list(v) for s,v in data.items()})
    records=[];timings={};lookup={r['idx']:r for r in data['validation']}
    setup=time.perf_counter()-started
    for name in plan['conditions']:
        stage=time.perf_counter();requests=[];direct={};contexts={}
        template=yaml.safe_load((ROOT/'official_sst2.yaml').read_text())
        choices=spec(name,lookup[selected[0]])[1]
        option_mapping=spec(name,lookup[selected[0]])[2]
        template.update(custom_dataset=local_dataset,doc_to_choice=choices,doc_to_target=lambda doc,m=option_mapping:m.index(doc['label']),metric_list=[{'metric':'acc'},{'metric':'acc_norm'}])
        task=ConfigurableTask(config=template)
        for idx in selected:
            row=lookup[idx];ctx,choices,mapping=spec(name,row);contexts[idx]=ctx
            assert max(len(tok.encode(ctx+' '+c,add_special_tokens=False)) for c in choices)<=cfg['max_length']
            if name=='official_text':assert ctx==task.doc_to_text(row)
            for j,c in enumerate(choices):
                requests.append(Instance(request_type='loglikelihood',doc={'idx':idx},arguments=(ctx,' '+c),idx=j,metadata=('sst2_096',idx,1)))
            direct[idx]=score_direct(model,tok,ctx,choices)
        official=backend.loglikelihood(requests,disable_tqdm=True)
        assert len(official)==2*len(selected)
        for k,idx in enumerate(selected):
            row=lookup[idx];ctx,choices,mapping=spec(name,row);cs=direct[idx];response=official[2*k:2*k+2];scores=[float(x[0]) for x in response]
            error=max(abs(x-c['score']) for x,c in zip(scores,cs));assert error<=2e-5
            pred=predictions(cs,mapping);position_gold=mapping.index(row['label'])
            doc={**row,'position_gold':position_gold};metrics=task.process_results(doc,response)
            assert metrics['acc']==int(pred['sum']['semantic']==row['label'])
            assert metrics['acc_norm']==int(pred['character_mean']['semantic']==row['label'])
            records.append({'condition':name,'idx':idx,'gold':row['label'],'mapping':mapping,'position_gold':position_gold,'sentence_sha256':sha(row['sentence']),'context_sha256':sha(ctx),'context_tokens':len(tok.encode(ctx,add_special_tokens=False)),'candidates':cs,'official_scores':scores,'max_score_error':error,'official_metrics':metrics,'predictions':pred})
        timings[name]=time.perf_counter()-stage
        save(a.out/'records.json',records)
        print(json.dumps({'condition':name,'n':len(selected),'seconds':timings[name],'correct':sum(r['predictions']['sum']['semantic']==r['gold'] for r in records if r['condition']==name)}),flush=True)
    elapsed=time.perf_counter()-started;rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
    projected=setup+64*sum(timings.values()) if a.smoke else None
    save(a.out/'summary.json',summarize(records))
    save(a.out/'environment.json',{'os':platform.platform(),'python':platform.python_version(),'torch':torch.__version__,'transformers':transformers.__version__,'harness':version('lm_eval'),'device':'cpu','dtype':'float32','threads':cfg['threads'],'elapsed_after_import_seconds':elapsed,'peak_rss_bytes':rss,'gpu_memory':'not measured; CPU only','setup_seconds':setup,'condition_seconds':timings,'projected_full_seconds':projected})
    assert elapsed<cfg['walltime_budget_seconds'] and rss<cfg['rss_budget_bytes']
    if a.smoke:assert projected<cfg['walltime_budget_seconds'],'Projected budget exceeded'
    save(a.out/'stages.json',{'download':'cached resources verified','preprocess':'passed','inference':'passed','evaluation':'passed','max_score_error':max(r['max_score_error'] for r in records),'confirmation_inference':False,'exit_code':0})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    try:main(a)
    except Exception as e:
        if a.out.is_dir():save(a.out/'failure.json',{'error':str(e),'type':type(e).__name__,'exit_code':1})
        traceback.print_exc();sys.exit(1)
