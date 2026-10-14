"""Real fixed-model ten-protocol development search, then frozen confirmation."""
import argparse,platform,resource,sys,time,traceback
from pathlib import Path
from common import ROOT,load,save,sha,digest,now,frozen_inputs,summarize
from prepare import prepare
from spec import specification,block

def score_direct(model,tok,ctx,choices):
    import torch
    prefix=tok.encode(ctx,add_special_tokens=False);out=[]
    for choice in choices:
        ids=tok.encode(ctx+' '+choice,add_special_tokens=False)
        assert ids[:len(prefix)]==prefix and len(ids)>len(prefix)
        target=ids[len(prefix):]
        with torch.inference_mode():
            logits=model(torch.tensor([ids[:-1]])).logits;lp=logits[0].log_softmax(-1)
        values=[float(lp[len(prefix)-1+j,t]) for j,t in enumerate(target)]
        out.append({'choice':choice,'continuation_ids':target,'token_logprobs':values,'score':sum(values),'input_tokens':len(ids)-1,'input_shape':[1,len(ids)-1],'logits_shape':list(logits.shape)})
    return out

def main(a):
    import torch,transformers,pyarrow.parquet as pq,datasets,yaml
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from lm_eval.api.instance import Instance
    from lm_eval.api.task import ConfigurableTask
    from lm_eval.models.huggingface import HFLM
    from importlib.metadata import version
    t=time.perf_counter();cfg=load(ROOT/'config.json');plan=load(ROOT/'plan.json');lock=prepare(a.cache)
    if a.phase=='confirmation':
        assert not a.smoke,'Confirmation smoke forbidden; use development'
        selection=load(ROOT/'selection.json');assert selection['frozen_inputs']==frozen_inputs()
        assert selection['confirmation_ids']==plan['reserved'] and not selection['confirmation_seen']
        names=list(dict.fromkeys([selection['baseline'],selection['winner']]));selected=plan['reserved']
    else:names=cfg['conditions'];selected=plan['development'][:2] if a.smoke else plan['development']
    assert not set(plan['development'])&set(plan['reserved'])
    a.out.mkdir(parents=True,exist_ok=False)
    identity={'started_at':now(),'phase':a.phase,'smoke':a.smoke,'selected':selected,'conditions':names,'frozen_inputs':frozen_inputs(),'resources':lock,'prompt_definition_sha256':digest(ROOT/'spec.py'),'decoding':'teacher-forced two candidates; no generation/chat/BOS/EOS; sum loglikelihood','cache':'disabled; all inference real','selection_sha256':digest(ROOT/'selection.json') if a.phase=='confirmation' else None}
    identity['cache_key']=sha(__import__('json').dumps(identity,sort_keys=True));save(a.out/'identity.json',identity)
    save(a.out/'stages.json',{'download':'cached hashes passed','preprocessing':'started','inference':'pending','evaluation':'pending'})
    base=a.cache/'glue'/lock['glue']['revision']/'sst2'
    data={s:pq.read_table(base/f'{s}-00000-of-00001.parquet').to_pylist() for s in ['train','validation']}
    ordered=sorted(data['validation'],key=lambda r:sha('88:'+str(r['idx'])))
    assert [r['idx'] for r in ordered[:64]]==plan['development'] and [r['idx'] for r in ordered[64:192]]==plan['reserved']
    train={r['idx']:r for r in data['train']};valid={r['idx']:r for r in data['validation']}
    torch.set_num_threads(cfg['threads']);torch.manual_seed(cfg['seed'])
    tok=AutoTokenizer.from_pretrained(a.cache/'models'/lock['model']['revision'],local_files_only=True)
    for i,d in plan['demos'].items():
        row=train[int(i)];assert sha(row['sentence'])==d['sentence_sha256'] and sha(block(row))==d['block_sha256']
        assert len(tok.encode(block(row),add_special_tokens=False))==d['block_tokens']
    model=AutoModelForCausalLM.from_pretrained(a.cache/'models'/lock['model']['revision'],local_files_only=True,torch_dtype=torch.float32,attn_implementation='eager').eval()
    backend=HFLM(pretrained=model,tokenizer=tok,batch_size=1,max_length=cfg['max_length'],add_bos_token=False)
    def local_dataset(**kwargs):return datasets.DatasetDict({s:datasets.Dataset.from_list(rows) for s,rows in data.items()})
    setup=time.perf_counter()-t;records=[];timings={}
    for name in names:
        ts=time.perf_counter();requests=[];direct={};contexts={}
        _,choices,mapping=specification(name,valid[selected[0]],train,plan)
        taskcfg=yaml.safe_load((ROOT/'official_sst2.yaml').read_text())
        taskcfg.update(custom_dataset=local_dataset,doc_to_choice=choices,doc_to_target=lambda doc,m=mapping:m.index(doc['label']),metric_list=[{'metric':'acc'}])
        task=ConfigurableTask(config=taskcfg)
        for idx in selected:
            ctx,choices,mapping=specification(name,valid[idx],train,plan);contexts[idx]=ctx
            assert max(len(tok.encode(ctx+' '+c,add_special_tokens=False)) for c in choices)<=cfg['max_length'],'No truncation allowed'
            if name=='zero':assert ctx==task.doc_to_text(valid[idx])
            for j,c in enumerate(choices):requests.append(Instance(request_type='loglikelihood',doc={'idx':idx},arguments=(ctx,' '+c),idx=j,metadata=('sst2_099',idx,1)))
            direct[idx]=score_direct(model,tok,ctx,choices)
        official=backend.loglikelihood(requests,disable_tqdm=True);assert len(official)==2*len(selected)
        for k,idx in enumerate(selected):
            row=valid[idx];ctx,choices,mapping=specification(name,row,train,plan);cs=direct[idx];response=official[2*k:2*k+2];scores=[float(v[0]) for v in response]
            err=max(abs(c['score']-v) for c,v in zip(cs,scores));assert err<=2e-5
            pos=max(range(2),key=lambda j:scores[j]);pred=mapping[pos];correct=int(pred==row['label']);metric=task.process_results(row,response)['acc'];assert metric==correct
            records.append({'condition':name,'idx':idx,'gold':row['label'],'mapping':mapping,'sentence_sha256':sha(row['sentence']),'context_sha256':sha(ctx),'context_tokens':len(tok.encode(ctx,add_special_tokens=False)),'candidates':cs,'official_scores':scores,'max_score_error':err,'prediction':pred,'correct':correct,'official_accuracy':metric})
        timings[name]=time.perf_counter()-ts;save(a.out/'records.json',records)
        print(__import__('json').dumps({'condition':name,'phase':a.phase,'n':len(selected),'correct':sum(r['correct'] for r in records if r['condition']==name),'seconds':timings[name]}),flush=True)
    elapsed=time.perf_counter()-t;rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
    projection=setup+64*sum(timings.values()) if a.smoke else None
    save(a.out/'summary.json',summarize(records))
    save(a.out/'environment.json',{'os':platform.platform(),'python':platform.python_version(),'torch':torch.__version__,'transformers':transformers.__version__,'harness':version('lm_eval'),'device':'cpu','dtype':'float32','threads':cfg['threads'],'seconds_after_imports':elapsed,'peak_rss_bytes':rss,'gpu_memory':'not measured; CPU','setup_seconds':setup,'condition_seconds':timings,'projected_development_seconds':projection,'projection_formula':'setup +2x32x sum of2-row ten-condition scoring times','first_shape':records[0]['candidates'][0]['logits_shape']})
    assert elapsed<cfg['walltime_budget_seconds'] and rss<cfg['rss_budget_bytes']
    if a.smoke:assert projection<cfg['walltime_budget_seconds'],'Projected budget exceeded'
    save(a.out/'stages.json',{'download':'cached hashes passed','preprocessing':'passed','inference':'passed','evaluation':'passed','max_score_error':max(r['max_score_error'] for r in records),'completed_at':now(),'exit_code':0})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--phase',choices=['development','confirmation'],default='development');p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    try:main(a)
    except Exception as e:
        if a.out.is_dir():save(a.out/'failure.json',{'type':type(e).__name__,'error':str(e),'exit_code':1})
        traceback.print_exc();sys.exit(1)
