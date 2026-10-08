"""Frozen few-shot selection/order experiment with official/independent parity."""
import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import sys
import time
import traceback
from pathlib import Path

from prepare import ROOT, digest, prepare

CHOICES = ['negative', 'positive']

def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str)+'\n')

def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()

def question(row):
    return row['sentence']+'\nQuestion: Is this sentence positive or negative?\nAnswer:'

def block(row):
    return question(row)+' '+CHOICES[row['label']]+'\n\n'

def normalized(text):
    return ' '.join(text.lower().split())

def data_rows(cache, lock):
    import pyarrow.parquet as pq
    base = cache/'glue'/lock['glue']['revision']/'sst2'
    return {s:pq.read_table(base/f'{s}-00000-of-00001.parquet').to_pylist() for s in ['train','validation']}

def construct_plan(data, tok, cfg):
    order = sorted(data['validation'], key=lambda r:sha('88:'+str(r['idx'])))
    excluded = {normalized(r['sentence']) for r in data['validation']}
    pools = {0:[], 1:[]}
    for row in sorted(data['train'], key=lambda r:sha('95:'+str(r['idx']))):
        key = normalized(row['sentence'])
        if key in excluded or len(pools[row['label']]) == 4:
            continue
        if len(tok.encode(block(row), add_special_tokens=False)) != cfg['demo_block_tokens']:
            continue
        pools[row['label']].append(row)
        excluded.add(key)
        if all(len(v)==4 for v in pools.values()):
            break
    assert all(len(v)==4 for v in pools.values()), 'Insufficient matched training demos'
    groups = {name:[pools[y][i] for i in indices for y in [0,1]] for name,indices in [('A',[0,1]),('B',[2,3])]}
    conditions = {'zero':[]}
    for name, rows in groups.items():
        conditions[name+'_forward'] = rows
        conditions[name+'_reverse'] = list(reversed(rows))
    lengths = {name:len(tok.encode(''.join(map(block,rows)),add_special_tokens=False)) for name,rows in conditions.items()}
    assert len(set(lengths[n] for n in lengths if n!='zero')) == 1
    demos = {str(r['idx']):{'idx':r['idx'],'label':r['label'],'sentence_sha256':sha(r['sentence']),'block_sha256':sha(block(r)),'block_tokens':len(tok.encode(block(r),add_special_tokens=False))} for rows in groups.values() for r in rows}
    return {'rule':'PROTOCOL.md; no model scores used','development':[r['idx'] for r in order[:64]],'reserved':[r['idx'] for r in order[64:192]],'conditions':{k:[r['idx'] for r in v] for k,v in conditions.items()},'demos':demos,'prefix_tokens':lengths,'train_rows':len(data['train']),'validation_rows':len(data['validation']),'exact_normalized_overlap':False,'confirmation_inference':False}

def independent_scores(model, tok, context):
    import torch
    ci = tok.encode(context, add_special_tokens=False)
    candidates=[]
    for choice in CHOICES:
        ids=tok.encode(context+' '+choice, add_special_tokens=False)
        assert ids[:len(ci)]==ci and len(ids)>len(ci)
        target=ids[len(ci):]
        with torch.inference_mode():
            output=model(torch.tensor([ids[:-1]]))
            lp=output.logits[0].log_softmax(-1)
        values=[float(lp[len(ci)-1+j,t]) for j,t in enumerate(target)]
        candidates.append({'continuation_ids':target,'token_logprobs':values,'score':sum(values),'input_tokens':len(ids)-1,'input_shape':[1,len(ids)-1],'logits_shape':list(output.logits.shape)})
    return candidates

def summarize(records):
    names=list(dict.fromkeys(r['condition'] for r in records))
    by={name:{r['idx']:r for r in records if r['condition']==name} for name in names}
    first=set(by[names[0]])
    assert all(set(v)==first for v in by.values())
    conditions={name:{'n':len(rows),'correct':sum(r['correct'] for r in rows.values()),'accuracy':sum(r['correct'] for r in rows.values())/len(rows),'positive_predictions':sum(r['prediction'] for r in rows.values()),'context_tokens_min':min(r['context_tokens'] for r in rows.values()),'context_tokens_max':max(r['context_tokens'] for r in rows.values()),'candidate_input_tokens':sum(c['input_tokens'] for r in rows.values() for c in r['candidates'])} for name,rows in by.items()}
    contrasts=[('zero',n) for n in names if n!='zero']+[('A_forward','A_reverse'),('B_forward','B_reverse'),('A_forward','B_forward'),('A_reverse','B_reverse')]
    paired={}
    for left,right in contrasts:
        changed=[i for i in sorted(first) if by[left][i]['prediction']!=by[right][i]['prediction']]
        gains=sum(by[left][i]['correct']==0 and by[right][i]['correct']==1 for i in first)
        losses=sum(by[left][i]['correct']==1 and by[right][i]['correct']==0 for i in first)
        paired[right+' minus '+left]={'n':len(first),'gains':gains,'losses':losses,'prediction_flips':len(changed),'changed_ids':changed,'accuracy_delta':(gains-losses)/len(first)}
    return {'conditions':conditions,'paired_contrasts':paired,'independent_units':len(first),'outputs':len(records),'selection_of_best':'none','heldout_inference':False}

def main(args):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    started=time.perf_counter()
    cfg=json.loads((ROOT/'config.json').read_text())
    lock=prepare(args.cache)
    data=data_rows(args.cache,lock)
    torch.set_num_threads(cfg['threads'])
    torch.manual_seed(cfg['seed'])
    tok=AutoTokenizer.from_pretrained(args.cache/'models'/lock['model']['revision'],local_files_only=True)
    plan=construct_plan(data,tok,cfg)
    plan_path=ROOT/'plan.json'
    if args.plan_only:
        if plan_path.exists():
            assert json.loads(plan_path.read_text())==plan
        else:
            save(plan_path,plan)
        print(json.dumps(plan,indent=2))
        return
    assert plan_path.exists() and json.loads(plan_path.read_text())==plan, 'Run --plan-only before inference'
    args.out.mkdir(parents=True,exist_ok=False)
    args.private.mkdir(parents=True,exist_ok=False)
    selected=plan['development'][:2] if args.smoke else plan['development']
    assert not set(selected)&set(plan['reserved'])
    train={r['idx']:r for r in data['train']}
    validation={r['idx']:r for r in data['validation']}
    identity={'plan_sha256':digest(plan_path),'resources':{k:v['sha256'] for k,v in lock['files'].items()},'code':{p.name:digest(p) for p in ROOT.glob('*.py')},'protocol_sha256':digest(ROOT/'PROTOCOL.md'),'config':cfg,'selected':selected,'conditions':plan['conditions'],'inference_cache':'disabled','harness_commit':lock['harness']['commit']}
    identity['key']=sha(json.dumps(identity,sort_keys=True))
    save(args.out/'identity.json',identity)
    save(args.out/'plan.json',plan)
    save(args.out/'stages.json',{'resource_hashes':'passed','plan_before_inference':'passed','inference':'started'})
    model=AutoModelForCausalLM.from_pretrained(args.cache/'models'/lock['model']['revision'],local_files_only=True,torch_dtype=torch.float32,attn_implementation='eager').eval()
    import datasets
    import lm_eval
    import yaml
    from lm_eval.api.samplers import ContextSampler
    from lm_eval.api.task import ConfigurableTask
    from lm_eval.models.huggingface import HFLM

    class FixedSampler(ContextSampler):
        def sample(self,n):
            chosen=[train[i] for i in self.task.config.metadata['demo_ids']]
            assert len(chosen)==n
            return chosen

    def local_dataset(**kwargs):
        return datasets.DatasetDict({s:datasets.Dataset.from_list(rows) for s,rows in data.items()})

    backend=HFLM(pretrained=model,tokenizer=tok,batch_size=1,max_length=cfg['max_length'],add_bos_token=False)
    all_records=[]
    timing={}
    context_lengths={}
    for name,demo_ids in plan['conditions'].items():
        stage=time.perf_counter()
        demos=[train[i] for i in demo_ids]
        task_cfg=yaml.safe_load((ROOT/'official_sst2.yaml').read_text())
        task_cfg.update(custom_dataset=local_dataset,fewshot_split='train',fewshot_config={'sampler':FixedSampler},metadata={'demo_ids':demo_ids})
        task=ConfigurableTask(config=task_cfg)
        contexts={i:''.join(map(block,demos))+question(validation[i]) for i in selected}
        for i,ctx in contexts.items():
            assert ctx==task.fewshot_context(validation[i],len(demos),apply_chat_template=False)
            lengths=[len(tok.encode(ctx+' '+choice,add_special_tokens=False)) for choice in CHOICES]
            assert max(lengths)<=cfg['max_length'], 'Refuse silent truncation'
            context_lengths[name,i]=len(tok.encode(ctx,add_special_tokens=False))
            if name not in ['zero','A_forward']:
                assert context_lengths[name,i]==context_lengths['A_forward',i]
        save(args.private/f'{name}_contexts.json',contexts)
        score_start=time.perf_counter()
        independent={i:independent_scores(model,tok,contexts[i]) for i in selected}
        independent_end=time.perf_counter()
        result=lm_eval.simple_evaluate(model=backend,tasks=[task],num_fewshot=len(demos),samples={'sst2':sorted(selected)},bootstrap_iters=0,log_samples=True,apply_chat_template=False,random_seed=95,numpy_random_seed=95,torch_random_seed=95,fewshot_random_seed=95)
        save(args.private/f'{name}_official_raw.json',result)
        samples=result['samples']['sst2']
        assert len(samples)==len(selected)
        for row in samples:
            idx=row['doc']['idx']
            assert row['doc_id']==idx and idx in selected
            assert row['target']==validation[idx]['label']
            assert all(arg[0]==contexts[idx] and arg[1]==' '+CHOICES[j] for j,arg in enumerate(row['arguments']))
            candidates=independent[idx]
            scores=[float(v[0][0]) for v in row['resps']]
            error=max(abs(scores[j]-candidates[j]['score']) for j in range(2))
            assert error<=2e-5
            pred=0 if scores[0]>=scores[1] else 1
            correct=int(pred==row['target'])
            assert correct==row['acc']
            all_records.append({'condition':name,'idx':idx,'gold':row['target'],'demo_ids':demo_ids,'context_sha256':sha(contexts[idx]),'context_tokens':context_lengths[name,idx],'candidates':candidates,'official_scores':scores,'max_score_error':error,'prediction':pred,'correct':correct})
        own=sum(r['correct'] for r in all_records if r['condition']==name)/len(selected)
        assert own==result['results']['sst2']['acc,none']
        timing[name]={'setup_seconds':score_start-stage,'independent_seconds':independent_end-score_start,'official_seconds':time.perf_counter()-independent_end,'seconds':time.perf_counter()-stage,'official_accuracy':own,'raw_sha256':digest(args.private/f'{name}_official_raw.json')}
        save(args.out/'records.json',all_records)
        print(json.dumps({'condition':name,'n':len(selected),'accuracy':own,'seconds':timing[name]['seconds']}),flush=True)
    save(args.out/'summary.json',summarize(all_records))
    elapsed=time.perf_counter()-started
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
    assert elapsed<cfg['walltime_budget_seconds'] and rss<cfg['rss_budget_bytes']
    estimate=sum(v['setup_seconds']+32*2*(v['independent_seconds']+v['official_seconds']) for v in timing.values()) if args.smoke else None
    save(args.out/'preflight.json',{'projected_full_seconds':estimate,'budget_seconds':cfg['walltime_budget_seconds'],'peak_rss_bytes':rss,'rss_budget_bytes':cfg['rss_budget_bytes'],'formula':'fixed setup once + 2x32x per-condition scoring/evaluation time; excludes imports/download','smoke':args.smoke})
    save(args.out/'environment.json',{'os':platform.platform(),'python':platform.python_version(),'torch':torch.__version__,'transformers':transformers.__version__,'harness':importlib.metadata.version('lm_eval'),'device':'cpu','dtype':'float32','threads':cfg['threads'],'elapsed_after_import_seconds':elapsed,'peak_rss_bytes':rss,'gpu_memory':'not measured; CPU only','timing':timing})
    if args.smoke:
        assert estimate<cfg['walltime_budget_seconds'], 'Full-run budget estimate exceeded'
    save(args.out/'stages.json',{'resource_hashes':'passed','plan_before_inference':'passed','inference':'passed','official_independent_score_parity':'passed','max_score_error':max(r['max_score_error'] for r in all_records),'equal_fewshot_token_budget':'passed','confirmation_inference':False,'exit_code':0})
    print(json.dumps({'status':'passed','outputs':len(all_records),'elapsed':elapsed,'rss':rss}))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--out',type=Path,default=ROOT/'results')
    parser.add_argument('--private',type=Path)
    parser.add_argument('--plan-only',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if not args.plan_only and args.private is None:
        parser.error('--private required for real inference (never distribute original data)')
    try:
        main(args)
    except Exception as error:
        if args.out.is_dir():
            save(args.out/'failure.json',{'error':str(error),'type':type(error).__name__,'exit_code':1})
        traceback.print_exc()
        sys.exit(1)
