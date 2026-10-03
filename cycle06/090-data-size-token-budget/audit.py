"""Independent output, split, mask, supervised-budget and paired-score audit."""
import argparse, collections, hashlib, json, pathlib, re
from prepare import ROOT, prepare, digest

def parse(t,k,g):
    words=set(re.findall(r'\b(?:positive|negative)\b',t.lower()));content=words=={('negative','positive')[g]}
    fmt=t.strip() in ['negative','positive'] if k=='plain' else False
    if k=='json':
        try:
            pairs=json.loads(t,object_pairs_hook=lambda x:x)
            fmt=isinstance(pairs,list) and len(pairs)==1 and pairs[0][0]=='sentiment' and pairs[0][1] in ['negative','positive']
        except (ValueError,TypeError,IndexError):fmt=False
    return bool(fmt),content,bool(fmt and content)
def main(a):
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    lock=prepare(a.cache);c=json.loads((ROOT/'config.json').read_text());p=a.out
    load=lambda n:json.loads((p/n).read_text());split=load('split.json');rows=load('rows.json');steps=load('steps.json');summary=load('summary.json');budget=load('budget.json');pred=[json.loads(x) for x in (p/'predictions.jsonl').read_text().splitlines()]
    dp=a.cache/'sst2'/lock['data']['revision']/'data';tr=pq.read_table(dp/'train-00000-of-00001.parquet').to_pylist();va=pq.read_table(dp/'validation-00000-of-00001.parquet').to_pylist()
    key=lambda r,s:hashlib.sha256(f'{s}:{r["idx"]}'.encode()).hexdigest();groups=[sorted([r for r in tr if r['label']==g],key=lambda r:key(r,89))[:16] for g in [0,1]];ids=[r['idx'] for pair in zip(*groups) for r in pair];assert ids==split['training_pool']
    vv=sorted(va,key=lambda r:key(r,88));dev=[r['idx'] for r in vv[:2 if split['smoke'] else 64]];assert dev==split['executed_development'];assert split['confirmation_reserved']==[r['idx'] for r in vv[64:192]];assert not set(dev)&set(split['confirmation_reserved'])
    source={r['idx']:r for r in tr};val={r['idx']:r for r in va};tok=AutoTokenizer.from_pretrained(a.cache/'models'/lock['model']['revision'],local_files_only=True)
    for row in rows:
        r=source[row['idx']];msgs=[{'role':'system','content':c['system']},{'role':'user','content':c['task']+'\n'+c['contracts']['json']+'\nReview: '+r['sentence']}]
        prefix=tok.apply_chat_template(msgs,tokenize=True,add_generation_prompt=True);target=json.dumps({'sentiment':('negative','positive')[r['label']]});full=tok.apply_chat_template(msgs+[{'role':'assistant','content':target}],tokenize=True);end=full.index(tok.eos_token_id,len(prefix));assert full[:len(prefix)]==prefix and end+1-len(prefix)==8 and row['length']==end+1 and row['prefix']==len(prefix)
    lens={r['idx']:r['length'] for r in rows}
    for arm,spec in c['arms'].items():
        exp=(ids[:spec['unique']]*spec['epochs'])[:4 if split['smoke'] else spec['unique']*spec['epochs']];rr=[x for x in steps if x['arm']==arm];assert [i for x in rr for i in x['idx']]==exp
        calculated={'unique_rows':len(set(exp)),'exposures':len(exp),'steps':len(exp)//2,'supervised_tokens':8*len(exp),'input_tokens':sum(lens[i] for i in exp),'padded_tokens':sum(2*max(lens[i],lens[j]) for i,j in zip(exp[::2],exp[1::2])),'attention_cells':sum(2*max(lens[i],lens[j])**2 for i,j in zip(exp[::2],exp[1::2]))};assert budget[arm]==calculated
        for x in rr:assert x['supervised_tokens']==16 and x['ce_error']<2e-6 and 0<x['gradient_norm']<float('inf')
    assert budget['repeat8']['supervised_tokens']==budget['unique32']['supervised_tokens']
    seen=set()
    for x in pred:
        z=(x['stage'],x['contract'],x['idx']);assert z not in seen;seen.add(z);assert x['idx'] in dev and x['gold']==val[x['idx']]['label'];assert tok.decode(x['generated_ids'],skip_special_tokens=True)==x['output']
        f,co,j=parse(x['output'],x['contract'],x['gold']);assert (f,co,j)==(x['format'],x['content_proxy'],x['joint'])
    for stage in ['before']+list(c['arms']):
        for contract in ['plain','json']:
            rr=[x for x in pred if x['stage']==stage and x['contract']==contract];assert len(rr)==len(dev)
            for key,name in [('format','format_count'),('content_proxy','correct_count'),('joint','joint_count')]:assert sum(x[key] for x in rr)==summary[stage][contract][name]
    pair={}
    for left,right in [('repeat8','unique32'),('before','repeat8_short'),('repeat8_short','repeat8')]:
        pair[left+'->'+right]={}
        for k in ['plain','json']:
            a1={x['idx']:x for x in pred if x['stage']==left and x['contract']==k};b1={x['idx']:x for x in pred if x['stage']==right and x['contract']==k}
            pair[left+'->'+right][k]={metric:{'wins':sum(b1[i][metric]>a1[i][metric] for i in dev),'ties':sum(b1[i][metric]==a1[i][metric] for i in dev),'losses':sum(b1[i][metric]<a1[i][metric] for i in dev)} for metric in ['format','content_proxy','joint']}
            pair[left+'->'+right][k]['different_output_ids']=[i for i in dev if a1[i]['output']!=b1[i]['output']]
    nll=load('train_nll.json');small=set(ids[:8]);nll_summary={st:{'small8':sum(x['mean'] for x in xx if x['idx'] in small)/8,'pool32':sum(x['mean'] for x in xx)/32,'new24':sum(x['mean'] for x in xx if x['idx'] not in small)/24} for st,xx in nll.items()}
    identity=load('identity.json');assert all(digest(ROOT/n)==h for n,h in identity['code_hashes'].items())
    result={'status':'passed','predictions_checked':len(pred),'steps_checked':len(steps),'rows_reencoded':len(rows),'equal_supervised_budget':True,'paired':pair,'train_nll':nll_summary,'plain_outputs':{s:dict(collections.Counter(x['output'] for x in pred if x['stage']==s and x['contract']=='plain')) for s in summary},'max_ce_error':max(x['ce_error'] for x in steps)}
    (p/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--cache',type=pathlib.Path,required=True);ap.add_argument('--out',type=pathlib.Path,required=True);main(ap.parse_args())
