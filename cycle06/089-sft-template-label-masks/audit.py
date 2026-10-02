"""Rebuild masks from public source data and independently check saved predictions."""
import argparse,hashlib,json,re,pathlib
from prepare import ROOT,prepare

def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main(a):
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    lock=prepare(a.cache);cfg=json.loads((ROOT/'config.json').read_text());split=json.loads((a.out/'split.json').read_text());tok=AutoTokenizer.from_pretrained(a.cache/'models'/lock['model']['revision'],local_files_only=True)
    dp=a.cache/'sst2'/lock['data']['revision']/'data';train={r['idx']:r for r in pq.read_table(dp/'train-00000-of-00001.parquet').to_pylist()};val={r['idx']:r for r in pq.read_table(dp/'validation-00000-of-00001.parquet').to_pylist()}
    masks=json.loads((a.out/'mask_audit.json').read_text());expected={}
    for audit in masks:
        r=train[audit['idx']];msgs=[{'role':'system','content':cfg['system']},{'role':'user','content':cfg['task']+'\n'+cfg['contracts']['json']+'\nReview: '+r['sentence']}]
        prefix=tok.apply_chat_template(msgs,tokenize=True,add_generation_prompt=True);target=json.dumps({'sentiment':['negative','positive'][r['label']]});ids=tok.apply_chat_template(msgs+[{'role':'assistant','content':target}],tokenize=True);e=ids.index(tok.eos_token_id,len(prefix));n=e+1
        assert audit['prefix']==len(prefix) and audit['eos']==e and audit['sequence_length']==n and ids[:len(prefix)]==prefix
        assert audit['template_tail_removed']==ids[n:]
        for j,p in enumerate(audit['positions']):
            assert p['position']==j and p['full_label_active']==(j>0) and p['assistant_label_active']==(j>=len(prefix))
        for q in audit['truncation']:
            retained=sum(j<q['limit'] for j in range(len(prefix),n));assert retained==q['retained_answer_targets'] and n-len(prefix)-retained==q['lost_targets'] and q['eos_retained']==(e<q['limit'])
        expected[r['idx']]=(n,len(prefix))
    steps=json.loads((a.out/'steps.json').read_text())
    for s in steps:
        rr=[expected[i] for i in s['idx']];count=sum(n-1 if s['arm']=='full' else n-pre for n,pre in rr)
        assert s['supervised_tokens']==count and s['input_tokens']==sum(n for n,pre in rr)
        assert s['padding_tokens']==len(rr)*max(n for n,pre in rr)-s['input_tokens']
        assert abs(s['loss']-s['manual_ce'])<2e-6 and s['ignored_logit_grad_max']==0 and s['prompt_embedding_grad_l1']>0
    preds=[json.loads(l) for l in (a.out/'predictions.jsonl').read_text().splitlines()];summary=json.loads((a.out/'summary.json').read_text());recomputed={};seen=set()
    def obj(pairs):
        assert len(set(k for k,v in pairs))==len(pairs)
        return dict(pairs)
    for r in preds:
        key=(r['stage'],r['contract'],r['idx']);assert key not in seen;seen.add(key)
        assert r['idx'] in split['executed_development'] and r['idx'] not in split['confirmation_reserved'];assert val[r['idx']]['label']==r['gold']
        text=tok.decode(r['generated_ids'],skip_special_tokens=True);assert text==r['output']
        words=set(re.findall(r'\b(?:positive|negative)\b',text.lower()));correct=len(words)==1 and next(iter(words))==['negative','positive'][r['gold']]
        if r['contract']=='plain':fmt=text.strip() in ['negative','positive']
        else:
            try:o=json.loads(text,object_pairs_hook=obj);fmt=isinstance(o,dict) and list(o)==['sentiment'] and o['sentiment'] in ['negative','positive']
            except (ValueError,AssertionError,TypeError):fmt=False
        assert fmt==r['format'] and correct==r['content_proxy'] and (fmt and correct)==r['joint']
        d=recomputed.setdefault(r['stage'],{}).setdefault(r['contract'],{'n':0,'format_count':0,'correct_count':0,'joint_count':0});d['n']+=1;d['format_count']+=fmt;d['correct_count']+=correct;d['joint_count']+=fmt and correct
    for st,contracts in recomputed.items():
        for k,d in contracts.items():
            for key,v in d.items():assert summary[st][k][key]==v
    report={'passed':True,'predictions':len(preds),'training_steps':len(steps),'mask_sequences':len(masks),'max_manual_loss_error':max(s['difference'] for s in steps),'supervised_tokens':{arm:sum(s['supervised_tokens'] for s in steps if s['arm']==arm) for arm in ('full','assistant')},'recomputed':recomputed,'scope':'independent masks, shifted-target counts, truncation, source labels, decoded outputs, F/C/J; gradient values checked from saved logs, real backward assertions in run.py'}
    save(a.out/'audit.json',report);print(json.dumps(report))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--cache',type=pathlib.Path,required=True);ap.add_argument('--out',type=pathlib.Path,required=True);main(ap.parse_args())
