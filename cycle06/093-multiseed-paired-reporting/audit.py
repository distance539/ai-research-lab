"""Audit saved experiment without training; independently parse scores/budgets/pairs."""
import argparse,json,pathlib,re,collections,math
from prepare import prepare,digest,ROOT
from common import build,message,sha,save
from mixture import select,news_message
from analysis import analyze

def parse(text,contract,gold):
    if contract=='letter':
        labels=set(re.findall(r'\b[A-D]\b',text));correct=len(labels)==1 and next(iter(labels))=='ABCD'[gold];fmt=len(text.strip())==1 and text.strip() in 'ABCD'
    else:
        labels=set(re.findall(r'\b(?:positive|negative)\b',text.lower()));correct=labels=={('negative','positive')[gold]}
        if contract=='plain':fmt=text.strip() in ('negative','positive')
        else:
            try:
                pairs=json.loads(text,object_pairs_hook=lambda p:p);fmt=isinstance(pairs,list) and len(pairs)==1 and pairs[0][0]=='sentiment' and pairs[0][1] in ('negative','positive')
            except (ValueError,TypeError,IndexError):fmt=False
    return {'format':bool(fmt),'content_proxy':bool(correct),'joint':bool(fmt and correct)}

def main(a):
    from transformers import AutoTokenizer
    c=json.loads((ROOT/'config.json').read_text());lock=prepare(a.cache);tok=AutoTokenizer.from_pretrained(a.cache/'models'/lock['model']['revision'],local_files_only=True)
    tr,dev,res,news,ndev,nres=select(c,a.cache,lock);sp=json.loads((a.out/'split.json').read_text());smoke=sp['smoke'];dev=dev[:2] if smoke else dev;ndev=ndev[:4] if smoke else ndev
    assert sp['target_train']==[r['idx'] for r in tr] and sp['target_dev']==[r['idx'] for r in dev] and sp['news_dev']==[r['idx'] for r in ndev]
    assert sp['target_reserved']==[r['idx'] for r in res] and sp['news_reserved']==[r['idx'] for r in nres]
    train={r['idx']:build(tok,c,r) for r in tr};rows=[json.loads(x) for x in (a.out/'predictions.jsonl').read_text().splitlines()];sources={('target',r['idx']):r for r in dev}|{('news',r['idx']):r for r in ndev};keys=set();red=0
    for r in rows:
        key=(r['stage'],r['task'],r['contract'],r['idx']);assert key not in keys;keys.add(key);source=sources[(r['task'],r['idx'])];assert r['gold']==source['label']
        msg=message(c,r['contract'],source['sentence']) if r['task']=='target' else news_message(c,source['text']);prompt=tok.apply_chat_template(msg,tokenize=False,add_generation_prompt=True);assert sha(prompt)==r['prompt_sha256'];assert len(tok(prompt,add_special_tokens=False)['input_ids'])==r['input_length']
        if 'output' not in r:assert 'output_sha256' in r and 'redacted' in r;red+=1;continue
        assert tok.decode(r['generated_ids'],skip_special_tokens=True)==r['output'];assert r['stop']==('eos' if r['generated_ids'][-1]==tok.eos_token_id else 'length')
        s=parse(r['output'],r['contract'],r['gold']);assert all(r[k]==v for k,v in s.items())
    assert len(rows)==6*(2*len(dev)+len(ndev))
    steps=json.loads((a.out/'steps.json').read_text());budget=json.loads((a.out/'budget.json').read_text());checkpoints=json.loads((a.out/'checkpoints.json').read_text())
    for arm in c['arms']:
        ss=[s for s in steps if s['arm']==arm];assert len(ss)==(2 if smoke else 64)
        for i,s in enumerate(ss):
            rr=[train[idx] for idx in s['idx']];assert s['step']==i+1 and [r['gold'] for r in rr]==[0,1]
            length=max(len(r['ids']) for r in rr);assert s['supervised_tokens']==16 and s['input_tokens']==sum(len(r['ids']) for r in rr) and s['padded_tokens']==2*length and s['attention_cells']==2*length**2
            assert math.isfinite(s['loss']) and s['gradient_norm']>0
            assert s['microbatches'][0]['input_shape']==[2,length] and s['microbatches'][0]['logits_shape']==[2,length,49152] and s['microbatches'][0]['ce_error']<2e-6
        if not smoke:assert set(collections.Counter(idx for s in ss for idx in s['idx']).values())=={4}
        for k in ['supervised_tokens','input_tokens','padded_tokens','attention_cells']:assert sum(s[k] for s in ss)==budget[arm][k]
    ck={r['arm']:r for r in checkpoints};assert ck['fixed93']['state_sha256']==ck['fixed94']['state_sha256']
    def canonical(stage):return [{k:v for k,v in r.items() if k not in ['stage','seconds']} for r in rows if r['stage']==stage]
    assert canonical('fixed93')==canonical('fixed94')
    for metric in ['format','content_proxy','joint']:
        for contract in ['plain','json','letter']:
            base={r['idx']:r for r in rows if r['stage']=='before' and r['contract']==contract}
            for arm in c['arms']:
                arr=[r for r in rows if r['stage']==arm and r['contract']==contract];d=[int(r[metric])-int(base[r['idx']][metric]) for r in arr];s=json.loads((a.out/'statistics.json').read_text())['results'][contract][metric]['arms'][arm]
                assert abs(sum(d)/len(d)-s['delta'])<1e-15 and sum(x>0 for x in d)==s['win'] and sum(x<0 for x in d)==s['loss']
    assert analyze(rows,c)==json.loads((a.out/'statistics.json').read_text())
    assert parse('{"sentiment":"positive","sentiment":"negative"}','json',1)['format']==False
    assert parse('positive and negative','plain',1)['content_proxy']==False
    result={'status':'passed','predictions':len(rows),'fully_reparsed':len(rows)-red,'redacted_not_reparsed':red,'training_steps':len(steps),'resources':len(lock['files']),'fixed_control_state_and_outputs_equal':True,'three_shuffled_states_distinct':len({ck[f'shuffle{s}']['state_sha256'] for s in [93,94,95]})==3,'max_manual_ce_error':max(m['ce_error'] for s in steps for m in s['microbatches']),'paired_delta_checks':'independent arithmetic passed','bootstrap_replay':'same frozen implementation deterministic replay passed','bootstrap_limit':'not an independent coverage proof','confirmation_inference':False,'predictions_sha256':digest(a.out/'predictions.jsonl')};save(a.out/'audit.json',result);print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,required=True);main(p.parse_args())
