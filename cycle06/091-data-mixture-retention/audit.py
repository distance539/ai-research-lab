"""Recompute outputs, class/split identity and per-update budget independently."""
import argparse,json,pathlib,re
from prepare import ROOT,prepare,digest
from common import sha,message,save
from mixture import select,news_message

def independent_score(text,contract,gold):
    if contract=='letter':
        labels={v for v in re.findall(r'\b[A-D]\b',text)};correct=labels=={'ABCD'[gold]};fmt=text.strip() in list('ABCD')
    else:
        labels=set(re.findall(r'\b(?:positive|negative)\b',text.lower()));correct=labels=={('negative','positive')[gold]};fmt=False
        if contract=='plain':fmt=text.strip() in ['negative','positive']
        else:
            try:
                pairs=json.loads(text,object_pairs_hook=lambda x:x);fmt=isinstance(pairs,list) and len(pairs)==1 and isinstance(pairs[0],tuple) and pairs[0][0]=='sentiment' and pairs[0][1] in ['negative','positive']
            except (ValueError,TypeError):pass
    return {'format':fmt,'content_proxy':correct,'joint':fmt and correct}

def main(a):
    from transformers import AutoTokenizer
    c=json.loads((ROOT/'config.json').read_text());lock=prepare(a.cache);tok=AutoTokenizer.from_pretrained(a.cache/'models'/lock['model']['revision'],local_files_only=True)
    spl=json.loads((a.out/'split.json').read_text());tr,dev,res,news,nd,nr=select(c,a.cache,lock)
    if spl['smoke']:dev=dev[:2];nd=nd[:4]
    for k,rs in [('target_train',tr),('target_dev',dev),('target_reserved',res),('news_train',news),('news_dev',nd),('news_reserved',nr)]:assert spl[k]==[r['idx'] for r in rs]
    raw={('target',r['idx']):r for r in tr};raw.update({('news',r['idx']):r for r in news});encoded={}
    for r in json.loads((a.out/'rows.json').read_text()):
        rr=raw[r['task'],r['idx']];msgs=message(c,'json',rr['sentence']) if r['task']=='target' else news_message(c,rr['text']);target=json.dumps({'sentiment':('negative','positive')[rr['label']]}) if r['task']=='target' else 'ABCD'[rr['label']]
        prefix=tok.apply_chat_template(msgs,tokenize=True,add_generation_prompt=True);full=tok.apply_chat_template(msgs+[{'role':'assistant','content':target}],tokenize=True);eos=full.index(tok.eos_token_id,len(prefix));ids=full[:eos+1]
        assert full[:len(prefix)]==prefix and len(ids)==r['length'] and len(prefix)==r['prefix'] and r['gold']==rr['label'] and eos==r['eos'] and eos-len(prefix)+1==r['answer_tokens'];encoded[r['task'],r['idx']]=ids
    steps=json.loads((a.out/'steps.json').read_text());bud=json.loads((a.out/'budget.json').read_text());maxerr=0
    for arm,spec in c['arms'].items():
        rr=[r for r in steps if r['arm']==arm];expected=(2 if arm=='target_half' else 4) if spl['smoke'] else spec['steps'];assert len(rr)==expected;cur={'target':0,'news':0}
        for i,r in enumerate(rr):
            task=spec['pattern'][i%len(spec['pattern'])];pool=tr if task=='target' else news;count=2 if task=='target' else 8;idx=[pool[(cur[task]+j)%len(pool)]['idx'] for j in range(count)];cur[task]+=count
            assert r['task']==task and r['idx']==idx and r['step']==i+1 and r['supervised_tokens']==16
            for j,m in enumerate(r['microbatches']):
                mids=idx[j*2:j*2+2];lens=[len(encoded[task,x]) for x in mids];L=max(lens)
                assert m['idx']==mids and m['input_tokens']==sum(lens) and m['padded_tokens']==2*L and m['attention_cells']==2*L*L and m['input_shape']==[2,L] and m['logits_shape']==[2,L,49152] and m['supervised_tokens']==(16 if task=='target' else 4)
                maxerr=max(maxerr,m['ce_error']);assert m['ce_error']<2e-6
            assert abs(r['loss']-sum(m['loss']*m['supervised_tokens']/16 for m in r['microbatches']))<2e-6
        for k in ['supervised_tokens','input_tokens','padded_tokens','attention_cells']:assert bud[arm][k]==sum(r[k] for r in rr)
    pred=[json.loads(x) for x in (a.out/'predictions.jsonl').read_text().splitlines()];summ=json.loads((a.out/'summary.json').read_text());data={('target',r['idx']):r for r in dev};data.update({('news',r['idx']):r for r in nd})
    expected_keys={(s,t,k,r['idx']) for s in ['before']+list(c['arms']) for t,rs,ks in [('target',dev,['plain','json']),('news',nd,['letter'])] for r in rs for k in ks};assert len(pred)==len(expected_keys) and {(r['stage'],r['task'],r['contract'],r['idx']) for r in pred}==expected_keys
    for r in pred:
        row=data[r['task'],r['idx']];assert r['gold']==row['label'];msgs=message(c,r['contract'],row['sentence']) if r['task']=='target' else news_message(c,row['text']);prompt=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
        assert sha(prompt)==r['prompt_sha256'] and len(tok(prompt,add_special_tokens=False).input_ids)==r['input_length'];
        if 'redacted' not in r:
            assert tok.decode(r['generated_ids'],skip_special_tokens=True)==r['output'];got=independent_score(r['output'],r['contract'],r['gold']);assert all(r[k]==v for k,v in got.items())
        else:
            assert r['task']=='news' and len(r['output_sha256'])==64 and len(r['generated_ids_sha256'])==64 and r['generated_token_count']<=32
    for stage,s in summ.items():
        for contract in ['plain','json','letter']:
            rr=[r for r in pred if r['stage']==stage and r['contract']==contract];key='news' if contract=='letter' else contract;q=s[key];assert q['n']==len(rr)
            for metric,count in [('format','format_count'),('joint','joint_count'),('content_proxy','content_proxy_count' if contract=='letter' else 'correct_count')]:assert q[count]==sum((r[metric] if 'redacted' in r else independent_score(r['output'],contract,r['gold'])[metric]) for r in rr)
    ck=json.loads((a.out/'checkpoints.json').read_text());assert len(ck)==4 and all(x['q_proj_max_abs_change']>0 and x['parameters']==134515008 and x['bytes']>530000000 for x in ck)
    identity=json.loads((a.out/'identity.json').read_text());key=identity.pop('cache_key');assert key==sha(json.dumps(identity,sort_keys=True));assert all(digest(ROOT/n)==h for n,h in identity['code_hashes'].items())
    eligible=[x for x in ['target_only','replay25','replay50'] if summ[x]['news']['joint_count']>=summ['before']['news']['joint_count']-1];selected=max(eligible,key=lambda x:summ[x]['json']['joint_count']) if eligible else None;assert json.loads((a.out/'selection.json').read_text())['selected_development_only']==selected
    
    if (a.out/'distribution.json').exists():
        dist=json.loads((a.out/'distribution.json').read_text());assert dist['public_predictions_sha256']==digest(a.out/'predictions.jsonl');assert dist['redacted_count']==sum('redacted' in r for r in pred)
    # Independently reconstruct paired transitions from saved per-row flags.
    pairs=json.loads((a.out/'paired.json').read_text());base={(r['task'],r['contract'],r['idx']):r for r in pred if r['stage']=='before'}
    for arm in c['arms']:
        for task,k in [('target','plain'),('target','json'),('news','letter')]:
            rr=[r for r in pred if (r['stage'],r['task'],r['contract'])==(arm,task,k)]
            for metric in ['format','content_proxy','joint']:
                delta=[int(r[metric])-int(base[task,k,r['idx']][metric]) for r in rr];assert pairs[arm][task+'_'+k][metric]=={'win':sum(x>0 for x in delta),'tie':sum(x==0 for x in delta),'loss':sum(x<0 for x in delta)}
    result={'status':'passed','predictions':len(pred),'steps':len(steps),'training_rows_reencoded':len(encoded),'fully_reparsed_predictions':sum('redacted' not in r for r in pred),'redacted_predictions':sum('redacted' in r for r in pred),'max_manual_ce_error':maxerr,'all_main_supervised_tokens_equal':len({bud[x]['supervised_tokens'] for x in ['target_only','replay25','replay50']})==1,'confirmation_inference':False,'source_hashes_unchanged':True};save(a.out/'audit.json',result);print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,required=True);main(p.parse_args())
