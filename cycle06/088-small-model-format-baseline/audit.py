"""Recompute recorded results independently; verify source labels, prompts and token decoding."""
import argparse,collections,hashlib,json,pathlib,re
from prepare import ROOT,prepare

def main(cache,out):
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    lock=prepare(cache);cfg=json.loads((ROOT/'config.json').read_text())
    data=pq.read_table(cache/'sst2'/lock['data']['revision']/'data/validation-00000-of-00001.parquet').to_pylist()
    data=sorted(data,key=lambda x:hashlib.sha256(f'{cfg["seed"]}:{x["idx"]}'.encode()).hexdigest())
    split=json.loads((out/'split.json').read_text());assert split['development']==[x['idx'] for x in data[:64]]
    assert split['confirmation_reserved']==[x['idx'] for x in data[64:192]]
    byid={r['idx']:r for r in data};tok=AutoTokenizer.from_pretrained(cache/'models'/lock['model']['revision'],local_files_only=True)
    rows=[json.loads(l) for l in (out/'predictions.jsonl').read_text().splitlines()];smoke=json.loads((out/'identity.json').read_text())['smoke'];n=2 if smoke else 64
    assert len(rows)==2*n and len({(r['idx'],r['contract']) for r in rows})==2*n
    assert {r['idx'] for r in rows}==set(split['development'][:n]);assert not {r['idx'] for r in rows}&set(split['confirmation_reserved'])
    counts={c:collections.Counter() for c in ('plain','json')}
    for r in rows:
        source=byid[r['idx']];assert r['gold']==source['label'];assert r['sentence_sha256']==hashlib.sha256(source['sentence'].encode()).hexdigest()
        text=tok.decode(r['generated_ids'],skip_special_tokens=True);assert text==r['output']
        labels={w.lower() for w in re.findall(r'\b\w+\b',text) if w.lower() in ('positive','negative')}
        pred=next(iter(labels)) if len(labels)==1 else None;correct=pred==('negative','positive')[r['gold']]
        if r['contract']=='plain': fmt=text.strip() in ('positive','negative')
        else:
            try:
                pairs=json.loads(text,object_pairs_hook=lambda p:p)
                fmt=(isinstance(pairs,list) and len(pairs)==1 and isinstance(pairs[0],tuple) and pairs[0][0]=='sentiment' and pairs[0][1] in ('negative','positive'))
            except (ValueError,TypeError,IndexError):fmt=False
        assert bool(fmt)==r['format'] and correct==r['content_proxy'] and bool(fmt and correct)==r['joint']
        assert r['bucket']==f'F{int(bool(fmt))}C{int(correct)}'
        counts[r['contract']].update({'format_count':int(bool(fmt)),'correct_count':int(correct),'joint_count':int(bool(fmt and correct))})
        prompt=tok.apply_chat_template([{'role':'system','content':cfg['system']},{'role':'user','content':cfg['task']+'\n'+cfg['contracts'][r['contract']]+'\nReview: '+source['sentence']}],tokenize=False,add_generation_prompt=True)
        assert r['prompt_sha256']==hashlib.sha256(prompt.encode()).hexdigest();assert len(tok.encode(prompt,add_special_tokens=False))==r['input_length']
    summary=json.loads((out/'summary.json').read_text())
    for c in counts:
        assert summary[c]['n']==n
        for key,v in counts[c].items():assert summary[c][key]==v
        assert summary[c]['format_rate']==counts[c]['format_count']/n
        assert summary[c]['content_proxy_accuracy']==counts[c]['correct_count']/n
        assert summary[c]['joint_accuracy']==counts[c]['joint_count']/n
    from scoring import score
    fixtures=[('positive','plain',1,True,True),('Positive','plain',1,False,True),('{"sentiment":"negative"}','json',0,True,True),('{"sentiment":"negative","sentiment":"positive"}','json',1,False,False),('{"sentiment":"positive","extra":1}','json',1,False,True),('```json\n{"sentiment":"positive"}\n```','json',1,False,True),('positive or negative','plain',1,False,False),('neutral','plain',1,False,False)]
    for text,c,g,f,correct in fixtures:
        x=score(text,c,g);assert x['format']==f and x['content_proxy']==correct
    result={'status':'passed','rows':len(rows),'unique_pairs':len(rows),'reserved_ids_inferred':0,'count_errors':0,'fixtures':len(fixtures),'scope':'offline re-evaluation; no model generation'}
    print(json.dumps(result));return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,required=True);a=p.parse_args();main(a.cache,a.out)
