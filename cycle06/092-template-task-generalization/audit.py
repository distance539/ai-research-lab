"""Independent parsing, pairing, identity and original-data reconstruction audit."""
import argparse,json,pathlib,re
from prepare import ROOT,prepare,digest
from common import sha,save
from mixture import select
from templates import messages

def parse(text,template,gold):
    news=template.startswith('news');labels=set(re.findall(r'\b[A-D]\b',text)) if news else set(re.findall(r'\b(?:negative|positive)\b',text.lower()));ex=next(iter(labels)) if len(labels)==1 else None
    expected='ABCD'[gold] if news else ('negative','positive')[gold]
    if news:fmt=text.strip() in ['A','B','C','D']
    elif template=='plain_original':fmt=text.strip() in ['negative','positive']
    else:
        try:
            pairs=json.loads(text,object_pairs_hook=lambda x:x);fmt=isinstance(pairs,list) and len(pairs)==1 and isinstance(pairs[0],tuple) and pairs[0][0]=='sentiment' and pairs[0][1] in ['negative','positive']
        except (ValueError,TypeError):fmt=False
    return {'format':bool(fmt),'content_proxy':ex==expected,'joint':bool(fmt and ex==expected)}

def main(a):
    from transformers import AutoTokenizer
    c=json.loads((ROOT/'config.json').read_text());lock=prepare(a.cache);tok=AutoTokenizer.from_pretrained(a.cache/'models'/lock['model']['revision'],local_files_only=True)
    tr,dev,res,_,nd,nres=select(c,a.cache,lock);split=json.loads((a.out/'split.json').read_text());pool={'sst2':{r['idx']:r for r in dev},'news':{r['idx']:r for r in nd}}
    expected={('sst2',i) for i in split['dev']}|{('news',i) for i in split['news_dev']};reserved={('sst2',r['idx']) for r in res}|{('news',r['idx']) for r in nres};assert not expected&reserved
    p=a.raw if a.raw else a.out/'predictions.jsonl';rr=[json.loads(x) for x in p.read_text().splitlines()];ix={(r['stage'],r['template'],r['idx']):r for r in rr};assert len(ix)==len(rr)==2*(4*len(split['dev'])+2*len(split['news_dev']))
    parsed=redacted=0
    for r in rr:
        assert (r['task'],r['idx']) in expected
        row=pool[r['task']][r['idx']];assert r['gold']==row['label'];s=row['text' if r['task']=='news' else 'sentence'];assert sha(s)==r['source_sha256']
        prompt=tok.apply_chat_template(messages(c,r['template'],s),tokenize=False,add_generation_prompt=True);ids=tok(prompt,add_special_tokens=False)['input_ids'];assert sha(prompt)==r['prompt_sha256'] and len(ids)==r['input_length']<=512
        if 'output' in r:
            assert tok.decode(r['generated_ids'],skip_special_tokens=True)==r['output'];assert sha(r['output'])==r['output_sha256'];assert len(r['generated_ids'])==r['generated_length']
            assert ('eos' if r['generated_ids'][-1]==tok.eos_token_id else 'length')==r['stop'];assert all(r[k]==v for k,v in parse(r['output'],r['template'],r['gold']).items());parsed+=1
        else:assert r.get('redacted') and len(r['output_sha256'])==64;redacted+=1
    su=json.loads((a.out/'summary.json').read_text())
    for stage,dd in su.items():
        for template,s in dd.items():
            rows=[r for r in rr if (r['stage'],r['template'])==(stage,template)];assert len(rows)==s['n']
            for k in ['format','content_proxy','joint']:assert sum(r[k] for r in rows)==s[k]
            assert sum(r['input_length'] for r in rows)==s['input_tokens'];assert sum(r['generated_length'] for r in rows)==s['generated_tokens']
    for name,sc in json.loads((a.out/'paired.json').read_text()).items():
        left,right=name.split(' -> ');sa,ta=left.split('/');sb,tb=right.split('/');rows=[r for r in rr if (r['stage'],r['template'])==(sa,ta)]
        for k,counts in sc.items():
            dd=[int(ix[sb,tb,r['idx']][k])-int(r[k]) for r in rows];assert counts=={'win':sum(x>0 for x in dd),'tie':sum(x==0 for x in dd),'loss':sum(x<0 for x in dd),'delta_count':sum(dd)}
    identity=json.loads((a.out/'identity.json').read_text());key=identity.pop('cache_key');assert sha(json.dumps(identity,sort_keys=True))==key
    for n,h in identity['code_hashes'].items():assert digest(ROOT/n)==h,n
    for text,t,g,want in [('negative','plain_original',0,(1,1,1)),('{"sentiment":"positive"}','json_original',1,(1,1,1)),('{"sentiment":"positive","sentiment":"positive"}','json_original',1,(0,1,0)),('positive or negative','plain_original',1,(0,0,0)),('Answer: B','news_original',1,(0,1,0)),('B','news_original',1,(1,1,1)),('B C','news_original',1,(0,0,0)),('[]','json_original',0,(0,0,0))]:assert tuple(parse(text,t,g).values())==tuple(map(bool,want))
    report={'status':'passed','rows':len(rr),'fully_reparsed':parsed,'redacted_not_reparsed':redacted,'prompt_and_gold_reconstruction':True,'independent_metrics_and_pairs':True,'source_identity':True,'reserved_inference':False,'parser_fixtures':8,'human_label_preservation':'pending;automated identity is not human review'};save(a.report or a.out/'audit.json',report);print(json.dumps(report))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,required=True);p.add_argument('--raw',type=pathlib.Path);p.add_argument('--report',type=pathlib.Path);main(p.parse_args())
