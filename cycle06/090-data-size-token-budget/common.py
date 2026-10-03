"""Shared audited template, mask and inference helpers from article 089."""
import argparse, gc, hashlib, json, pathlib, platform, resource, sys, time, traceback
from prepare import ROOT, digest, prepare
from scoring import score, aggregate

def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(x):return hashlib.sha256(x.encode()).hexdigest()
def ordered(rows,s):return sorted(rows,key=lambda r:sha(f'{s}:{r["idx"]}'))
def message(c,k,s):return [{'role':'system','content':c['system']},{'role':'user','content':c['task']+'\n'+c['contracts'][k]+'\nReview: '+s}]
def build(tok,c,row):
    msgs=message(c,'json',row['sentence']); prefix=tok.apply_chat_template(msgs,tokenize=True,add_generation_prompt=True)
    target=json.dumps({'sentiment':('negative','positive')[row['label']]})
    full=tok.apply_chat_template(msgs+[{'role':'assistant','content':target}],tokenize=True,add_generation_prompt=False)
    assert full[:len(prefix)]==prefix,'Boundary token merge: do not infer a mask from separate text lengths'
    eos=full.index(tok.eos_token_id,len(prefix)); assert tok.decode(full[len(prefix):eos])==target
    ids=full[:eos+1];assert len(ids)<=c['max_train_tokens'],'Truncated response prohibited'
    return {'idx':row['idx'],'gold':row['label'],'ids':ids,'prefix':len(prefix),'eos':eos,'template_tail_removed':full[eos+1:],'sentence_sha256':sha(row['sentence'])}
def collate(rows,arm,pad):
    import torch
    width=max(len(r['ids']) for r in rows);ids=torch.full((len(rows),width),pad,dtype=torch.long);att=torch.zeros_like(ids);lab=torch.full_like(ids,-100)
    for i,r in enumerate(rows):
        n=len(r['ids']);ids[i,:n]=torch.tensor(r['ids']);att[i,:n]=1;lab[i,:n]=ids[i,:n]
        if arm=='assistant':lab[i,:r['prefix']]=-100
        assert lab[i,r['eos']]==pad # real EOS remains despite EOS==PAD
        assert (lab[i,n:]==-100).all()
    return {'input_ids':ids,'attention_mask':att,'labels':lab}
def evaluate(model,tok,c,rows,stage):
    import torch
    from transformers import GenerationConfig
    gen=GenerationConfig(**c['decoding'],eos_token_id=tok.eos_token_id,pad_token_id=tok.pad_token_id,bos_token_id=tok.bos_token_id)
    model.eval();records=[]
    with torch.inference_mode():
        for r in rows:
            for k in ('plain','json'):
                text=tok.apply_chat_template(message(c,k,r['sentence']),tokenize=False,add_generation_prompt=True)
                x=tok(text,add_special_tokens=False,return_tensors='pt');assert x.input_ids.shape[1]<=c['max_input_tokens']
                start=time.perf_counter();y=model.generate(**x,generation_config=gen)[0,x.input_ids.shape[1]:].tolist();out=tok.decode(y,skip_special_tokens=True)
                records.append({'stage':stage,'idx':r['idx'],'gold':r['label'],'contract':k,'prompt_sha256':sha(text),'input_length':x.input_ids.shape[1],'generated_ids':y,'output':out,'stop':'eos' if y[-1]==tok.eos_token_id else 'length','seconds':time.perf_counter()-start,**score(out,k,r['label'])})
    return records
