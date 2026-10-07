"""Recheck saved public outcomes, identities and arithmetic without model inference."""
import argparse, json, pathlib
from prepare import ROOT,prepare,digest
from run import load_data,split,context,sha,save
def main(a):
    from transformers import AutoTokenizer
    lock=prepare(a.cache);data=load_data(a.cache,lock);dev,reserved=split(data['validation']);sp=json.loads((a.out/'split.json').read_text());assert set(sp['selected'])<=set(r['idx'] for r in dev);assert not set(sp['selected'])&set(r['idx'] for r in reserved)
    tok=AutoTokenizer.from_pretrained(a.cache/'models'/lock['model']['revision'],local_files_only=True)
    sources={r['idx']:r for r in dev};ind=json.loads((a.out/'independent.json').read_text());official=json.loads((a.out/'official_samples.json').read_text());byid={r['idx']:r for r in official};correct=0;errors=[]
    assert len(byid)==len(ind)==len(sp['selected']) and set(byid)==set(sp['selected'])
    for r in ind:
        src=sources[r['idx']];ctx=context(src);assert r['gold']==src['label'];assert r['context_sha256']==sha(ctx)==byid[r['idx']]['context_sha256']
        ci=tok.encode(ctx,add_special_tokens=False);assert len(ci)==r['context_tokens']
        values=[]
        for j,choice in enumerate(['negative','positive']):
            v=r['candidates'][j];ids=tok.encode(ctx+' '+choice,add_special_tokens=False)
            assert ids[:len(ci)]==ci and v['continuation_ids']==ids[len(ci):]
            assert v['input_shape']==[1,len(ids)-1] and v['logits_shape']==[1,len(ids)-1,49152]
            assert sum(v['token_logprobs'])==v['sum_logprob'];values.append(v['sum_logprob']);errors.append(abs(values[-1]-byid[r['idx']]['scores'][j]))
        pred=0 if values[0]>=values[1] else 1;acc=int(pred==src['label']);assert pred==r['pred'] and acc==r['acc']==byid[r['idx']]['acc'];correct+=acc
    score=correct/len(ind);assert score==json.loads((a.out/'official_metrics.json').read_text())['sst2']['acc,none'];assert max(errors)<=2e-5
    result={'status':'passed','rows':len(ind),'correct':correct,'accuracy':score,'max_likelihood_error':max(errors),'token_boundary_shape_and_arithmetic':'passed','gold_ids_prompts':'passed','confirmation_inference':False,'human_review':'separate; this is machine audit'}
    save(a.out/'offline_audit.json',result);print(json.dumps(result))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,required=True);main(p.parse_args())
