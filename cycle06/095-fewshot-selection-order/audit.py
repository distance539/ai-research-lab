"""Independent replay of saved likelihoods, paired statistics and prompt identity."""
import argparse
import hashlib
import json
from pathlib import Path

from prepare import ROOT, prepare, digest

def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()

def audit_records(records, summary, plan, selected):
    names=list(plan['conditions'])
    assert len(records)==len(names)*len(selected)
    grouped={name:{} for name in names}
    for row in records:
        name=row['condition']
        assert name in grouped and row['idx'] not in grouped[name]
        assert row['idx'] in selected and row['idx'] not in plan['reserved']
        assert row['demo_ids']==plan['conditions'][name]
        candidates=row['candidates']
        assert len(candidates)==len(row['official_scores'])==2
        values=[]
        for j,c in enumerate(candidates):
            assert len(c['continuation_ids'])==len(c['token_logprobs'])
            assert abs(sum(c['token_logprobs'])-c['score'])<1e-10
            assert abs(c['score']-row['official_scores'][j])<=2e-5
            assert c['input_tokens']==row['context_tokens']+len(c['continuation_ids'])-1
            assert c['input_shape']==[1,c['input_tokens']]
            assert c['logits_shape']==[1,c['input_tokens'],49152]
            values.append(c['score'])
        winner=values.index(max(values))
        assert winner==row['prediction']
        assert int(winner==row['gold'])==row['correct']
        assert abs(row['max_score_error']-max(abs(values[j]-row['official_scores'][j]) for j in range(2)))<1e-10
        grouped[name][row['idx']]=row
    assert all(set(v)==set(selected) for v in grouped.values())
    assert summary['independent_units']==len(selected) and summary['outputs']==len(records)
    for name,rows in grouped.items():
        target=summary['conditions'][name]
        assert target['n']==len(selected)
        assert target['correct']==sum(r['correct'] for r in rows.values())
        assert target['accuracy']==target['correct']/len(selected)
        assert target['positive_predictions']==sum(r['prediction'] for r in rows.values())
        assert target['context_tokens_min']==min(r['context_tokens'] for r in rows.values())
        assert target['context_tokens_max']==max(r['context_tokens'] for r in rows.values())
        assert target['candidate_input_tokens']==sum(c['input_tokens'] for r in rows.values() for c in r['candidates'])
    contrasts=[('zero',n) for n in names if n!='zero']+[('A_forward','A_reverse'),('B_forward','B_reverse'),('A_forward','B_forward'),('A_reverse','B_reverse')]
    assert set(summary['paired_contrasts'])=={b+' minus '+a for a,b in contrasts}
    for a,b in contrasts:
        x,y=grouped[a],grouped[b]
        assert all(x[i]['gold']==y[i]['gold'] for i in selected)
        gains=[i for i in selected if y[i]['correct']>x[i]['correct']]
        losses=[i for i in selected if y[i]['correct']<x[i]['correct']]
        changed=sorted(i for i in selected if x[i]['prediction']!=y[i]['prediction'])
        target=summary['paired_contrasts'][b+' minus '+a]
        assert target=={'n':len(selected),'gains':len(gains),'losses':len(losses),'prediction_flips':len(changed),'changed_ids':changed,'accuracy_delta':(len(gains)-len(losses))/len(selected)}
    for i in selected:
        assert len({grouped[n][i]['context_tokens'] for n in names if n!='zero'})==1
    return grouped

def main(args):
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    lock=prepare(args.cache)
    plan=json.loads((ROOT/'plan.json').read_text())
    identity=json.loads((args.out/'identity.json').read_text())
    assert identity['plan_sha256']==digest(ROOT/'plan.json')
    assert json.loads((args.out/'plan.json').read_text())==plan
    selected=identity['selected']
    assert selected in [plan['development'],plan['development'][:2]]
    records=json.loads((args.out/'records.json').read_text())
    summary=json.loads((args.out/'summary.json').read_text())
    grouped=audit_records(records,summary,plan,selected)
    base=args.cache/'glue'/lock['glue']['revision']/'sst2'
    rows={s:pq.read_table(base/f'{s}-00000-of-00001.parquet').to_pylist() for s in ['train','validation']}
    train={r['idx']:r for r in rows['train']}
    validation={r['idx']:r for r in rows['validation']}
    val_sorted=sorted(validation,key=lambda i:sha('88:'+str(i)))
    assert val_sorted[:64]==plan['development'] and val_sorted[64:192]==plan['reserved']
    tok=AutoTokenizer.from_pretrained(args.cache/'models'/lock['model']['revision'],local_files_only=True)
    choices=['negative','positive']
    suffix='\nQuestion: Is this sentence positive or negative?\nAnswer:'
    normalized=lambda t:' '.join(t.lower().split())
    demo_ids=plan['conditions']['A_forward']+plan['conditions']['B_forward']
    assert len(set(demo_ids))==8
    assert plan['conditions']['A_reverse']==plan['conditions']['A_forward'][::-1]
    assert plan['conditions']['B_reverse']==plan['conditions']['B_forward'][::-1]
    demo_texts={normalized(train[i]['sentence']) for i in demo_ids}
    assert len(demo_texts)==8 and not demo_texts&{normalized(r['sentence']) for r in rows['validation']}
    for i in demo_ids:
        src=train[i]
        text=src['sentence']+suffix+' '+choices[src['label']]+'\n\n'
        recorded=plan['demos'][str(i)]
        assert recorded['label']==src['label'] and recorded['block_tokens']==42
        assert recorded['sentence_sha256']==sha(src['sentence']) and recorded['block_sha256']==sha(text)
        assert len(tok.encode(text,add_special_tokens=False))==42
    for name,mapping in grouped.items():
        demos=[train[i] for i in plan['conditions'][name]]
        assert not demos or [r['label'] for r in demos].count(1)==2
        prefix=''.join(r['sentence']+suffix+' '+choices[r['label']]+'\n\n' for r in demos)
        assert len(tok.encode(prefix,add_special_tokens=False))==plan['prefix_tokens'][name]
        for i,row in mapping.items():
            context=prefix+validation[i]['sentence']+suffix
            ids=tok.encode(context,add_special_tokens=False)
            assert row['gold']==validation[i]['label']
            assert row['context_sha256']==sha(context) and row['context_tokens']==len(ids)
            for j,choice in enumerate(choices):
                full=tok.encode(context+' '+choice,add_special_tokens=False)
                assert len(full)<=512 and full[:len(ids)]==ids
                assert row['candidates'][j]['continuation_ids']==full[len(ids):]
    result={'status':'passed','conditions':len(grouped),'paired_questions':len(selected),'candidate_scores':2*len(records),'max_score_error':max(r['max_score_error'] for r in records),'prompt_identity_token_boundaries_gold_and_arithmetic':'passed','demonstrations_train_only_no_exact_overlap':'passed','paired_statistics':'passed','equal_fewshot_context_length':'passed','confirmation_inference':False,'human_review':'not performed or claimed','records_sha256':digest(args.out/'records.json'),'audit_source_sha256':digest(Path(__file__))}
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
