"""Independent stdlib reconstruction, no imports from production metric/taxonomy code."""
import argparse,csv,hashlib,json,math
from pathlib import Path

def check(root,out):
    load=lambda p:json.loads(p.read_text())
    rank=lambda x:sorted(x,key=lambda d:(-x[d],tuple(-ord(c) for c in d)))
    # Explicit reverse string sort agrees with trec_eval tie rule, including prefix IDs.
    def rank(x): return sorted(sorted(x,reverse=True),key=x.get,reverse=True)
    gold=load(root/'inputs/qrels.json');runs=load(out/'runs.json');rows=load(out/'taxonomy.json');s=load(out/'discovery.json')
    assert len(rows)==len({r['query_id'] for r in rows})==len(runs['original'])
    assert {r['query_id'] for r in rows}==set(runs['original'])
    pairs=list(csv.DictReader((out/'pair_scores.csv').open()))
    assert len({(p['query_id'],p['doc_id']) for p in pairs})==len(pairs)
    counts={k:0 for k in s['categories']};cross={False:[0,0,0.],True:[0,0,0.]}
    def ndcg(q,ranked):
        rel={d for d,v in gold[q].items() if v>0}
        return sum(1/math.log2(i+2) for i,d in enumerate(ranked[:10]) if d in rel)/sum(1/math.log2(i+2) for i in range(min(10,len(rel))))
    for r in rows:
        q=r['query_id'];g={d for d,v in gold[q].items() if v>0};found=g.intersection(runs['original'][q]);top=g.intersection(rank(runs['reranked'][q])[:10])
        delta=ndcg(q,rank(runs['reranked'][q]))-ndcg(q,rank(runs['original'][q]))
        assert abs(delta-r['delta'])<1e-12
        assert abs(ndcg(q,rank(runs['original'][q]))-r['original_ndcg'])<1e-12
        assert abs(ndcg(q,rank(runs['reranked'][q]))-r['reranked_ndcg'])<1e-12
        cat='candidate_absent' if len(found)==0 else 'top10_miss' if len(top)==0 else 'hit_regression' if delta < -1e-12 else 'hit_gain' if delta>1e-12 else 'hit_tie'
        assert cat==r['category'];counts[cat]+=1
        tr={p['doc_id'] for p in pairs if p['query_id']==q and p['doc_id'] in found and int(p['raw_tokens'])>512}
        assert tr==set(r['truncated_relevant_ids']) and len(tr)==r['relevant_truncated']
        assert (len(found),len(g),len(top))==(r['candidate_relevant'],r['all_relevant'],r['top10_relevant'])
        if found:
            z=cross[bool(tr)];z[0]+=1;z[1]+=delta< -1e-12;z[2]+=delta
    assert counts==s['categories'];assert sum(counts.values())==s['queries']
    for t in s['truncation_association']:
        n,k,d=cross[t['any_relevant_truncated']];assert (n,k)==(t['queries'],t['regressions'])
        if n: assert abs(k/n-t['regression_fraction'])<1e-12 and abs(d/n-t['mean_delta'])<1e-12
    for f,h in s['inputs'].items():assert hashlib.sha256((out/f).read_bytes()).hexdigest()==h
    for key,f in [('protocol_sha256','PROTOCOL.md'),('registry_sha256','research_registry.json')]:assert hashlib.sha256((root/f).read_bytes()).hexdigest()==s[key]
    assert s['confirmation_queries_evaluated']==0 and s['future_tests_executed'] is False
    report={'passed':True,'queries':len(rows),'score_rows':len(pairs),'independent_categories':counts,'input_sha256':s['inputs'],'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();r=check(Path(__file__).resolve().parent,a.out)
    (a.out/'independent_check.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
