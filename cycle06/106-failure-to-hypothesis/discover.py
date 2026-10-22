"""Deterministic failure taxonomy; descriptive associations, never semantic gold."""
import argparse, csv, hashlib
from pathlib import Path
from common import ROOT, read, write, sha, order

def discover(out):
    runs=read(out/'runs.json'); per=read(out/'per_query.json')
    gold=read(ROOT/'inputs/qrels.json'); prior=read(ROOT/'inputs/prior_runs.json')
    pairs=list(csv.DictReader((out/'pair_scores.csv').open()))
    rows=[]; categories={k:0 for k in ['candidate_absent','top10_miss','hit_regression','hit_gain','hit_tie']}
    for q in runs['original']:
        rel={d for d,v in gold[q].items() if v>0}
        retained=rel & runs['original'][q].keys()
        top=order(runs['reranked'][q])[:10]
        delta=per['reranked'][q]['ndcg_cut_10']-per['original'][q]['ndcg_cut_10']
        category=('candidate_absent' if not retained else 'top10_miss' if not rel.intersection(top)
                  else 'hit_regression' if delta < -1e-12 else 'hit_gain' if delta>1e-12 else 'hit_tie')
        categories[category]+=1
        rp=[x for x in pairs if x['query_id']==q and x['doc_id'] in retained]
        trunc=[x['doc_id'] for x in rp if int(x['raw_tokens'])>512]
        rows.append({'query_id':q,'category':category,'delta':delta,'candidate_relevant':len(retained),
                     'all_relevant':len(rel),'relevant_truncated':len(trunc),'truncated_relevant_ids':trunc,
                     'top10_relevant':len(rel.intersection(top)),
                     'original_first':next((i for i,d in enumerate(order(runs['original'][q]),1) if d in rel),None),
                     'reranked_first':next((i for i,d in enumerate(order(runs['reranked'][q]),1) if d in rel),None),
                     'original_ndcg':per['original'][q]['ndcg_cut_10'],
                     'reranked_ndcg':per['reranked'][q]['ndcg_cut_10']})
    eligible=[r for r in rows if r['candidate_relevant']]
    table=[]
    for value in [False,True]:
        group=[r for r in eligible if bool(r['relevant_truncated'])==value]
        reg=[r for r in group if r['delta'] < -1e-12]
        table.append({'any_relevant_truncated':value,'queries':len(group),'regressions':len(reg),
                      'regression_fraction':len(reg)/len(group) if group else None,
                      'mean_delta':sum(r['delta'] for r in group)/len(group) if group else None})
    examples=[]
    for cat in categories:
        group=[r for r in rows if r['category']==cat]
        if group: examples.append(min(group,key=lambda r:hashlib.sha256(('106:'+r['query_id']).encode()).hexdigest()))
    replay_errors=[abs(v-prior['reranked'][q][d]) for q,s in runs['reranked'].items() for d,v in s.items()]
    replay_order=all(order(s)==[d for d in order(prior['reranked'][q]) if d in s] for q,s in runs['reranked'].items())
    summary={'categories':categories,'queries':len(rows),'eligible_queries':len(eligible),
             'truncation_association':table,'not_causal':True,'no_human_semantic_labels':True,
             'regressions':sum(r['delta'] < -1e-12 for r in rows),
             'prior_logit_max_error':max(replay_errors),'prior_rank_equal':replay_order,
             'examples_rule':'one per nonempty class: smallest SHA256(106:query_id); illustrative, not population sample',
             'inputs':{f:sha(out/f) for f in ['runs.json','per_query.json','pair_scores.csv']},
             'protocol_sha256':sha(ROOT/'PROTOCOL.md'),'registry_sha256':sha(ROOT/'research_registry.json'),
             'future_tests_executed':False,'confirmation_queries_evaluated':0}
    write(out/'taxonomy.json',rows);write(out/'discovery.json',summary);write(out/'examples.json',examples)
    print(summary)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);discover(p.parse_args().out)
