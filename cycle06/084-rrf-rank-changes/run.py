"""Run pinned official Pyserini fusion over hash-locked real SciFact rankings."""
import argparse, csv, gzip, hashlib, importlib.util, importlib.metadata, json
import os, platform, random, resource, sys, time, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, obj): Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
def ranked(scores): return sorted(scores, key=lambda d:(scores[d],d), reverse=True)
def dump_gz(path, obj):
    with open(path,'wb') as raw:
        with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as f:
            f.write(json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode())

def official_modules():
    # Execute two complete, byte-identical upstream modules without Java/search imports.
    # The upstream fusion import points at this exact official trectools module.
    def load(name, filename):
        spec=importlib.util.spec_from_file_location(name,ROOT/'vendor'/filename)
        mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
    trec=load('pyserini.trectools','trectools_base.py')
    fusion=load('_pinned_pyserini_fusion','fusion_base.py')
    return trec, fusion

def alternate(a,b,depth):
    out=[];seen=set()
    for i in range(depth):
        for source in [a,b]:
            if i<len(source) and source[i] not in seen:
                out.append(source[i]);seen.add(source[i])
    return {d:float(len(out)-i) for i,d in enumerate(out[:100])}

def evaluate(qrels,run):
    import pytrec_eval
    ev=pytrec_eval.RelevanceEvaluator(qrels,{'ndcg_cut.10','recall.100'})
    metrics=ev.evaluate(run)
    for q,score in run.items():
        metrics[q]['mrr_10']=next((1/i for i,d in enumerate(ranked(score)[:10],1) if qrels[q].get(d,0)>0),0.)
    return metrics

def package_version(name):
    try: return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError: return "not installed"

def main(args):
    start=time.perf_counter();out=args.out;out.mkdir(parents=True,exist_ok=False)
    state={'command':['python','run.py',*sys.argv[1:]],'exit_code':None,'stages':{}}
    try:
        lock=read(ROOT/'input.lock.json');config=read(ROOT/'config.json');random.seed(config['seed'])
        for f,h in lock['files'].items():assert sha(ROOT/f)==h,f
        state['stages']['download']={'status':'passed','mode':'included hash-verified frozen rankings and two official modules; no download needed'}
        bm=read(ROOT/'inputs/bm25.json');dense=read(ROOT/'inputs/dense.json');gold=read(ROOT/'inputs/qrels.json')
        assert set(bm)==set(dense)==set(gold) and len(gold)==809
        ids=sorted(gold,key=int)
        if args.smoke:ids=ids[:5]
        bm={q:bm[q] for q in ids};dense={q:dense[q] for q in ids};gold={q:gold[q] for q in ids}
        orders={name:{q:ranked(run[q]) for q in ids} for name,run in [('bm25',bm),('dense',dense)]}
        assert all(0<len(v)<=100 for r in orders.values() for v in r.values())
        state['stages']['preprocessing']={'status':'passed','queries':len(ids),'input_rows':[sum(map(len,bm.values())),sum(map(len,dense.values()))]}
        trec,fusion=official_modules()
        def make(run):return trec.TrecRun.from_list([(q,'Q0',d,i,run[q][d],'frozen') for q in ids for i,d in enumerate(ranked(run[q]),1)])
        br,dr=make(bm),make(dense); originals=[br.run_data.copy(deep=True),dr.run_data.copy(deep=True)]
        runs={'bm25':bm,'dense':dense}; full={}; timing={};official_tie_differences={}
        settings=[('rrf10',10,100),('rrf60',60,100),('rrf100',100,100),('rrf60_depth50',60,50)]
        for name,k,depth in settings:
            t=time.perf_counter();rr=fusion.reciprocal_rank_fusion([br,dr],rrf_k=k,depth=depth,k=None)
            scores={q:{} for q in ids};asc={q:[] for q in ids}
            for row in rr.run_data.itertuples(index=False):scores[row.topic][row.docid]=float(row.score);asc[row.topic].append(row.docid)
            full[name]=scores;runs[name]={q:{d:scores[q][d] for d in ranked(scores[q])[:100]} for q in ids}
            official_tie_differences[name]=sum(asc[q][:100]!=ranked(scores[q])[:100] for q in ids)
            timing[name]=time.perf_counter()-t
        assert br.run_data.equals(originals[0]) and dr.run_data.equals(originals[1])
        duplicate=fusion.reciprocal_rank_fusion([br,br],rrf_k=60,depth=100,k=None)
        dupe={q:{} for q in ids}
        for row in duplicate.run_data.itertuples(index=False):dupe[row.topic][row.docid]=float(row.score)
        assert all(ranked(dupe[q])==orders['bm25'][q] for q in ids)
        for depth in [50,100]:runs['alternate'+str(depth)]={q:alternate(orders['bm25'][q],orders['dense'][q],depth) for q in ids}
        state['stages']['fusion']={'status':'passed','official_calls':5,'seconds_by_setting':timing,'input_unchanged':True,'duplicate_baseline_identical':True}
        per={name:evaluate(gold,r) for name,r in runs.items()};summary={};n=len(ids)
        for name,rows in per.items():
            deltas=[rows[q]['ndcg_cut_10']-per['bm25'][q]['ndcg_cut_10'] for q in ids]
            summary[name]={k:sum(rows[q][k] for q in ids)/n for k in ['ndcg_cut_10','recall_100','mrr_10']}
            summary[name].update(delta_vs_bm25=sum(deltas)/n,wins=sum(x>1e-12 for x in deltas),ties=sum(abs(x)<=1e-12 for x in deltas),losses=sum(x< -1e-12 for x in deltas))
        budget={};trace=[];cases=[]
        for depth in [50,100]:
            sizes=[];recall=[];lost=0;hit0=0
            method='rrf60' if depth==100 else 'rrf60_depth50'
            for q in ids:
                union=set(orders['bm25'][q][:depth])|set(orders['dense'][q][:depth]);rel={d for d,r in gold[q].items() if r>0}
                sizes.append(len(union));recall.append(len(union&rel)/len(rel));lost+=len((union&rel)-set(runs[method][q]));hit0+=not bool(union&rel)
            budget[str(depth)]={'union_min':min(sizes),'union_max':max(sizes),'union_mean':sum(sizes)/n,'union_total':sum(sizes),'union_recall':sum(recall)/n,'relevant_pairs_dropped_by_output100':lost,'no_relevant_candidate_queries':hit0}
        for q in ids:
            rb={d:i for i,d in enumerate(orders['bm25'][q],1)};rd={d:i for i,d in enumerate(orders['dense'][q],1)};rf={d:i for i,d in enumerate(ranked(full['rrf60'][q]),1)}
            for d in ranked(full['rrf60'][q]):
                trace.append([q,d,rb.get(d),rd.get(d),1/(60+rb[d]) if d in rb else 0.,1/(60+rd[d]) if d in rd else 0.,full['rrf60'][q][d],rf[d],int(rf[d]<=100),int(gold[q].get(d,0)>0)])
            cases.append({'query_id':q,'delta':per['rrf60'][q]['ndcg_cut_10']-per['bm25'][q]['ndcg_cut_10'],'relevant_ranks':[{ 'doc_id':d,'bm25':rb.get(d),'dense':rd.get(d),'rrf60_union':rf.get(d)} for d in sorted(gold[q]) if gold[q][d]>0]})
        selected={'largest_gains':sorted(cases,key=lambda x:(-x['delta'],int(x['query_id'])))[:3],'largest_losses':sorted(cases,key=lambda x:(x['delta'],int(x['query_id'])))[:3]}
        write(out/'summary.json',{'scope':'smoke5' if args.smoke else 'all809 development queries','methods':summary,'budgets':budget,'official_ascending_tie_order_differs':official_tie_differences,'settings':settings})
        write(out/'per_query.json',per);write(out/'cases.json',selected);dump_gz(out/'runs.json.gz',runs)
        with open(out/'trace.csv.gz','wb') as raw:
            import io
            with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as f:
                with io.TextIOWrapper(f,encoding='utf-8',newline='') as text:
                    w=csv.writer(text);w.writerow(['query_id','doc_id','bm25_rank','dense_rank','bm25_contribution','dense_contribution','rrf_score','union_rank','output100','relevant']);w.writerows(trace)
        identity={'config':config,'inputs':lock,'model_revision':'1110a243fdf4706b3f48f1d95db1a4f5529b4d41','tokenizer_revision':'1110a243fdf4706b3f48f1d95db1a4f5529b4d41','prompt':None,'decoding':None,'code':{f.name:sha(f) for f in sorted(ROOT.glob('*.py'))},'dependencies':{k:package_version(k) for k in ['numpy','pandas','pytrec_eval-terrier']},'requirements_sha256':sha(ROOT/'requirements.txt'),'scope':'smoke' if args.smoke else 'development'}
        identity['cache_key']=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest();write(out/'identity.json',identity)
        state['stages']['evaluation']={'status':'passed','queries':n,'methods':len(runs),'candidate_trace_rows':len(trace),'primary_evaluator':'pytrec_eval-terrier'};state['exit_code']=0
        print(json.dumps({'methods':summary,'budgets':budget,'tie_changes':official_tie_differences},indent=2))
    except Exception:
        state['exit_code']=1;state['error']=traceback.format_exc();raise
    finally:
        state['elapsed_seconds']=time.perf_counter()-start;write(out/'status.json',state)
        write(out/'environment.json',{'python':sys.version,'platform':platform.platform(),'device':'CPU','peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),'packages':{k:package_version(k) for k in ['numpy','pandas','pytrec_eval-terrier']},'retrieval_time':'not remeasured; frozen real input rankings reused','gpu_memory':'not applicable'})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke',action='store_true');main(p.parse_args())
