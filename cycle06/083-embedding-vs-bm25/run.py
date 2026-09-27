"""Official SentenceTransformer + pinned BEIR exact search; local audit adapter."""
import argparse, hashlib, importlib.metadata, json, os, platform, random, resource, sys, time, traceback
from pathlib import Path
from common import ROOT, read_json, write_json, sha, ranked, metrics, verify_resources

class Encoder:
    """Independent single-process BEIR adapter; official encode executes the model."""
    def __init__(self, model, out, ids):
        self.model, self.out, self.ids = model, out, ids
        self.trace = {}
    def encode(self, texts, role, **kwargs):
        import numpy as np
        t=time.perf_counter()
        lengths=[len(x) for x in self.model.tokenizer(texts, truncation=False, verbose=False)['input_ids']]
        self.trace[role]={'count':len(texts),'tokens_untruncated':sum(lengths),'tokens_after_truncation':sum(min(x,256) for x in lengths),'truncated_count':sum(x>256 for x in lengths),'max_tokens':max(lengths)}
        arr=self.model.encode(texts, **kwargs)
        assert arr.shape==(len(texts),384) and arr.isfinite().all()
        assert (arr.norm(dim=1)-1).abs().max()<1e-5
        self.trace[role].update(seconds=time.perf_counter()-t,shape=list(arr.shape),max_norm_error=float((arr.norm(dim=1)-1).abs().max()))
        np.save(self.out/(role+'.npy'),arr.cpu().numpy())
        write_json(self.out/(role+'_lengths.json'),dict(zip(self.ids[role],lengths)))
        write_json(self.out/'encoding.json', self.trace)
        return arr
    def encode_queries(self, queries, **kwargs):return self.encode(queries,'queries',**kwargs)
    def encode_corpus(self, corpus, **kwargs):return self.encode([(x.get('title','')+' '+x['text']).strip() for x in corpus],'corpus',**kwargs)

def main(a):
    start=time.perf_counter(); a.out.mkdir(parents=True,exist_ok=False)
    state={'command':[sys.executable,*sys.argv],'exit_code':None,'stages':{}}
    try:
        c=read_json(ROOT/'config.json'); lock,upstream=verify_resources(a.cache)
        ml=read_json(ROOT/'model.lock.json'); modeldir=a.cache/'models'/c['revision']
        for f,x in ml['files'].items():assert sha(modeldir/f)==x['sha256'],f
        state['stages']['download']={'status':'passed','mode':'verified cache','model_revision':c['revision']}
        import numpy as np, torch
        from sentence_transformers import SentenceTransformer
        import pytrec_eval
        sys.path.insert(0,str(upstream.resolve()))
        from beir.datasets.data_loader import GenericDataLoader
        from beir.retrieval.search.dense import DenseRetrievalExactSearch
        from beir.retrieval.evaluation import EvaluateRetrieval
        torch.set_num_threads(c['threads']);torch.manual_seed(c['seed']);np.random.seed(c['seed']);random.seed(c['seed']);torch.use_deterministic_algorithms(True)
        t=time.perf_counter()
        corpus,queries,gold=GenericDataLoader(str(a.cache/'scifact')).load(split='train')
        queries=dict(sorted(queries.items(),key=lambda x:int(x[0])))
        assert len(corpus)==5183 and len(queries)==809
        if a.smoke:
            queries=dict(list(queries.items())[:5]);corpus=dict(list(corpus.items())[:16])
        gold={q:gold[q] for q in queries}
        dids=sorted(corpus,key=lambda k:len(corpus[k].get('title','')+corpus[k]['text']),reverse=True)
        ids={'queries':list(queries),'corpus':dids};write_json(a.out/'ids.json',ids)
        assert not set(queries)&set(corpus)
        identity={'config':c,'model':ml,'data':lock['data_files'],'beir':lock['beir_commit'],'code':{p.name:sha(p) for p in sorted(ROOT.glob('*.py'))},'requirements':sha(ROOT/'requirements.txt'),'scope':'smoke' if a.smoke else 'development','ids':ids}
        key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest();write_json(a.out/'identity.json',{'cache_key':key,**identity})
        state['stages']['preprocessing']={'status':'passed','seconds':time.perf_counter()-t}
        t=time.perf_counter();model=SentenceTransformer(str(modeldir),device='cpu',local_files_only=True,trust_remote_code=False)
        model.max_seq_length=256;model.eval();model.requires_grad_(False)
        assert model.get_sentence_embedding_dimension()==384
        state['stages']['model_load']={'status':'passed','seconds':time.perf_counter()-t,'parameters':sum(p.numel() for p in model.parameters())}
        # Verify the installed official pipeline against model-card masked-mean recipe.
        toks=model.tokenizer(['A scientific claim.','Protein expression changes under treatment.'],padding=True,truncation=True,max_length=256,return_tensors='pt')
        with torch.inference_mode():
            h=model[0].auto_model(**toks).last_hidden_state
            pooled=(h*toks['attention_mask'][...,None]).sum(1)/toks['attention_mask'].sum(1)[:,None]
            direct=torch.nn.functional.normalize(pooled,dim=1)
            official=model.encode(['A scientific claim.','Protein expression changes under treatment.'],convert_to_tensor=True)
        error=float((direct-official).abs().max());assert error<1e-5
        write_json(a.out/'shape_check.json',{'input_ids':list(toks['input_ids'].shape),'hidden':list(h.shape),'pooled':list(pooled.shape),'model_card_max_error':error,'model_training':model.training,'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad)})
        encoder=Encoder(model,a.out,ids)
        search=DenseRetrievalExactSearch(encoder,batch_size=16,corpus_chunk_size=50000,show_progress_bar=True)
        t=time.perf_counter();results=search.search(corpus,queries,top_k=100,score_function='cos_sim')
        state['stages']['inference_retrieval']={'status':'passed','seconds':time.perf_counter()-t}
        write_json(a.out/'results.json',results);write_json(a.out/'qrels_used.json',gold)
        with (a.out/'run.trec').open('w') as f:
            for q,r in results.items():
                for i,d in enumerate(ranked(r),1):f.write(f'{q} Q0 {d} {i} {r[d]:.10g} cycle6-083\n')
        t=time.perf_counter();official=EvaluateRetrieval.evaluate(gold,results,[1,10,100])
        per=pytrec_eval.RelevanceEvaluator(gold,{'ndcg_cut.1,10,100','recall.1,10,100'}).evaluate(results)
        worst=0.
        for q in queries:
            ind=metrics(gold[q],ranked(results[q]));worst=max(worst,max(abs(per[q][k]-ind[k]) for k in per[q]));per[q]['mrr_10']=ind['mrr_10']
        assert worst<1e-10 and set(per)==set(queries)
        write_json(a.out/'per_query.json',per)
        summary={'scope':'SMOKE: reduced corpus, not a benchmark' if a.smoke else 'development full corpus','n_documents':len(corpus),'n_queries':len(queries),'metrics':{k:sum(v[k] for v in per.values())/len(per) for k in next(iter(per.values()))},'official_beir_rounded':official,'metric_max_error':worst}
        write_json(a.out/'summary.json',summary);state['stages']['evaluation']={'status':'passed','seconds':time.perf_counter()-t}
        state['exit_code']=0;print(json.dumps(summary,indent=2),flush=True)
    except Exception:
        state['exit_code']=1;state['error']=traceback.format_exc();raise
    finally:
        state['elapsed_seconds']=time.perf_counter()-start
        write_json(a.out/'status.json',state)
        write_json(a.out/'environment.json',{'python':sys.version,'platform':platform.platform(),'device':'CPU','dtype':'float32','peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),'packages':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()},'gpu_memory':'not applicable','timing':'includes imports/model loading; download excluded'})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke',action='store_true');main(p.parse_args())
