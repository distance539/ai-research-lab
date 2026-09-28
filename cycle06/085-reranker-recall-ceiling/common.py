"""Independent audit and file utilities; no copied third-party implementation."""
import hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def order(r):return sorted(r,key=lambda d:(r[d],d),reverse=True)
def metric(gold,rank):
 rel={d for d,s in gold.items() if s>0};assert rel
 dcg=sum(1/math.log2(i+1) for i,d in enumerate(rank[:10],1) if d in rel)
 ideal=sum(1/math.log2(i+1) for i in range(1,min(10,len(rel))+1))
 return {'ndcg_cut_10':dcg/ideal,'recall_10':len(set(rank[:10])&rel)/len(rel),'recall_50':len(set(rank[:50])&rel)/len(rel),'mrr_10':next((1/i for i,d in enumerate(rank[:10],1) if d in rel),0.)}
def verify(cache):
 for f,h in read(ROOT/'input.lock.json')['files'].items():assert sha(ROOT/f)==h,f
 data=read(ROOT/'resources.lock.json')
 # Verify only resources consumed here. Confirmation qrels are deliberately not loaded.
 for f in ['scifact/corpus.jsonl','scifact/queries.jsonl','scifact/qrels/train.tsv']:
  assert sha(cache/f)==data['data_files'][f],f
 model=read(ROOT/'model.lock.json');mp=cache/'models'/model['revision']
 for f,r in model['files'].items():assert sha(mp/f)==r['sha256'],f
 return mp
