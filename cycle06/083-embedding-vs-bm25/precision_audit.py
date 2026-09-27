"""Diagnose float64 replay order changes without replacing official float32 results."""
import argparse
from pathlib import Path
import numpy as np
from common import read_json,write_json,ranked,metrics
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();r=a.run
q=np.load(r/'queries.npy').astype('float64');d=np.load(r/'corpus.npy').astype('float64');q/=np.linalg.norm(q,axis=1,keepdims=True);d/=np.linalg.norm(d,axis=1,keepdims=True)
ids=read_json(r/'ids.json');saved=read_json(r/'results.json');gold=read_json(r/'qrels_used.json');changes=[]
for i,qid in enumerate(ids['queries']):
 s=q[i]@d.T;ix=sorted(range(len(d)),key=lambda j:(float(s[j]),ids['corpus'][j]),reverse=True);order=[ids['corpus'][j] for j in ix[:100]];orig=ranked(saved[qid])
 if order!=orig:
  m1,m2=metrics(gold[qid],orig),metrics(gold[qid],order)
  changes.append({'query_id':qid,'same_top100_set':set(order)==set(orig),'metric_deltas':{k:m2[k]-m1[k] for k in m1}})
write_json(r/'precision_audit.json',{'changed_queries':changes,'all_metrics_unchanged':all(all(v==0 for v in x['metric_deltas'].values()) for x in changes),'scope':'float64 re-score of saved float32 embeddings, not model rerun'})
print(len(changes),'changed orders; metrics unchanged:',all(all(v==0 for v in x['metric_deltas'].values()) for x in changes))
