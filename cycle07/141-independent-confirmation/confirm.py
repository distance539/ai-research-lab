"""Offline confirmation evaluation, never fits or runs model inference."""
import json,hashlib
from pathlib import Path
import numpy as np
from analyze import evaluate
R=Path(__file__).resolve().parent
def read(n):return json.loads((R/n).read_text())
def put(n,x):(R/n).write_text(json.dumps(x,indent=2)+'\n')
def main():
 for n,v in read('release141.json')['sha256'].items():assert hashlib.sha256((R/n).read_bytes()).hexdigest()==v
 rows=read('formal.json'); pools=read('pools.json');cfg=read('final_config.json')
 assert len(rows)==1000 and [r['id'] for r in rows]==[r['id'] for r in pools['confirmation']]
 for f in ['id','movie','text']:assert len({r[f] for p in pools.values() for r in p})==3000
 ts=cfg['temperatures'];pred={k:evaluate(rows,t) for k,t in [('global',[ts['global']]*2),('group',ts['length256'])]}
 v=np.array([b['loss']-a['loss'] for a,b in zip(pred['global']['rows'],pred['group']['rows'])]);rng=np.random.default_rng(137);boot=[float(v[rng.integers(0,1000,1000)].mean()) for _ in range(10000)];ci=np.quantile(boot,[.025,.975]).tolist();risk=pred['group']['risk80']-pred['global']['risk80']
 results=dict(cells={k:{f:x for f,x in p.items() if f not in ['rows','order','prefix']} for k,p in pred.items()},delta_nll=float(v.mean()),ci95=ci,delta_risk80=risk,registered_success=bool(ci[1]<0 and risk<=.01),group_sizes=[sum((r['retained_tokens']>256)==g for r in rows) for g in [False,True]],formal_sha256=hashlib.sha256((R/'formal.json').read_bytes()).hexdigest(),bootstrap_seed=137,bootstrap_replicates=10000)
 put('confirmation_predictions.json',pred);put('confirmation_results.json',results);print(json.dumps(results,indent=2))
if __name__=='__main__':main()
