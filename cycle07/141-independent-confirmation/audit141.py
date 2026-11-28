"""Independent two-logit softmax audit; does not import evaluator."""
import math,json,hashlib
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent
def read(n):return json.loads((R/n).read_text())
def main():
 rows=read('formal.json'); ps=read('confirmation_predictions.json'); result=read('confirmation_results.json');cfg=read('final_config.json');d=[]
 for i,r in enumerate(rows):
  losses=[]
  for name in ['global','group']:
   t=cfg['temperatures']['global'] if name=='global' else cfg['temperatures']['length256'][int(r['retained_tokens']>256)]
   z=[v/t for v in r['logits']];m=max(z);ex=[math.exp(v-m) for v in z];p=[v/sum(ex) for v in ex];loss=-math.log(p[r['label']]);q=ps[name]['rows'][i]
   assert q['id']==r['id'] and abs(q['loss']-loss)<1e-12
   assert abs(q['confidence']-max(p))<1e-12 and q['correct']==int(max(range(2),key=lambda j:z[j])==r['label'])
   assert abs(q['brier']-sum((v-int(j==r['label']))**2 for j,v in enumerate(p)))<1e-12
   losses.append(loss)
  d.append(losses[1]-losses[0])
 for name in ps:
  rr=ps[name]['rows'];order=sorted(rr,key=lambda r:(-r['confidence'],r['id']));assert [x['id'] for x in order]==ps[name]['order'];assert sum(1-r['correct'] for r in order[:800])==result['cells'][name]['errors80']
  assert abs(sum(r['loss'] for r in rr)/1000-result['cells'][name]['nll'])<1e-12
 rng=np.random.default_rng(137);v=np.array(d);bs=[]
 for _ in range(10000):bs.append(float(np.sum(v[rng.integers(1000,size=1000)])/1000))
 ci=np.percentile(bs,[2.5,97.5]);assert np.max(np.abs(ci-result['ci95']))<1e-12
 assert abs(sum(d)/1000-result['delta_nll'])<1e-12
 out=dict(status='passed',rows=1000,method_predictions=2000,bootstrap_replicates=10000,input_sha256={n:hashlib.sha256((R/n).read_bytes()).hexdigest() for n in ['formal.json','final_config.json','confirmation_predictions.json','confirmation_results.json']},audit_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),human_review=False)
 (R/'audit141_result.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
