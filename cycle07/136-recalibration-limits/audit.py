import json,math,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent
read=lambda n:json.loads((R/n).read_text())
def main():
 dev=read('exploration.json');cal=read('calibration.json');res=read('results.json');dec=read('decisions.json');pools=read('pools.json')
 assert [r['id'] for r in dev]==[r['id'] for r in pools['exploration']]
 assert [r['id'] for r in cal]==[r['id'] for r in pools['calibration']]
 def probs(r,t):
  z=[v/t for v in r['logits']];a=max(z);w=[math.exp(v-a) for v in z];return [v/sum(w) for v in w]
 for name in ['raw','source','target']:
  v=res[name];t=v['temperature'];q=sorted(1-probs(r,t)[r['label']] for r in cal)[900];loss=0;hits=0
  for r,d in zip(dev,dec[name]):
   ps=probs(r,t);s=[y for y in range(2) if 1-ps[y]<=q];assert s==d['labels'];assert abs(max(ps)-d['confidence'])<1e-14;loss-=math.log(ps[r['label']]);hits+=max(range(2),key=lambda y:r['logits'][y])==r['label']
  assert abs(loss/1000-v['nll'])<1e-12 and hits==v['hits']
  order=sorted(range(1000),key=lambda i:(-abs(dev[i]['logits'][1]-dev[i]['logits'][0]),dev[i]['id']))
  assert [dev[i]['id'] for i in order]==v['rank']
  assert sum(r['covered'] for r in dec[name])==v['covered']
 smoke=R/'smoke.json'
 if smoke.exists():
  sr=read('smoke.json')
  assert all(all(a[k]==b[k] for k in ['id','label','retained_tokens']) for a,b in zip(sr,dev))
  assert max(abs(x-y) for a,b in zip(sr,dev) for x,y in zip(a['logits'],b['logits']))<1e-5
 out={'status':'passed','rows':3000,'memberships':6000,'method':'independent two-class softmax and absolute-margin ordering','inputs':{n:hashlib.sha256((R/n).read_bytes()).hexdigest() for n in ['results.json','decisions.json','audit.py','exploration.json','calibration.json']}}
 (R/'audit_result.json').write_text(json.dumps(out,indent=2));print(out)
if __name__=='__main__':main()
