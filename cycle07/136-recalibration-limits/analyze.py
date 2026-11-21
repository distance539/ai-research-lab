import json,math,hashlib
from pathlib import Path
from metrics import sigmoid,metric
R=Path(__file__).resolve().parent
def read(n):return json.loads((R/n).read_text())
def prob(r,t):return sigmoid((r['logits'][1]-r['logits'][0])/t)
def main():
 cfg=read('protocol.json');dev=read('exploration.json');cal=read('calibration.json');out={};details={}
 for name,t in [('raw',1),('source',cfg['fixed_T']),('target',cfg['target_T'])]:
  scores=sorted(1-(prob(r,t) if r['label'] else 1-prob(r,t)) for r in cal);k=math.ceil((len(cal)+1)*(1-cfg['alpha']));q=scores[k-1]
  rows=[]
  for r in dev:
   p=prob(r,t);pred=int(p>.5);c=max(p,1-p);s=[y for y,v in enumerate([1-p,p]) if 1-v<=q]
   rows.append(dict(id=r['id'],confidence=c,error=int(pred!=r['label']),accepted=c>=.9,labels=s,covered=r['label'] in s))
  rank=sorted(range(len(rows)),key=lambda i:(-rows[i]['confidence'],rows[i]['id']));prefix=[];err=0
  for i in rank:err+=rows[i]['error'];prefix.append(err)
  out[name]=dict(temperature=t,**metric(dev,t),accepted=sum(r['accepted'] for r in rows),accepted_errors=sum(r['accepted']*r['error'] for r in rows),q=q,k=k,covered=sum(r['covered'] for r in rows),sizes=[sum(len(r['labels'])==k for r in rows) for k in range(3)],rank=[rows[i]['id'] for i in rank],prefix_errors=prefix)
  details[name]=rows
 base=out['raw'];out['comparisons']={n:dict(rank_equal=out[n]['rank']==base['rank'],prefix_equal=out[n]['prefix_errors']==base['prefix_errors'],set_changes=sum(a['labels']!=b['labels'] for a,b in zip(details['raw'],details[n]))) for n in ['source','target']}
 out['inputs']={n:hashlib.sha256((R/n).read_bytes()).hexdigest() for n in ['protocol.json','calibration.json','exploration.json','pools.json','analyze.py','metrics.py']}
 (R/'results.json').write_text(json.dumps(out,indent=2));(R/'decisions.json').write_text(json.dumps(details,indent=2));print(json.dumps({n:{k:v for k,v in x.items() if k not in ['rank','prefix_errors']} for n,x in out.items()},indent=2))
if __name__=='__main__':main()
