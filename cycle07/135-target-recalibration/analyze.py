import json,math,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent
def read(n):return json.loads((R/n).read_text())
def sigmoid(x):return 1/(1+math.exp(-x)) if x>=0 else math.exp(x)/(1+math.exp(x))
def grad(rows,b):return sum((r['logits'][1]-r['logits'][0])*(sigmoid(b*(r['logits'][1]-r['logits'][0]))-r['label']) for r in rows)/len(rows)
def fit(rows):
 lo,hi=read('protocol.json')['beta_bounds'];trace=[]
 if grad(rows,lo)>=0:return dict(beta=lo,temperature=1/lo,boundary=True,trace=[])
 if grad(rows,hi)<=0:return dict(beta=hi,temperature=1/hi,boundary=True,trace=[])
 for i in range(100):
  mid=(lo+hi)/2;g=grad(rows,mid);trace.append([mid,g])
  if g>0:hi=mid
  else:lo=mid
 b=(lo+hi)/2
 return dict(beta=b,temperature=1/b,boundary=False,trace=trace)
def metric(rows,t):
 losses=[];br=[];hits=0;bins=[[0,0,0.] for _ in range(10)]
 for r in rows:
  x=(r['logits'][1]-r['logits'][0])/t;p=sigmoid(x);ok=int((x>0)==r['label']);hits+=ok;c=max(p,1-p)
  losses.append(max(x,0)+math.log1p(math.exp(-abs(x)))-r['label']*x);br.append(2*(p-r['label'])**2);v=bins[min(9,int(c*10))];v[0]+=1;v[1]+=ok;v[2]+=c
 return dict(n=len(rows),hits=hits,nll=sum(losses)/len(rows),brier=sum(br)/len(rows),ece10=sum(abs(k-s) for n,k,s in bins)/len(rows))
def main():
 cfg=read('protocol.json');cal=read('formal.json');dev=read('exploration.json');pools=read('pools.json')
 assert [r['id'] for r in cal]==[r['id'] for r in pools['calibration']]
 assert [r['id'] for r in dev]==[r['id'] for r in pools['exploration']]
 out=dict(baselines={k:metric(dev,t) for k,t in [('raw',1),('source',cfg['fixed_T'])]},budgets=[])
 for n in cfg['budgets']:
  f=fit(cal[:n]);out['budgets'].append(dict(n=n,**f,calibration=metric(cal[:n],f['temperature']),exploration=metric(dev,f['temperature'])))
 out['inputs']={n:hashlib.sha256((R/n).read_bytes()).hexdigest() for n in ['protocol.json','pools.json','formal.json','exploration.json','analyze.py']}
 (R/'results.json').write_text(json.dumps(out,indent=2));print(json.dumps({**out,'budgets':[{k:v for k,v in x.items() if k!='trace'} for x in out['budgets']]},indent=2))
if __name__=='__main__':main()
