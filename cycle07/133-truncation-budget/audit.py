"""Independent softmax/count reconstruction, no imports from run/common."""
import json,math,hashlib
from pathlib import Path
p=Path(__file__).resolve().parent
read=lambda n:json.loads((p/n).read_text())
rows=read('formal.json');result=read('results.json');pool=read('pools.json');base={r['id']:r for r in rows if r['length']==512}
assert len(base)==1000 and set(base)=={r['id'] for r in pool['exploration']}
assert hashlib.sha256((p/'pools.json').read_bytes()).hexdigest()=='d472a0676f5f9b58aea434ce694ddd40ea2d5b4ab50f9c3aa28aec896d982764'
def calc(r):
 z=r['logits'];m=max(z);v=[math.exp(x-m) for x in z];v=[x/sum(v) for x in v];pred=max(range(2),key=lambda i:z[i]);return pred,max(v),-math.log(v[r['label']]),sum((x-(i==r['label']))**2 for i,x in enumerate(v))
for length in [128,256,512]:
 group=[r for r in rows if r['length']==length];assert len(group)==1000
 hit=flips=up=down=drop=trunc=ret=accepted=errors=0;nll=brier=delta=0.;bins=[[0,0,0.] for _ in range(10)]
 for r in group:
  b=base[r['id']];assert r['label']==b['label'] and r['full_tokens']==b['full_tokens'];assert r['retained_tokens']==min(length,r['full_tokens']+2);assert r['dropped']==r['full_tokens']-r['retained_tokens']+2
  pred,c,l,br=calc(r);bp,bc,_,_=calc(b);ok=pred==r['label'];hit+=ok;flips+=pred!=bp;up+=ok and bp!=r['label'];down+=not ok and bp==r['label'];delta+=abs(c-bc);nll+=l;brier+=br;drop+=r['dropped'];trunc+=r['dropped']>0;ret+=r['retained_tokens'];v=bins[min(9,int(c*10))];v[0]+=1;v[1]+=ok;v[2]+=c
  if c>=.9:accepted+=1;errors+=not ok
 out=result[str(length)];sc=out['score']
 for k,v in dict(hits=hit,nll=nll/1000,brier=brier/1000,ece=sum(abs(b[1]-b[2]) for b in bins)/1000,accepted=accepted,errors=errors).items():assert abs(sc[k]-v)<1e-12,(length,k)
 for k,v in dict(flips=flips,mean_abs_conf_delta=delta/1000,improved=up,worsened=down,dropped=drop,truncated=trunc,retained=ret).items():assert abs(out[k]-v)<1e-12,(length,k)
 assert flips==up+down
print(json.dumps({'status':'passed','rows':len(rows),'input_sha256':hashlib.sha256((p/'formal.json').read_bytes()).hexdigest(),'checks':'two-column softmax, label argmax, paired flips, metrics, retained and discarded tokens, locked exploration identity'}))
