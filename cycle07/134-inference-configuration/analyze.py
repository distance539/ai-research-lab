from common import *
r=read('formal.json');groups={k:[v for v in r if v['condition']==k] for k in ['eager','sdpa']};q=read('protocol.json')['threshold']
def probs(v):
 p=1/(1+math.exp(v['logits'][0]-v['logits'][1]));return [1-p,p]
def labels(v):return [i for i,p in enumerate(probs(v)) if 1-p<=q]
out={}
for k,g in groups.items():
 sets=[labels(v) for v in g];out[k]=dict(n=len(g),hits=sum(max(range(2),key=lambda i:v['logits'][i])==v['label'] for v in g),covered=sum(v['label'] in s for v,s in zip(g,sets)),size_counts=[sum(len(s)==i for s in sets) for i in range(3)])
a,b=groups.values();assert len(a)==len(b)==1000
for x,y in zip(a,b):assert (x['id'],x['input_sha256'],x['label'])==(y['id'],y['input_sha256'],y['label'])
out['paired']=dict(max_logit_delta=max(abs(u-v) for x,y in zip(a,b) for u,v in zip(x['logits'],y['logits'])),max_prob_delta=max(abs(u-v) for x,y in zip(a,b) for u,v in zip(probs(x),probs(y))),flips=sum((x['logits'][1]>x['logits'][0])!=(y['logits'][1]>y['logits'][0]) for x,y in zip(a,b)),set_changes=sum(labels(x)!=labels(y) for x,y in zip(a,b)),min_boundary_distance=min(abs(1-p-q) for x in a for p in probs(x)))
put('results.json',out);print(json.dumps(out,indent=2))
