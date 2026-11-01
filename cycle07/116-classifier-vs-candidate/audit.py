import json,hashlib,math
from pathlib import Path
p=Path(__file__).resolve().parent
read=lambda n:json.loads((p/n).read_text())
a=read('formal_candidate_predictions.json');b=read('formal_predictions.json');s=read('formal_comparison.json');assert len(a)==len(b)==256
for x,y in zip(a,b):
 assert (x['idx'],x['label'],x['text_sha256'])==(y['idx'],y['label'],y['text_sha256'])
 assert all(len(t)==1 for t in x['candidate_ids'])
 for v,t in zip(x['scores'],x['token_logprobs']):assert abs(v-math.fsum(t))<1e-12
 pos=1/(1+math.exp(x['scores'][0]-x['scores'][1]));q=[1-pos,pos]
 assert max(abs(u-v) for u,v in zip(q,x['probabilities']))<1e-12
 pred=int(pos>0.5);assert pred==x['prediction'];assert x['correct']==int(pred==x['label'])
 assert abs(math.fsum(math.exp(v) for v in x['scores'])-x['candidate_mass'])<1e-12
for rows,name in [(a,'candidate'),(b,'classifier')]:
 correct=sum(int(x['prediction']==x['label']) for x in rows);assert correct==s[name]['correct']
 for key,val in [('mean_confidence',sum(max(x['probabilities']) for x in rows)/256),('nll',sum(-math.log(x['probabilities'][x['label']]) for x in rows)/256),('brier',sum(2*(x['probabilities'][1]-x['label'])**2 for x in rows)/256)]:assert abs(val-s[name][key])<1e-12
report={'status':'passed','rows':256,'kind':'independent deterministic arithmetic, not human review','input_hashes':{n:hashlib.sha256((p/n).read_bytes()).hexdigest() for n in ['formal_candidate_predictions.json','formal_predictions.json','formal_comparison.json']},'candidate_predictions_by_label':[sum(x['prediction']==k for x in a) for k in [0,1]]}
(p/'audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
