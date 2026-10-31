"""Independent arithmetic reconstruction from logits, without analysis imports."""
import json,math,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent
rows=json.loads((P/'formal_predictions.json').read_text()); result=json.loads((P/'group_results.json').read_text()); cfg=json.loads((P/'analysis_config.json').read_text())
assert hashlib.sha256((P/'formal_predictions.json').read_bytes()).hexdigest()==cfg['prediction_sha256']
assert len(rows)==256 and len({r['idx'] for r in rows})==256
reconstructed=[]
for r in rows:
 z=r['logits']; pred=int(z[1]>z[0]); conf=1/(1+math.exp(-abs(z[1]-z[0])))
 assert pred==r['prediction'] and int(pred==r['label'])==r['correct'] and abs(conf-r['confidence'])<1e-12
 reconstructed.append((r,pred,conf))
def include(r,name):
 parts=name.split('/')
 return all(k=='all' or k=='short' and r['raw_tokens']<=16 or k=='medium' and 16<r['raw_tokens']<=32 or k=='long' and r['raw_tokens']>32 or k=='label0' and r['label']==0 or k=='label1' and r['label']==1 for k in parts)
assert len(result['groups'])==12
for g in result['groups']:
 rr=[x for x in reconstructed if include(x[0],g['group'])]; hi=[x for x in rr if x[2]>=.9]; errors=[x[0]['idx'] for x in hi if x[1]!=x[0]['label']]
 assert (len(rr),sum(p!=r['label'] for r,p,c in rr),len(hi),errors)==(g['n'],g['errors'],g['high_n'],g['high_error_ids'])
 assert g['high_errors']==len(errors) and g['high_error_rate']==(len(errors)/len(hi) if hi else None)
expected=sorted([x for x in reconstructed if x[1]!=x[0]['label']],key=lambda x:(-x[2],x[0]['idx']))[:5]
assert [x[0]['idx'] for x in expected]==[x['idx'] for x in result['top_errors']]
report={'passed':True,'n':256,'groups':12,'correct':sum(r['correct'] for r in rows),'input_sha256':cfg['prediction_sha256'],'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'method':'independent binary sigmoid, argmax, original-label and raw-token predicates'}
(P/'group_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
