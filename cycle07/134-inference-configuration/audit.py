import json,math,hashlib
from pathlib import Path
from decimal import Decimal,getcontext
getcontext().prec=50
p=Path(__file__).parent;rows=json.loads((p/'formal.json').read_bytes());results=json.loads((p/'results.json').read_bytes());q=Decimal(str(json.loads((p/'protocol.json').read_bytes())['threshold']));sets={};counts={};min_margin=Decimal(1)
for mode in ['eager','sdpa']:
 g=[r for r in rows if r['condition']==mode];covered=hits=0;sizes=[0]*3;sets[mode]={}
 for r in g:
  z=[Decimal(str(x)) for x in r['logits']];e=[(x-max(z)).exp() for x in z];ps=[x/sum(e) for x in e];s=[i for i,x in enumerate(ps) if 1-x<=q];sets[mode][r['id']]=s;covered+=r['label'] in s;hits+=z.index(max(z))==r['label'];sizes[len(s)]+=1;min_margin=min(min_margin,*[abs(1-x-q) for x in ps])
 assert dict(n=len(g),hits=hits,covered=covered,size_counts=sizes)==results[mode]
assert sum(sets['eager'][k]!=sets['sdpa'][k] for k in sets['eager'])==results['paired']['set_changes']
out=dict(status='passed',rows=len(rows),membership_decisions=2*len(rows),input_sha256=hashlib.sha256((p/'formal.json').read_bytes()).hexdigest(),minimum_margin=str(min_margin),method='50-digit Decimal softmax independent of analysis sigmoid')
(p/'audit_result.json').write_text(json.dumps(out,indent=2));print(out)
