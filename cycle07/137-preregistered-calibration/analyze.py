"""Protocol feasibility audit; deliberately does not fit either candidate."""
import json,hashlib,math
from pathlib import Path
R=Path(__file__).resolve().parent
H=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def read(n):return json.loads((R/n).read_text())
def validate(pools, rows, name):
 for key in ['id','movie','text']:
  assert len({r[key] for a in pools.values() for r in a})==3000
 assert [r['id'] for r in rows]==[r['id'] for r in pools[name]]
 assert all(r['label'] in (0,1) and len(r['logits'])==2 and all(math.isfinite(v) for v in r['logits']) and 2<=r['retained_tokens']<=512 for r in rows)
def main():
 assert H(R/'protocol.json')==(R/'protocol.sha256').read_text().strip()
 p=read('pools.json');out={'inputs':{n:H(R/n) for n in ['protocol.json','pools.json','exploration.json','calibration.json']},'confirmation_predictions_read':False,'candidate_fits':0}
 for name in ['exploration','calibration']:
  rows=read(name+'.json');validate(p,rows,name)
  orders=[]
  for t in [1,read('protocol.json')['fixed_T'],read('protocol.json')['target_T']]:
   orders.append([r['id'] for r in sorted(rows,key=lambda r:(-abs(r['logits'][1]-r['logits'][0])/t,r['id']))])
  out[name]={'n':len(rows),'group_counts':[sum(r['retained_tokens']<=256 for r in rows),sum(r['retained_tokens']>256 for r in rows)],'global_margin_order_identical':orders[0]==orders[1]==orders[2]}
  assert out[name]['global_margin_order_identical']
 (R/'results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
