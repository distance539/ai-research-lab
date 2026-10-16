"""Independent stdlib arithmetic/token-position audit; not a human review."""
import argparse,json,math,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def audit(folder):
 rows=json.loads((folder/'records.json').read_text());summary=json.loads((folder/'summary.json').read_text());samples=json.loads((ROOT/'samples.json').read_text())
 assert len(rows)==len(set(r['id'] for r in rows))
 for r in rows:
  src=samples[r['id']]
  for k,v in src.items():assert r[k]==v,(r['id'],k)
  ids=r['token_ids'];assert ids[r['io_position']]==r['io_id'] and ids.count(r['io_id'])==1
  assert [i for i,v in enumerate(ids) if v==r['s_id']]==r['s_positions'] and len(r['s_positions'])==2
  assert r['end_position']==len(ids)-1 and r['logits_shape']==[1,len(ids),50257] and r['hidden_shape']==[768]
  assert hashlib.sha256(r['prompt'].encode()).hexdigest()==r['prompt_sha256']
  d=r['io_logit']-r['s_logit'];assert abs(d-r['logit_diff'])<1e-5
  assert abs(1/(1+math.exp(-d))-r['pair_prob'])<1e-6
  assert abs(math.log(r['io_prob']/r['s_prob'])-d)<1e-5
  assert r['top1_logit']>=r['io_logit'] and (r['top1_id']==r['io_id'])==(r['io_rank']==1)
  assert r['readout_max_error']<1e-4
 for key,rr in [('overall',rows)]+[(k,[r for r in rows if f"{r['template_index']}-{r['order']}"==k]) for k in summary['cells']]:
  s=summary['overall'] if key=='overall' else summary['cells'][key]
  assert s['n']==len(rr) and s['positive']==sum(r['io_logit']>r['s_logit'] for r in rr)
  assert s['top1_correct']==sum(r['top1_id']==r['io_id'] for r in rr)
  for label,field in [('mean_logit_diff','logit_diff'),('mean_io_prob','io_prob'),('mean_s_prob','s_prob'),('mean_pair_prob','pair_prob')]:assert abs(s[label]-sum(r[field] for r in rr)/len(rr))<1e-8
 if len(rows)==64:
  assert len(summary['cells'])==4 and all(v['n']==16 for v in summary['cells'].values())
  for r in rows:
   mates=[q for q in rows if q['template_index']==r['template_index'] and q['order']==r['order'] and q['pair_id']==r['pair_id'] and q['swapped']!=r['swapped']]
   assert len(mates)==1 and mates[0]['IO']==r['S'] and mates[0]['S']==r['IO']
 return {'status':'passed','rows':len(rows),'input_sha256':hashlib.sha256((folder/'records.json').read_bytes()).hexdigest(),'human_review':False}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--results',default=str(ROOT/'results'));a=p.parse_args();print(json.dumps(audit(Path(a.results)),indent=2))
