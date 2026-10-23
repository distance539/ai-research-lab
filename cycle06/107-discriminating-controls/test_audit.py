"""Evidence mutation tests: altered score, candidate, identity, arithmetic or approval rejected."""
import argparse,json,tempfile,shutil
from pathlib import Path
from audit import check
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();cases=[]
for kind in ['score','candidate','identity','delta','token','confirmation','false_review']:
 with tempfile.TemporaryDirectory() as tmp:
  d=Path(tmp)/'result';shutil.copytree(a.out,d)
  f='runs.json' if kind in ['score','candidate'] else ('pairs.json' if kind in ['identity','token'] else 'summary.json');x=json.loads((d/f).read_text())
  if kind=='score':q=next(iter(x['front']));k=next(iter(x['front'][q]));x['front'][q][k]+=1
  elif kind=='candidate':q=next(iter(x['front']));x['front'][q].pop(next(iter(x['front'][q])))
  elif kind=='identity':x[0]['input_sha256']='bad'
  elif kind=='token':x[0]['retained_tokens']+=1
  elif kind=='delta':x['difference_of_group_deltas']+=0.1
  elif kind=='confirmation':x['confirmation_evaluated']=True
  else:x['semantic_review']='passed'
  (d/f).write_text(json.dumps(x))
  try:check(d)
  except (AssertionError,KeyError):cases.append({'mutation':kind,'rejected':True})
  else:raise AssertionError(kind+' escaped')
(a.out/'mutation_tests.json').write_text(json.dumps(cases,indent=2)+'\n');print(json.dumps(cases))
