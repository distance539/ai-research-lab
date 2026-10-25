"""Reject material evidence corruption; seven changes to otherwise valid outputs."""
import argparse,json,shutil,tempfile
from pathlib import Path
from audit import check
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();check(a.out)
mutations=[('pairs.json',lambda x:x.pop()),('pairs.json',lambda x:x.append(x[0])),('pairs.json',lambda x:x[0].update(score=x[0]['score']+1)),('per_query.json',lambda x:x[0]['effects'].update(interaction=1)),('summary.json',lambda x:x.update(confirmation_evaluated=True)),('budget_ledger.json',lambda x:x.update(extra_pairs_for_B=1)),('fusion_trace.json',lambda x:x[0].update(prior_rank=99))];results=[]
for i,(f,change) in enumerate(mutations):
 with tempfile.TemporaryDirectory() as tmp:
  d=Path(tmp)/'out';shutil.copytree(a.out,d);x=json.loads((d/f).read_text());change(x);(d/f).write_text(json.dumps(x))
  try:check(d)
  except (AssertionError,KeyError,IndexError):results.append({'case':i,'file':f,'rejected':True})
  else:raise AssertionError('corruption accepted '+str(i))
(a.out/'mutation_tests.json').write_text(json.dumps({'status':'passed','tests':results},indent=2)+'\n');print('7/7 corruptions rejected')
