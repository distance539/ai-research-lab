"""Negative tests must reject broken evidence, not merely run happy paths."""
import json,tempfile,shutil
from pathlib import Path
from audit import audit
ROOT=Path(__file__).resolve().parent
cases=[('score',lambda d:d[0]['clean'].__setitem__('diff',0)),('identity',lambda d:d[0].__setitem__('prompt','wrong')),('missing_subset',lambda d:d[0]['conditions']['zero']['full'].pop('AB')),('duplicate_row',lambda d:d.append(d[0])),('argmax_range',lambda d:d[0]['clean'].__setitem__('top1_id',999999)),('self_control',lambda d:d[0]['controls'].__setitem__('all_head_self_error',1)),('nan_score',lambda d:d[0]['conditions']['mean_abc']['retained'][''].__setitem__('io_logit',float('nan')))]
passed=[]
with tempfile.TemporaryDirectory() as tmp:
 p=Path(tmp)/'evidence';shutil.copytree(ROOT/'smoke',p);audit(p);original=(p/'records.json').read_text()
 for name,mutate in cases:
  d=json.loads(original);mutate(d);(p/'records.json').write_text(json.dumps(d))
  try:audit(p)
  except (AssertionError,KeyError,ValueError):passed.append(name)
  else:raise AssertionError('corruption accepted: '+name)
 (p/'records.json').write_text(original);d=json.loads((p/'summary.json').read_text());d['old_old']['modes']['zero']['subsets']['']['row_mae']+=1;(p/'summary.json').write_text(json.dumps(d))
 try:audit(p)
 except AssertionError:passed.append('aggregate_mae')
 else:raise AssertionError('corrupt aggregate accepted')
print(json.dumps({'status':'passed','rejected_mutations':passed,'count':len(passed)}))
