"""Evidence-corruption checks for the independent fusion audit."""
import argparse,shutil,tempfile,json
from pathlib import Path
from audit import main
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();report=[]
for kind in ['score','missing_candidate','delta','budget','confirmation','trace','tradeoff']:
 with tempfile.TemporaryDirectory() as t:
  d=Path(t)/'results';shutil.copytree(a.out,d)
  fn={'score':'blend_runs.json','missing_candidate':'blend_runs.json','delta':'comparison_summary.json','budget':'budget_ledger.json','confirmation':'comparison_summary.json','trace':'fusion_trace.json','tradeoff':'comparison_summary.json'}[kind];x=json.loads((d/fn).read_text())
  if kind=='score':q=next(iter(x));x[q][next(iter(x[q]))]+=.01
  elif kind=='missing_candidate':q=next(iter(x));del x[q][next(iter(x[q]))]
  elif kind=='delta':x['comparisons']['blend_minus_ce']['mean']+=.01
  elif kind=='budget':x['blend_extra_model_pairs']=100
  elif kind=='confirmation':x['confirmation_evaluated']=not x['confirmation_evaluated']
  elif kind=='trace':x[0]['ce_rank']+=1
  else:x['override_tradeoff']['old_ce_losses']+=1
  (d/fn).write_text(json.dumps(x))
  try:main(d,False)
  except (AssertionError,KeyError,StopIteration):report.append({'mutation':kind,'rejected':True})
  else:raise AssertionError(kind+' escaped')
(a.out/'mutation_tests.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
