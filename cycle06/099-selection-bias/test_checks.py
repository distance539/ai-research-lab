"""Negative tests for arithmetic, row identity, and development-only selection."""
import copy,json
from common import ROOT,load,save
from audit import check_rows
import runpy
choose=runpy.run_path(str(ROOT/'select.py'))['choose']
rows=load(ROOT/'development/records.json');cfg=load(ROOT/'config.json');plan=load(ROOT/'plan.json');names=cfg['conditions'];ids=plan['development'];out=[]
check_rows(rows,names,ids)
def rejects(name,modify):
    x=copy.deepcopy(rows);modify(x)
    try:check_rows(x,names,ids)
    except (AssertionError,KeyError):out.append({'case':name,'passed':True});return
    raise AssertionError(name+' accepted corrupted records')
rejects('candidate_sum',lambda x:x[0]['candidates'][0].update(score=999))
rejects('prediction',lambda x:x[0].update(prediction=1-x[0]['prediction']))
rejects('semantic_mapping',lambda x:x[0].update(mapping=list(reversed(x[0]['mapping']))))
rejects('duplicate_id',lambda x:x.__setitem__(1,copy.deepcopy(x[0])))
rejects('gold_disagreement',lambda x:x[0].update(gold=1-x[0]['gold']))
rejects('missing_row',lambda x:x.pop())
rejects('holdout_leakage',lambda x:x[0].update(idx=plan['reserved'][0]))
try:choose(rows[:-1],cfg,plan)
except AssertionError:out.append({'case':'incomplete_search_rejected','passed':True})
assert len(out)==8
save(ROOT/'test_results.json',{'tests':out,'all_passed':True});print(json.dumps(out))
