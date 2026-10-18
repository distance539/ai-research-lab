"""Reject plausible corruption of intervention evidence; never invent human review."""
import json,copy
from pathlib import Path
from audit import validate
p=Path(__file__).resolve().parent/'smoke'
original=[json.loads((p/f).read_text()) for f in ['records.json','patches.json','summary.json']]

def tests():
 def wrong_candidate(r,p,s):r[0]['io_id']=r[0]['s_id']
 def wrong_position(r,p,s):r[0]['s2_position']=0
 def changed_gain(r,p,s):p[0]['gain']+=1
 def missing_cell(r,p,s):p.pop()
 def duplicate_cell(r,p,s):p[-1]=copy.deepcopy(p[0])
 def false_control(r,p,s):r[0]['full']['diff']+=1
 def false_rank(r,p,s):s['conditions']['s2_flip']['ranking'].reverse()
 def false_ratio(r,p,s):s['conditions']['s2_flip']['ranking'][0]['recovery_ratio']+=.1
 def false_correlation(r,p,s):s['comparison']['ordinal_rank_correlation']+=.1
 def false_overlap(r,p,s):s['comparison']['top10_overlap']+=.1
 def false_condition(r,p,s):p[0]['condition']='invented'
 for f in [false_correlation,false_overlap,false_condition,wrong_candidate,wrong_position,changed_gain,missing_cell,duplicate_cell,false_control,false_rank,false_ratio]:
  data=copy.deepcopy(original);f(*data)
  try:validate(*data)
  except (AssertionError,KeyError):print('rejected',f.__name__)
  else:raise AssertionError('accepted corrupted evidence '+f.__name__)
 print('11 mutation tests passed')
if __name__=='__main__':tests()
