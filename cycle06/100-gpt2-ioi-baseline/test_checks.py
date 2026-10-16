"""Deliberately damage independent evidence; each mutation must be rejected."""
import json,tempfile,shutil,copy
from pathlib import Path
from audit import audit
ROOT=Path(__file__).resolve().parent

def main():
 passed=[]
 for key,value in [('logit_diff',1000),('io_position',0),('IO','wrong'),('io_prob',0.000001)]:
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);shutil.copy(ROOT/'results/summary.json',p/'summary.json');r=json.loads((ROOT/'results/records.json').read_text());r[0][key]=value;(p/'records.json').write_text(json.dumps(r))
   try:audit(p)
   except AssertionError:passed.append(key)
   else:raise AssertionError('mutation not detected: '+key)
 print(json.dumps({'status':'passed','rejected_corruptions':passed,'count':len(passed)},indent=2))
if __name__=='__main__':main()
