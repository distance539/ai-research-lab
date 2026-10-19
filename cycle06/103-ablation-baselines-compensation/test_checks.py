"""Meaningful damage tests against frozen scientific evidence."""
import tempfile,shutil,json
from pathlib import Path
from audit import audit
ROOT=Path(__file__).resolve().parent
cases=[('drop',lambda x:x['conditions']['zero']['AB'].__setitem__('mean_drop',999),'summary.json'),('interaction',lambda x:x['interactions']['mean_abc']['means'].__setitem__('I_AB',999),'summary.json'),('donor',lambda x:x[0].__setitem__('donor_index',99),'records.json'),('vector',lambda x:x[0]['replacement_z']['mean_abc']['A'].__setitem__(0,999),'records.json'),('candidate',lambda x:x[0]['clean'].__setitem__('diff',999),'records.json'),('missingrow',lambda x:x.pop(),'records.json'),('control',lambda x:x[0]['controls'].__setitem__('self_error',1),'records.json')]
for name,change,file in cases:
 with tempfile.TemporaryDirectory() as t:
  out=Path(t)/'results';shutil.copytree(ROOT/'results',out);p=out/file;x=json.loads(p.read_text());change(x);p.write_text(json.dumps(x))
  try:audit(ROOT,out)
  except (AssertionError,IndexError,KeyError):pass
  else:raise AssertionError('damage escaped: '+name)
print('passed',len(cases),'damage tests')
