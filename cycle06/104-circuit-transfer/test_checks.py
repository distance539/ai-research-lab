"""Seven adversarial mutations must be rejected by independent audit."""
import tempfile,json,shutil
from pathlib import Path
from audit import audit
ROOT=Path(__file__).resolve().parent
mutations=[('wrong_diff',lambda r:r[0]['zero'].__setitem__('diff',999)),('wrong_group',lambda r:r[0].__setitem__('group','new_new')),('extra_corruption',lambda r:r[0]['corrupt_ids'].__setitem__(0,0)),('wrong_end',lambda r:r[0].__setitem__('end_position',0)),('missing_row',lambda r:r.pop()),('self_patch_failed',lambda r:r[0]['controls'].__setitem__('self_error',.1)),('short_activation',lambda r:r[0]['clean_z']['A'].pop())]
def main():
 audit(ROOT/'results');passed=[]
 for name,mutate in mutations:
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)
   for f in ['records.json','summary.json','identity.json']:shutil.copyfile(ROOT/'results'/f,p/f)
   r=json.loads((p/'records.json').read_text());mutate(r);(p/'records.json').write_text(json.dumps(r))
   try:audit(p)
   except AssertionError:passed.append(name)
   else:raise AssertionError('corruption not detected: '+name)
 print(json.dumps({'passed':passed,'count':len(passed)}))
if __name__=='__main__':main()
