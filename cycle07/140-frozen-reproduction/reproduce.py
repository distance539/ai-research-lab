"""Single offline entry; independent replay in a clean temporary directory."""
import json,subprocess,sys,tempfile,shutil,hashlib,time,platform
from pathlib import Path
from freeze import verify
R=Path(__file__).resolve().parent
verify();start=time.perf_counter();commands=[]
with tempfile.TemporaryDirectory() as td:
 d=Path(td)/'code';shutil.copytree(R,d,ignore=shutil.ignore_patterns('__pycache__'))
 for script in ['analyze.py','audit.py','damage_tests.py']:
  r=subprocess.run([sys.executable,script],cwd=d,text=True,capture_output=True)
  commands.append(dict(command=[sys.executable,script],exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr))
  if r.returncode:raise RuntimeError(commands[-1])
 exact={n:hashlib.sha256((d/n).read_bytes()).hexdigest()==hashlib.sha256((R/n).read_bytes()).hexdigest() for n in ['predictions.json','results.json']}
 assert all(exact.values())
 for n in ['audit_result.json','damage_result.json']:shutil.copy2(d/n,R/n)
 result=dict(status='passed',commands=commands,exact=exact,seconds=time.perf_counter()-start,python=sys.version,platform=platform.platform(),confirmation_accessed=False)
 (R/'reproduction_result.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k!='commands'},indent=2))
