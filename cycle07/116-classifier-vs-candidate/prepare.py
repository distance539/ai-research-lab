import argparse,json,hashlib,urllib.request
from pathlib import Path
p=Path(__file__).resolve().parent;a=argparse.ArgumentParser();a.add_argument('--cache',type=Path,default=Path('.cache'));a=a.parse_args();a.cache.mkdir(parents=True,exist_ok=True)
def get(url,target,sha):
 target.parent.mkdir(parents=True,exist_ok=True)
 if not target.exists():urllib.request.urlretrieve(url,target)
 assert hashlib.sha256(target.read_bytes()).hexdigest()==sha,str(target)
m=json.loads((p/'model.lock.json').read_text())
for f in m['files']:get(f"https://huggingface.co/{m['model']}/resolve/{m['revision']}/{f['name']}",a.cache/'model'/f['name'],f['sha256'])
d=json.loads((p/'resources.lock.json').read_text())['dataset'];get(d['url'],a.cache/'validation.parquet',d['sha256']);print('resources verified')
