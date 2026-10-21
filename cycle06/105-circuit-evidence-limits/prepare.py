"""Download immutable public assets and verify every SHA-256; no auth required."""
import argparse,hashlib,json,urllib.request
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def prepare(cache):
 cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
 lock=json.loads(Path(__file__).with_name('resources.lock.json').read_text())
 for row in lock['files']:
  p=cache/row['file']
  if not p.exists():
   tmp=p.with_suffix(p.suffix+'.part');urllib.request.urlretrieve(row['url'],tmp);tmp.rename(p)
  assert sha(p)==row['sha256'],f'asset hash mismatch: {p.name}'
 return lock
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--cache',required=True);args=a.parse_args();print(json.dumps(prepare(args.cache),indent=2))
