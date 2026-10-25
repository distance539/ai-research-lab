"""Download fixed model/data into an explicit cache; hash-check before use."""
import argparse,urllib.request,zipfile,tarfile
from pathlib import Path
from common import ROOT,read,sha,verify

def fetch(url,p,h):
 p.parent.mkdir(parents=True,exist_ok=True)
 if not p.exists():
  t=p.with_name(p.name+'.partial')
  with urllib.request.urlopen(url,timeout=180) as r,t.open('wb') as f:
   while b:=r.read(1<<20):f.write(b)
  assert sha(t)==h,str(p);t.replace(p)
 assert sha(p)==h,str(p)
def main(cache):
 r=read(ROOT/'resources.lock.json')['resources']['scifact'];z=cache/'downloads'/r['filename'];fetch(r['url'],z,r['sha256'])
 if not (cache/'scifact/corpus.jsonl').exists():
  with zipfile.ZipFile(z) as f:
   for n in f.namelist():assert not Path(n).is_absolute() and '..' not in Path(n).parts
   f.extractall(cache)
 original=read(ROOT/'resources.lock.json')['resources']['original'];archive=cache/'downloads'/original['filename'];fetch(original['url'],archive,original['sha256'])
 if not (cache/'downloads/data/claims_train.jsonl').exists():
  with tarfile.open(archive) as f:
   for member in f.getmembers():
    if member.name in ['data/corpus.jsonl','data/claims_train.jsonl']:
     assert member.isfile();target=cache/'downloads'/member.name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(f.extractfile(member).read())
 for name,h in read(ROOT/'resources.lock.json')['original_files'].items():assert sha(cache/name)==h
 m=read(ROOT/'model.lock.json')
 for name,r in m['files'].items():fetch(r['url'],cache/'models'/m['revision']/name,r['sha256'])
 verify(cache);print('Model/tokenizer/data hashes passed; no confirmation evaluation.')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);main(p.parse_args().cache)
