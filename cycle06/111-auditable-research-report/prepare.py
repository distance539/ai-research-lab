"""Hash-locked data, BEIR source, two models and optional macOS-arm64 ES."""
import argparse,urllib.request,tarfile,zipfile
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
def main(cache,es=False):
 lock=read(ROOT/'resources.lock.json')
 for name,r in lock['resources'].items():
  if name.startswith('elasticsearch') and not es:continue
  z=cache/'downloads'/r['filename'];fetch(r['url'],z,r['sha256'])
  if name=='scifact':
   if not (cache/'scifact/corpus.jsonl').exists():
    with zipfile.ZipFile(z) as f:
     for n in f.namelist():assert not Path(n).is_absolute() and '..' not in Path(n).parts
     f.extractall(cache)
  else:
   dest=cache/'upstream' if name=='beir' else cache;target=dest/('beir-'+lock['beir_commit'] if name=='beir' else 'elasticsearch-7.17.9')
   if not target.exists():
    dest.mkdir(parents=True,exist_ok=True)
    with tarfile.open(z) as f:f.extractall(dest,filter='data')
 for lockname in ['model.lock.json','dense_model.lock.json']:
  m=read(ROOT/lockname)
  for name,r in m['files'].items():fetch(r['url'],cache/'models'/m['revision']/name,r['sha256'])
 verify(cache)
 for rel,h in lock['beir_files'].items():assert sha(cache/'upstream'/('beir-'+lock['beir_commit'])/rel)==h,rel
 print('Data, both models, BEIR fixed source and inputs verified; no inference.')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--with-es-macos-arm64',action='store_true');a=p.parse_args();main(a.cache,a.with_es_macos_arm64)
