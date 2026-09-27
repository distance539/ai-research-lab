"""Download pinned public resources; verify all bytes before use."""
import argparse,tarfile,urllib.request,zipfile
from pathlib import Path
from common import ROOT,read_json,sha,verify_resources

def fetch(url,target,expected):
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        assert sha(target)==expected, f'Existing file hash mismatch: {target}'
        return
    tmp=target.with_name(target.name+'.partial')
    with urllib.request.urlopen(url,timeout=120) as r,tmp.open('wb') as f:
        while chunk:=r.read(1<<20):f.write(chunk)
    assert sha(tmp)==expected,f'Download hash mismatch: {target}'
    tmp.replace(target)

def prepare(cache):
    lock=read_json(ROOT/'resources.lock.json');model=read_json(ROOT/'model.lock.json')
    for name in ['beir','scifact']:
        r=lock['resources'][name];archive=cache/'downloads'/r['filename'];fetch(r['url'],archive,r['sha256'])
        if name=='beir':
            dest=cache/'upstream';dest.mkdir(parents=True,exist_ok=True)
            if not (dest/('beir-'+lock['beir_commit'])).exists():
                with tarfile.open(archive) as t:t.extractall(dest,filter='data')
        elif not (cache/'scifact/corpus.jsonl').exists():
            with zipfile.ZipFile(archive) as z:
                for n in z.namelist():assert not Path(n).is_absolute() and '..' not in Path(n).parts
                z.extractall(cache)
    for f,r in model['files'].items():fetch(r['url'],cache/'models'/model['revision']/f,r['sha256'])
    verify_resources(cache);print('Pinned data, BEIR and model files verified.')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);prepare(p.parse_args().cache)
