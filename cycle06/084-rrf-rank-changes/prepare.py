"""Restore only fixed, licensed small inputs/modules; no credentials or large models."""
import argparse,hashlib,json,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def main(download):
    lock=json.loads((ROOT/'input.lock.json').read_text())
    vendor={'vendor/fusion_base.py':'pyserini/fusion/_base.py','vendor/trectools_base.py':'pyserini/trectools/_base.py','vendor/LICENSE.txt':'LICENSE.txt'}
    for rel,h in lock['files'].items():
        dest=ROOT/rel
        if not dest.exists():
            if not download:raise FileNotFoundError(rel+'; pass --download to restore')
            if rel in vendor:url='https://raw.githubusercontent.com/castorini/pyserini/'+lock['pyserini']['sha']+'/'+vendor[rel]
            else:url='https://raw.githubusercontent.com/distance539/ai-research-lab/'+lock['source_commit']+'/'+lock['source_path']+'/'+lock['source_files'][rel]
            with urllib.request.urlopen(url,timeout=60) as r:data=r.read()
            assert hashlib.sha256(data).hexdigest()==h,rel
            dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
        assert hashlib.sha256(dest.read_bytes()).hexdigest()==h,rel
    print('All 6 fixed input/module/license hashes verified.')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--download',action='store_true');main(p.parse_args().download)
