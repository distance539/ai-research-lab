"""Download only pinned public resources; verify cached bytes before any inference."""
import argparse, hashlib, json, pathlib, urllib.request
ROOT = pathlib.Path(__file__).resolve().parent

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while block := f.read(1048576): h.update(block)
    return h.hexdigest()

def prepare(cache, download=False):
    lock = json.loads((ROOT/'resources.lock.json').read_text())
    for name, spec in lock['files'].items():
        p = cache/name
        if not p.exists():
            if not download: raise FileNotFoundError(f'{name}: run prepare.py --cache ...')
            p.parent.mkdir(parents=True, exist_ok=True)
            temp = p.with_name(p.name+'.part')
            with urllib.request.urlopen(spec['url'], timeout=180) as r, temp.open('wb') as w:
                while b := r.read(1048576): w.write(b)
            assert digest(temp) == spec['sha256'], f'Hash mismatch {name}'
            temp.rename(p)
        assert digest(p) == spec['sha256'], f'Hash mismatch {name}'
    return lock

if __name__ == '__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--cache',type=pathlib.Path,required=True);a=ap.parse_args()
    lock=prepare(a.cache,True);print(json.dumps({'status':'passed','files':len(lock['files'])}))
