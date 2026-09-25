"""Download immutable-by-hash public resources; never silently accept changed latest data."""
import argparse
from pathlib import Path
import tarfile
import urllib.request
import zipfile
from common import ROOT, read_json, sha, verify_resources

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', type=Path, required=True)
    ap.add_argument('--with-es-macos-arm64', action='store_true')
    args = ap.parse_args()
    lock = read_json(ROOT / 'resources.lock.json')
    dl = args.cache / 'downloads'
    dl.mkdir(parents=True, exist_ok=True)
    for name, spec in lock['resources'].items():
        if name.startswith('elasticsearch') and not args.with_es_macos_arm64:
            continue
        path = dl / spec['filename']
        if not path.exists():
            print('download', name, spec['url'], flush=True)
            tmp = path.with_suffix(path.suffix + '.part')
            with urllib.request.urlopen(spec['url'], timeout=120) as src, tmp.open('wb') as dst:
                while chunk := src.read(1 << 20):
                    dst.write(chunk)
            assert sha(tmp) == spec['sha256'], f'changed upstream bytes: {name}'
            tmp.replace(path)
        assert sha(path) == spec['sha256'], f'cache hash mismatch: {name}'
        if name == 'scifact':
            with zipfile.ZipFile(path) as z:
                for member in z.namelist():
                    assert not Path(member).is_absolute() and '..' not in Path(member).parts
                z.extractall(args.cache)
        else:
            dest = args.cache / 'upstream' if name == 'beir' else dl if name == 'original' else args.cache
            dest.mkdir(parents=True, exist_ok=True)
            with tarfile.open(path) as t:
                t.extractall(dest, filter='data')
        print('verified', name, spec['sha256'])
    verify_resources(args.cache)
    print('All source/data hashes verified.')

if __name__ == '__main__':
    main()
