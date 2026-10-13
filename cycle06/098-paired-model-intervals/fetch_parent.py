"""Verified fixed-commit acquisition of the complete self-written 093 experiment."""
import argparse, hashlib, json, pathlib, urllib.request
ROOT=pathlib.Path(__file__).resolve().parent
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=pathlib.Path,required=True);a=p.parse_args();lock=json.loads((ROOT/'parent_source.json').read_text());a.out.mkdir(parents=True,exist_ok=False)
    for name,expected in lock['files'].items():
        url=f"https://raw.githubusercontent.com/distance539/ai-research-lab/{lock['commit']}/{lock['path']}/{name}"
        with urllib.request.urlopen(url,timeout=60) as r:data=r.read()
        assert hashlib.sha256(data).hexdigest()==expected,name
        dest=a.out/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
    print(json.dumps({'status':'passed','commit':lock['commit'],'files':len(lock['files'])}))
