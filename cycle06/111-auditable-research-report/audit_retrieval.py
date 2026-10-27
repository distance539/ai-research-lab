"""Independent scalar RRF / input association audit; stdlib, no official imports."""
import json,math,hashlib,argparse
from pathlib import Path

def main(out):
 load=lambda f:json.loads((out/f).read_text());bm=load('bm25.json');dense=load('dense.json');rr=load('rrf_union.json');cand=load('candidates.json')
 rank=lambda s:sorted(s,key=lambda d:(s[d],d),reverse=True)
 assert set(bm)==set(dense)==set(rr)==set(cand)
 maxerr=0
 for q in bm:
  expected={}
  for run in [bm,dense]:
   assert 0<len(run[q])<=100 and all(math.isfinite(x) for x in run[q].values())
   for i,d in enumerate(rank(run[q]),1):expected[d]=expected.get(d,0.)+1/(60+i)
  assert set(expected)==set(rr[q])
  for d,x in expected.items():maxerr=max(maxerr,abs(rr[q][d]-x))
  assert len(cand[q])==50 and list(cand[q])==rank(rr[q])[:50]
  assert all(cand[q][d]==rr[q][d] for d in cand[q])
 assert maxerr<1e-14
 result={'passed':True,'queries':len(bm),'candidates':sum(map(len,cand.values())),'max_score_error':maxerr,'inputs':{f:hashlib.sha256((out/f).read_bytes()).hexdigest() for f in ['bm25.json','dense.json','rrf_union.json','candidates.json']},'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()};(out/'retrieval_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
