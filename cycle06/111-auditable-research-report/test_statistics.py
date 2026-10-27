import argparse,json,gzip,shutil,tempfile
from pathlib import Path
from audit_statistics import check

def main(out):
 mutations=[]
 for kind in ['interval','group','unpaired','draw_index','seed']:
  with tempfile.TemporaryDirectory() as tmp:
   d=Path(tmp)/'data';shutil.copytree(out,d)
   if kind in ['interval','seed']:
    p=d/'uncertainty.json';x=json.loads(p.read_text())
    if kind=='seed':x['seed']=999
    else:x['comparisons']['blend_minus_ce']['component_percentile_95'][0]+=.1
    p.write_text(json.dumps(x))
   elif kind=='group':
    p=d/'components.json';x=json.loads(p.read_text());x['groups'][0].append('invented');p.write_text(json.dumps(x))
   elif kind=='unpaired':
    p=d/'bootstrap_replicates.json.gz';x=json.loads(gzip.decompress(p.read_bytes()));x['component']['blend_minus_ce'][0]+=.1;p.write_bytes(gzip.compress(json.dumps(x).encode()))
   else:
    p=d/'bootstrap_indices.json.gz';x=json.loads(gzip.decompress(p.read_bytes()));x['component'][0][0]=-1;p.write_bytes(gzip.compress(json.dumps(x).encode()))
   try:check(d,False)
   except (AssertionError,KeyError,ValueError):mutations.append({'mutation':kind,'rejected':True})
   else:raise AssertionError(kind+' escaped')
 # Unequal-sized clusters: pooled query estimator must not become unweighted mean of cluster means.
 groups=[[1.0,1.0],[0.0]];assert sum(map(sum,groups))/sum(map(len,groups))==2/3;assert sum(sum(g)/len(g) for g in groups)/2==.5
 result={'passed':True,'mutations':mutations,'unequal_size_estimand_control':True};(out/'statistics_mutation_tests.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
