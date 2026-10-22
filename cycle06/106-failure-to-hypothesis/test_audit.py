"""Reject materially corrupted scientific records (no model rerun)."""
import argparse,json,tempfile,shutil
from pathlib import Path
from check_discovery import check

def main(out):
    root=Path(__file__).resolve().parent
    check(root,out);reports=[]
    mutations=[('category','taxonomy.json',lambda x:x[0].update(category='fabricated')),
               ('delta','taxonomy.json',lambda x:x[0].update(delta=42)),
               ('truncation','taxonomy.json',lambda x:x[0].update(relevant_truncated=99)),
               ('duplicate_query','taxonomy.json',lambda x:x.append(x[0])),
               ('denominator','discovery.json',lambda x:x.update(queries=101)),
               ('input_identity','discovery.json',lambda x:x['inputs'].update({'runs.json':'0'*64})),
               ('registry_identity','discovery.json',lambda x:x.update(registry_sha256='0'*64)),
               ('confirmation_claim','discovery.json',lambda x:x.update(confirmation_queries_evaluated=300))]
    for name,file,mutate in mutations:
        with tempfile.TemporaryDirectory() as tmp:
            dst=Path(tmp)/'results';shutil.copytree(out,dst);p=dst/file;x=json.loads(p.read_text());mutate(x);p.write_text(json.dumps(x))
            try:check(root,dst)
            except (AssertionError,KeyError):reports.append({'case':name,'rejected':True})
            else:raise AssertionError('Corruption accepted: '+name)
    (out/'mutation_tests.json').write_text(json.dumps({'passed':True,'tests':reports},indent=2)+'\n');print(reports)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
