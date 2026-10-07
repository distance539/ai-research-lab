"""Keep raw generation locally; omit long news echoes from distributable evidence."""
import argparse,json,pathlib,shutil
from prepare import digest
from common import sha,save

def main(a):
    a.private.mkdir(parents=True,exist_ok=True);src=a.out/'predictions.jsonl';raw=a.private/(a.out.name+'-raw-predictions.jsonl')
    assert not raw.exists(),'Preserve prior raw evidence; use a new private directory'
    shutil.copy2(src,raw);rows=[json.loads(x) for x in src.read_text().splitlines()];red=[]
    old=json.loads((a.out/'audit.json').read_text());old['raw_predictions_sha256']=digest(raw);save(a.out/'raw_audit_before_export.json',old)
    for r in rows:
        if r['task']=='news' and len(r['output'])>20:
            red.append({'stage':r['stage'],'idx':r['idx']});r['output_sha256']=sha(r.pop('output'));ids=r.pop('generated_ids');r['generated_ids_sha256']=sha(json.dumps(ids));r['generated_token_count']=len(ids);r['redacted']='potential source-news echo; full output and token IDs kept in private raw archive'
    src.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows));save(a.out/'distribution.json',{'raw_predictions_sha256':digest(raw),'public_predictions_sha256':digest(src),'redacted_rows':red,'redacted_count':len(red),'raw_local_filename':raw.name,'raw_audit':'raw_audit_before_export.json'})
    print(json.dumps({'redacted':len(red),'raw_sha256':digest(raw)}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=pathlib.Path,required=True);p.add_argument('--private',type=pathlib.Path,required=True);main(p.parse_args())
