"""Distributable evidence; omit copied source text, preserve identities and facts."""
import argparse,json,shutil,re
from pathlib import Path
from prepare import prepare,digest

def export(cache,source,out):
    import pyarrow.parquet as pq
    lock=prepare(cache);rows=json.loads((source/'records.json').read_text())
    data={r['idx']:r['sentence'] for r in pq.read_table(cache/'glue'/lock['glue']['revision']/'sst2/validation-00000-of-00001.parquet').to_pylist()}
    out.mkdir(parents=True,exist_ok=False);redacted=[]
    for r in rows:
        a=re.findall(r'\w+',data[r['idx']].lower());b=re.findall(r'\w+',r['output'].lower());grams={tuple(a[i:i+8]) for i in range(len(a)-7)}
        if any(tuple(b[i:i+8]) in grams for i in range(len(b)-7)):
            r['output']=None;r['redaction']='source_overlap_8_words';redacted.append(r['idx'])
    (out/'records.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
    for n in ['summary.json','identity.json','environment.json','stages.json']:shutil.copy2(source/n,out/n)
    (out/'distribution.json').write_text(json.dumps({'raw_records_sha256':digest(source/'records.json'),'redacted_ids':redacted,'reason':'SST2 license unknown; no source reviews/full prompts or 8-word copied spans redistributed','raw_audit':'full local raw audit required; public audit reports its reduced parsing scope'},indent=2)+'\n')
    print(json.dumps({'files':6,'redacted_ids':redacted}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();export(a.cache,a.source,a.out)
