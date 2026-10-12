"""Freeze human sample BEFORE generation, using only labels, IDs and text hashes."""
import argparse,json,hashlib,datetime
from pathlib import Path
from prepare import prepare,digest,ROOT

def select(cache):
    import pyarrow.parquet as pq
    lock=prepare(cache);plan=json.loads((ROOT/'plan.json').read_text())
    source=cache/'glue'/lock['glue']['revision']/'sst2/validation-00000-of-00001.parquet'
    rows=pq.read_table(source).to_pylist();lookup={r['idx']:r for r in rows}
    ordered=sorted(rows,key=lambda r:hashlib.sha256(('88:'+str(r['idx'])).encode()).hexdigest())
    assert [r['idx'] for r in ordered[:64]]==plan['development']
    assert [r['idx'] for r in ordered[64:192]]==plan['reserved']
    groups={g:sorted([lookup[i] for i in plan['development'] if lookup[i]['label']==g],key=lambda r:hashlib.sha256(('97:human:'+str(r['idx'])).encode()).hexdigest()) for g in (0,1)}
    chosen=[groups[g][j] for j in range(8) for g in (0,1)]
    return {'created_at':datetime.datetime.now().astimezone().isoformat(),'rule':"8 per gold class; SHA256('97:human:'+idx), ascending; class interleaved",'script_sha256':digest(Path(__file__)),'data_sha256':digest(source),'plan_sha256':digest(ROOT/'plan.json'),'protocol_sha256':digest(ROOT/'PROTOCOL.md'),'class_sizes':{str(g):len(v) for g,v in groups.items()},'items':[{'idx':r['idx'],'gold':r['label'],'sentence_sha256':hashlib.sha256(r['sentence'].encode()).hexdigest()} for r in chosen]}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();result=select(a.cache)
    with a.out.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))
