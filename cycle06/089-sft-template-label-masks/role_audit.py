"""Supplementary actual-token audit; no reviews or reversible input IDs emitted."""
import argparse,json,pathlib
from prepare import ROOT,prepare
from run import build,ordered

def main(a):
 import pyarrow.parquet as pq
 from transformers import AutoTokenizer
 cfg=json.loads((ROOT/'config.json').read_text());lock=prepare(a.cache);tok=AutoTokenizer.from_pretrained(a.cache/'models'/lock['model']['revision'],local_files_only=True)
 rows=ordered(pq.read_table(a.cache/'sst2'/lock['data']['revision']/'data/train-00000-of-00001.parquet').to_pylist(),cfg['seed'])[:cfg['train_count']];result=[]
 for r in rows:
  e=build(tok,cfg,r);ids=e['ids'];starts=[j for j,t in enumerate(ids) if t==tok.bos_token_id];ends=[j for j,t in enumerate(ids) if t==tok.eos_token_id];assert len(starts)==len(ends)==3
  roles=[]
  for role,s,end in zip(['system','user','assistant'],starts,ends):
   assert tok.decode(ids[s+1:end]).startswith(role+'\n');roles.append({'role':role,'im_start_position':s,'im_end_position':end})
  good=list(range(e['prefix'],len(ids)));bad=[j for j in good if ids[j]!=tok.pad_token_id]
  assert len(good)-len(bad)==1 and e['eos'] not in bad
  result.append({'idx':r['idx'],'roles':roles,'answer_start':e['prefix'],'response_supervision':len(good),'wrong_id_based_mask_supervision':len(bad),'wrong_id_based_mask_loses_eos':True})
 (a.out).write_text(json.dumps({'passed':True,'records':result},indent=2)+'\n');print('8 real samples: role boundaries and EOS-mask counterexample passed')
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--cache',type=pathlib.Path,required=True);ap.add_argument('--out',type=pathlib.Path,default=ROOT/'role_audit.json');main(ap.parse_args())
