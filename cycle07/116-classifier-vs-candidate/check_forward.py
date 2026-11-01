"""Independent next-token check: one context forward, full-vocab softmax."""
import argparse,json,hashlib
from pathlib import Path
import torch,pyarrow.parquet as pq
from transformers import AutoTokenizer,AutoModelForCausalLM
p=Path(__file__).resolve().parent;a=argparse.ArgumentParser();a.add_argument('--model-dir',required=True);a.add_argument('--data',required=True);a=a.parse_args();torch.set_num_threads(4)
t=AutoTokenizer.from_pretrained(a.model_dir,local_files_only=True);m=AutoModelForCausalLM.from_pretrained(a.model_dir,local_files_only=True,attn_implementation='eager').eval();rows={x['idx']:x for x in pq.read_table(a.data).to_pylist()};pred=json.loads((p/'formal_candidate_predictions.json').read_text());err=0
for x in pred[:8]:
 text='Review: '+rows[x['idx']]['sentence']+'\nQuestion: Is this review positive or negative?\nAnswer:';ids=t.encode(text,add_special_tokens=False)
 with torch.no_grad():v=m(torch.tensor([ids])).logits[0,-1].double().softmax(-1)
 assert [t.decode(z) for z in x['candidate_ids']]==[' negative',' positive']
 for k in [0,1]:err=max(err,abs(float(v[x['candidate_ids'][k][0]].log())-x['scores'][k]))
assert err<1e-10
out={'passed':True,'n':8,'max_logprob_error':err,'label_mapping':[' negative',' positive'],'method':'independent prompt-only forward, full-vocabulary softmax, then log; compared with teacher-forced candidate scores'};(p/'forward_audit.json').write_text(json.dumps(out,indent=2));print(out)
