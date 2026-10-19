"""Prespecified descriptive summaries. No inference over a population of sentences."""
import json,argparse,itertools
from pathlib import Path
from statistics import mean

def analyze(rows):
 modes=['zero','mean_abc','resample_abc']; subsets=['A','B','C','AB','AC','BC','ABC']
 out={'n':len(rows),'mean_clean_diff':mean(r['clean']['diff'] for r in rows),'conditions':{},'interactions':{}}
 for mode in modes:
  out['conditions'][mode]={}
  for s in subsets:
   vals=[r['interventions'][mode][s] for r in rows]
   out['conditions'][mode][s]={'mean_diff':mean(v['diff'] for v in vals),'mean_drop':mean(r['clean']['diff']-v['diff'] for r,v in zip(rows,vals)),'positive_diff':sum(v['diff']>0 for v in vals),'vocab_correct':sum(v['top1_id']==r['io_id'] for r,v in zip(rows,vals)),'mean_p_io':mean(v['p_io'] for v in vals)}
  effects=[]
  for r in rows:
   d={s:r['clean']['diff']-r['interventions'][mode][s]['diff'] for s in subsets}
   effects.append({'id':r['id'],'I_AB':d['AB']-d['A']-d['B'],'I_AC':d['AC']-d['A']-d['C'],'I_BC':d['BC']-d['B']-d['C'],'conditional_C_after_AB':d['ABC']-d['AB'],'conditional_C_minus_single':d['ABC']-d['AB']-d['C'],'triple_minus_singles':d['ABC']-d['A']-d['B']-d['C']})
  out['interactions'][mode]={'rows':effects,'means':{k:mean(r[k] for r in effects) for k in effects[0] if k!='id'}}
 return out
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('out');a=p.parse_args();d=Path(a.out);(d/'summary.json').write_text(json.dumps(analyze(json.loads((d/'records.json').read_text())),indent=2)+'\n')
