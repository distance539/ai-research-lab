"""Prespecified descriptive paired cells, no independent-row CI."""
import json,statistics,argparse
from pathlib import Path
M=statistics.mean
CONDITIONS=['clean','zero','corrupt','restored']
def analyze(rows):
 groups={}
 for group in dict.fromkeys(r['group'] for r in rows):
  rr=[r for r in rows if r['group']==group]
  means={k:M(r[k]['diff'] for r in rr) for k in CONDITIONS}
  gap=means['clean']-means['corrupt'];drop=means['clean']-means['zero'];gain=means['restored']-means['corrupt']
  groups[group]={'n':len(rr),'mean_diff':means,'drop':drop,'gain':gain,'gap':gap,'recovery':gain/gap if abs(gap)>1e-6 else None,'positive_counts':{k:sum(r[k]['diff']>0 for r in rr) for k in CONDITIONS},'top1_counts':{k:sum(r[k]['top1_id']==r['io_id'] for r in rr) for k in CONDITIONS},'drop_nonpositive_ids':[r['id'] for r in rr if r['clean']['diff']-r['zero']['diff']<=0],'gain_nonpositive_ids':[r['id'] for r in rr if r['restored']['diff']-r['corrupt']['diff']<=0]}
 contrasts={}
 if len(groups)==4:
  for g in ['new_old','old_new','new_new']:
   contrasts[g]={k:groups[g][k]-groups['old_old'][k] for k in ['drop','gain']}
   contrasts[g]['clean_diff']=groups[g]['mean_diff']['clean']-groups['old_old']['mean_diff']['clean']
  contrasts['factor_interaction']={k:groups['new_new'][k]-groups['new_old'][k]-groups['old_new'][k]+groups['old_old'][k] for k in ['drop','gain']}
 return {'groups':groups,'contrasts':contrasts,'inference':'dependent controlled cells; no population CI; no head reselection'}
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('directory');a=ap.parse_args();p=Path(a.directory);print(json.dumps(analyze(json.loads((p/'records.json').read_text())),indent=2))
