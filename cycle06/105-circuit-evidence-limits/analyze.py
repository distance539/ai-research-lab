"""Descriptive grouped comparisons, not iid confidence intervals."""
import json,argparse,statistics
from pathlib import Path
SUBSETS=['','A','B','C','AB','AC','BC','ABC']
def analyze(rows):
 out={}
 for group in dict.fromkeys(r['group'] for r in rows):
  rr=[r for r in rows if r['group']==group];g={'n':len(rr),'clean_mean':statistics.mean(r['clean']['diff'] for r in rr),'clean_top1':sum(r['clean']['top1_id']==r['io_id'] for r in rr),'modes':{}}
  for mode in ['zero','mean_abc']:
   subsets={}
   for k in SUBSETS:
    f=[r['conditions'][mode]['full'][k]['diff'] for r in rr];c=[r['conditions'][mode]['retained'][k]['diff'] for r in rr]
    subsets[k]={'full_mean':statistics.mean(f),'retained_mean':statistics.mean(c),'signed_mean_gap':statistics.mean(x-y for x,y in zip(f,c)),'absolute_mean_gap':abs(statistics.mean(x-y for x,y in zip(f,c))),'row_mae':statistics.mean(abs(x-y) for x,y in zip(f,c)),'retained_top1':sum(r['conditions'][mode]['retained'][k]['top1_id']==r['io_id'] for r in rr),'full_top1':sum(r['conditions'][mode]['full'][k]['top1_id']==r['io_id'] for r in rr),'retained_positive':sum(x>0 for x in c),'top1_disagreements':sum(r['conditions'][mode]['full'][k]['top1_id']!=r['conditions'][mode]['retained'][k]['top1_id'] for r in rr)}
   worst=max(rr,key=lambda r:(abs(r['clean']['diff']-r['conditions'][mode]['retained']['']['diff']),-r['id']))
   maxk=max(SUBSETS,key=lambda k:subsets[k]['absolute_mean_gap'])
   g['modes'][mode]={'subsets':subsets,'max_gap_subset':maxk,'max_gap':subsets[maxk]['absolute_mean_gap'],'worst_row_id':worst['id'],'worst_row_residual':worst['clean']['diff']-worst['conditions'][mode]['retained']['']['diff']}
  out[group]=g
 return out
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--results',default='results');ap.add_argument('--out',required=True);a=ap.parse_args();Path(a.out).write_text(json.dumps(analyze(json.loads((Path(a.results)/'records.json').read_text())),indent=2)+'\n')
