"""Predeclared aggregation: raw gains primary; ratio of means secondary."""
import json,argparse,statistics
from pathlib import Path

def analyze(records,patches):
 mean=statistics.mean;byid={r['id']:r for r in records};gap=mean(r['batch_clean']['diff']-r['batch_corrupt']['diff'] for r in records);assert gap>0.1
 ends=[p for p in patches if p['position']==byid[p['id']]['end_position']]
 ranked=[]
 for l in range(12):
  for h in range(12):
   pp=[p for p in ends if p['layer']==l and p['head']==h];g=mean(p['gain'] for p in pp)
   ranked.append({'layer':l,'head':h,'mean_gain':g,'recovery_ratio':g/gap,'mean_diff':mean(p['diff'] for p in pp),'mean_p_io':mean(p['p_io'] for p in pp),'positive_gains':sum(p['gain']>0 for p in pp),'n':len(pp)})
 ranked.sort(key=lambda p:(-p['mean_gain'],p['layer'],p['head']))
 return {'n':len(records),'patch_count':len(patches),'mean_clean_diff':mean(r['batch_clean']['diff'] for r in records),'mean_corrupt_diff':mean(r['batch_corrupt']['diff'] for r in records),'mean_gap':gap,'clean_positive':sum(r['clean']['diff']>0 for r in records),'corrupt_positive_original':sum(r['corrupt']['diff']>0 for r in records),'mean_clean_p_io':mean(r['batch_clean']['p_io'] for r in records),'mean_corrupt_p_io':mean(r['batch_corrupt']['p_io'] for r in records),'end_head_ranking':ranked,'negative_patch_gains':sum(p['gain']<0 for p in patches),'max_absolute_gain_before_changed_token':max(abs(p['gain']) for p in patches if p['position']<byid[p['id']]['s2_position'])}
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--results',default='results');args=a.parse_args();p=Path(args.results);print(json.dumps(analyze(json.loads((p/'records.json').read_text()),json.loads((p/'patches.json').read_text())),indent=2))
