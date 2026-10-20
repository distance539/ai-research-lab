"""Post-evaluation descriptive template/order breakdown, no selection."""
import json,statistics,argparse
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('directory');a=ap.parse_args();p=Path(a.directory);rows=json.loads((p/'records.json').read_text());out=[]
for group,ti,order in sorted({(r['group'],r['template_index'],r['order']) for r in rows}):
 rr=[r for r in rows if (r['group'],r['template_index'],r['order'])==(group,ti,order)]
 out.append(dict(group=group,template_index=ti,order=order,n=len(rr),clean_top1=sum(r['clean']['top1_id']==r['io_id'] for r in rr),drop=statistics.mean(r['clean']['diff']-r['zero']['diff'] for r in rr),gain=statistics.mean(r['restored']['diff']-r['corrupt']['diff'] for r in rr),tokens=sorted({len(r['ids']) for r in rr})))
print(json.dumps(out,indent=2))
