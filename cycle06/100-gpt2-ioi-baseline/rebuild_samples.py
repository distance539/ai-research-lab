"""Fetch fixed public template source, parse literals, reproduce frozen sample bytes."""
import argparse,ast,json,hashlib,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parent
URL='https://raw.githubusercontent.com/redwoodresearch/Easy-Transformer/ea15315dd24481e9e2ac5c3ef335d82907a1dc34/easy_transformer/ioi_dataset.py'
SHA='94a72de90e426382322fd690669024a0d2d4bd72a45ed9ce820a99ae9173d231'
def main():
 a=argparse.ArgumentParser();a.add_argument('--out',required=True);a.add_argument('--source',help='optional already downloaded fixed official file');args=a.parse_args()
 b=Path(args.source).read_bytes() if args.source else urllib.request.urlopen(URL,timeout=60).read();assert hashlib.sha256(b).hexdigest()==SHA
 vals={}
 for n in ast.parse(b).body:
  if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ['NAMES','BABA_TEMPLATES']:vals[n.targets[0].id]=ast.literal_eval(n.value)
 cfg=json.loads((ROOT/'config.json').read_text());names=vals['NAMES'][:8];assert names==cfg['names'];assert vals['NAMES'][8:16]==cfg['heldout_names']
 pairs=[(names[i],names[(i+1)%8]) for i in range(8)];rows=[]
 for ti in cfg['template_indices']:
  for order in ['BABA','ABBA']:
   template=vals['BABA_TEMPLATES'][ti]
   if order=='ABBA':template=template.replace('[B] and [A]','[A] and [B]',1)
   for pi,pair in enumerate(pairs):
    for swap in [False,True]:
     a,b=pair[::-1] if swap else pair;full=template.replace('[A]',a).replace('[B]',b).replace('[PLACE]',cfg['place']).replace('[OBJECT]',cfg['object'])
     rows.append({'id':len(rows),'template_index':ti,'order':order,'pair_id':pi,'swapped':swap,'IO':a,'S':b,'prompt':full[:-len(' '+a)],'answer':' '+a,'distractor':' '+b})
 output=json.dumps(rows,indent=2)+'\n';assert output==(ROOT/'samples.json').read_text();Path(args.out).write_text(output);print('64 samples exactly reconstructed from fixed official literals')
if __name__=='__main__':main()
