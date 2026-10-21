"""SHA-checked official literals -> balanced four-cell transfer design."""
import argparse,ast,json,hashlib,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parent
URL='https://raw.githubusercontent.com/redwoodresearch/Easy-Transformer/ea15315dd24481e9e2ac5c3ef335d82907a1dc34/easy_transformer/ioi_dataset.py'
SHA='94a72de90e426382322fd690669024a0d2d4bd72a45ed9ce820a99ae9173d231'
def build(b,cfg):
 assert hashlib.sha256(b).hexdigest()==SHA
 vals={n.targets[0].id:ast.literal_eval(n.value) for n in ast.parse(b).body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ['NAMES','BABA_TEMPLATES']}
 assert vals['NAMES'][:8]==cfg['names'] and vals['NAMES'][8:16]==cfg['heldout_names']
 rows=[]
 for group in cfg['groups']:
  ng,tg=group.split('_');names=cfg['names'] if ng=='old' else cfg['heldout_names']
  for slot,ti in enumerate(cfg['template_groups'][tg]):
   for order in ['BABA','ABBA']:
    template=vals['BABA_TEMPLATES'][ti]
    if order=='ABBA':template=template.replace('[B] and [A]','[A] and [B]',1)
    for pi in range(8):
     pair=[names[pi],names[(pi+1)%8]]
     for swap in [False,True]:
      a,b=pair[::-1] if swap else pair
      full=template.replace('[A]',a).replace('[B]',b).replace('[PLACE]',cfg['place']).replace('[OBJECT]',cfg['object'])
      assert full.endswith(' '+a)
      rows.append(dict(id=len(rows),group=group,name_group=ng,template_group=tg,template_index=ti,template_slot=slot,order=order,pair_id=pi,swapped=swap,IO=a,S=b,prompt=full[:-len(' '+a)],answer=' '+a,distractor=' '+b,paired_key=f'{slot}-{order}-{pi}-{int(swap)}'))
 assert len(rows)==256 and len({r['prompt'] for r in rows})==256
 return rows
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--source');a=ap.parse_args()
 b=Path(a.source).read_bytes() if a.source else urllib.request.urlopen(URL,timeout=60).read()
 rows=build(b,json.loads((ROOT/'config.json').read_text()));s=json.dumps(rows,indent=2)+'\n'
 if (ROOT/'samples.json').exists():assert s==(ROOT/'samples.json').read_text()
 Path(a.out).write_text(s);print('256 samples reconstructed; no model inference')
if __name__=='__main__':main()
