"""Select only on complete development data; refuse to overwrite a frozen selection."""
import argparse
from common import ROOT,load,save,digest,now,frozen_inputs,summarize

def choose(rows,cfg,plan):
    summary=summarize(rows)
    assert list(summary)==cfg['conditions'] and len(rows)==640
    assert all({r['idx'] for r in rows if r['condition']==n}==set(plan['development']) for n in cfg['conditions'])
    winner=max(cfg['conditions'],key=lambda n:summary[n]['correct'])
    return winner,summary
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--development',type=__import__('pathlib').Path,default=ROOT/'development');p.add_argument('--out',type=__import__('pathlib').Path,default=ROOT/'selection.json');a=p.parse_args()
    assert not a.out.exists(),'Refuse overwrite'
    cfg=load(ROOT/'config.json');plan=load(ROOT/'plan.json');rows=load(a.development/'records.json')
    identity=load(a.development/'identity.json');assert identity['frozen_inputs']==frozen_inputs()
    assert load(a.development/'stages.json')['inference']=='passed'
    winner,summary=choose(rows,cfg,plan)
    obj={'frozen_at':now(),'winner':winner,'baseline':cfg['baseline'],'rule':cfg['selection'],'development_sha256':digest(a.development/'records.json'),'development_identity_sha256':digest(a.development/'identity.json'),'all_attempts':summary,'frozen_inputs':frozen_inputs(),'confirmation_ids':plan['reserved'],'confirmation_seen':False,'historical_development_seen':True}
    save(a.out,obj);print(obj['winner'],obj['all_attempts'])
