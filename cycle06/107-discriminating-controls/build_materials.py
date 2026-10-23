"""Select four pre-score diagnostic probes and prepare an unlabelled review packet."""
import argparse,hashlib,html,json,datetime
from pathlib import Path
from common import ROOT,read,write,sha,verify

def text_of(d,indices):return (d['title']+' '+' '.join(d['abstract'][i] for i in indices)).strip()
def variants(d,e):
 rest=[i for i in range(len(d['abstract'])) if i not in e]
 return {'original':list(range(len(d['abstract']))),'identity':list(range(len(d['abstract']))),'front':e+rest,'tail':rest+e}
def visibility(tok,q,d,indices,e):
 text=text_of(d,indices); full=tok(q,text,truncation=False,verbose=False);enc=tok(q,text,truncation='longest_first',max_length=512,return_offsets_mapping=True)
 spans={};pos=len(d['title'])+1
 for i in indices:spans[i]=(pos,pos+len(d['abstract'][i]));pos+=len(d['abstract'][i])+1
 doc_offsets=[off for off,sid in zip(enc['offset_mapping'],enc.sequence_ids()) if sid==1]
 retained_end=max([b for a,b in doc_offsets],default=0)
 return {'raw_tokens':len(full['input_ids']),'retained_tokens':len(enc['input_ids']),'evidence_fully_visible':all(spans[i][1]<=retained_end for i in e),'evidence_visible_sentences':sum(spans[i][1]<=retained_end for i in e),'evidence_sentences':len(e),'document_retained_char_end':retained_end,'input_ids_sha256':hashlib.sha256(json.dumps(enc['input_ids']).encode()).hexdigest()}
def main(cache):
 from transformers import AutoTokenizer
 mp=verify(cache);r=read(ROOT/'resources.lock.json')
 for f,h in r['original_files'].items():assert sha(cache/f)==h
 corpus={str(x['doc_id']):x for x in map(json.loads,(cache/'downloads/data/corpus.jsonl').read_text().splitlines())}
 beir={x['_id']:x for x in map(json.loads,(cache/'scifact/corpus.jsonl').read_text().splitlines())}
 claims={str(x['id']):x for x in map(json.loads,(cache/'downloads/data/claims_train.jsonl').read_text().splitlines())}
 cand=read(ROOT/'inputs/candidates.json');gold=read(ROOT/'inputs/qrels.json');tok=AutoTokenizer.from_pretrained(mp,local_files_only=True,use_fast=True)
 pools={'long':[],'short':[]}
 for q in cand:
  rel=[d for d,v in gold[q].items() if v>0]
  if len(rel)!=1 or rel[0] not in cand[q]:continue
  d=rel[0];claim=claims[q];doc=dict(corpus[d]);doc['abstract']=[s.strip() for s in doc['abstract']];e=sorted({i for x in claim['evidence'].get(d,[]) for i in x['sentences']})
  if not e or len(e)==len(doc['abstract']):continue
  assert ' '.join(doc['abstract'])==beir[d]['text'] and doc['title']==beir[d]['title']
  vv=variants(doc,e);vis={k:visibility(tok,claim['claim'],doc,v,e) for k,v in vv.items()}
  if len({x['raw_tokens'] for x in vis.values()})!=1:continue
  if vis['front']['evidence_fully_visible'] and not vis['tail']['evidence_fully_visible']:group='long'
  elif vis['original']['raw_tokens']<=512 and all(x['evidence_fully_visible'] for x in vis.values()):group='short'
  else:continue
  pools[group].append({'query_id':q,'doc_id':d,'group':group,'query':claim['claim'],'document':doc,'evidence_indices':e,'orders':vv,'visibility':vis,'original_evidence':claim['evidence'][d]})
 selected=[]
 for group in ['long','short']:
  rows=sorted(pools[group],key=lambda x:hashlib.sha256(('107-probe:'+x['query_id']).encode()).hexdigest());assert len(rows)>=2
  selected+=rows[:2]
 packet={'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'selection':'exactly one relevant document in fixed50; equal raw token count; front evidence fully retained and tail not for long; <=512 and all retained for short; two lowest SHA256(107-probe:+query_id) per stratum; no107 scores read','eligible':{k:[x['query_id'] for x in v] for k,v in pools.items()},'semantic_review_status':'pending','rows':selected}
 write(ROOT/'materials.json',packet)
 write(ROOT/'config.json',{'seed':107,'conditions':['original','identity','front','tail'],'queries':[x['query_id'] for x in selected],'top_k':50,'max_length':512,'batch_size':16,'threads':4,'device':'cpu','dtype':'float32','prompt':'none','decode':'scalar logit; Identity activation','tie_break':'score DESC, doc ID string DESC','primary':'per-query nDCG10 front-minus-tail, then mean(long)-mean(short); diagnostic, no population inference','confirmation':'300 BEIR test queries not evaluated','human_review':'pending; all transformed relevance metrics provisional','budget_seconds':1200,'peak_rss_limit_bytes':4294967296})
 sections=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>107 文本变换人工审核</title><style>body{max-width:1100px;margin:40px auto;font:17px/1.65 system-ui}section{border-top:2px solid #bbb;margin-top:35px}mark{background:#fff2a8}pre{white-space:pre-wrap}table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:10px}</style><h1>107：文本变换与标签保持审核</h1><p>状态：待真实人员审核。共4个查询、8个变换。没有展示模型分数。黄色是原数据集证据句，数字是原句号索引（从0开始）。请比较原文、前移、后移：指代、否定、时间、对照组是否仍清楚？完整变换文档与原查询的相关性是否保持？这不是判断截断后输入仍有足够证据。</p><p>请逐项回复 query_id/front 或 tail：通过／不通过／不确定，附简短原因。可在本聊天一次提交8项结果，或编辑 HUMAN_REVIEW_RESPONSE.json；不通过者保留并登记，不按模型胜负剔除。真实回复前不得称语义审核通过。</p>']
 response={'reviewer':None,'reviewed_at':None,'materials_sha256':sha(ROOT/'materials.json'),'decisions':[]}
 for x in selected:
  sections.append('<section><h2>查询 '+x['query_id']+' / 文档 '+x['doc_id']+'</h2><p>'+html.escape(x['query'])+'</p><h3>'+html.escape(x['document']['title'])+'</h3>')
  for k in ['original','front','tail']:
   sections.append('<h3>'+k+'</h3>')
   for i in x['orders'][k]:
    s='['+str(i)+'] '+html.escape(x['document']['abstract'][i]);sections.append('<p>'+('<mark>'+s+'</mark>' if i in x['evidence_indices'] else s)+'</p>')
   if k!='original':response['decisions'].append({'query_id':x['query_id'],'condition':k,'label_preserved':None,'transformation_valid':None,'reason':None})
  sections.append('</section>')
 sections.append('</html>');(ROOT.parent/'HUMAN_REVIEW.html').write_text('\n'.join(sections));write(ROOT.parent/'HUMAN_REVIEW_RESPONSE.json',response)
 write(ROOT/'input.lock.json',{'files':{str(f.relative_to(ROOT)):sha(f) for f in [*sorted((ROOT/'inputs').glob('*.json')),ROOT/'materials.json']}})
 print(json.dumps({'eligible':{k:len(v) for k,v in pools.items()},'selected':[{k:x[k] for k in ['query_id','doc_id','group','evidence_indices','visibility']} for x in selected]},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);main(p.parse_args().cache)
