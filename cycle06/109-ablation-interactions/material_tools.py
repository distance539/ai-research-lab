"""Exact107 text/visibility helpers; no new material generation."""
import hashlib,json
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
