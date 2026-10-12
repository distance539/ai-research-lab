"""Independent deterministic replay, deliberately no imports from run/scoring."""
import json,re,hashlib,argparse
from pathlib import Path

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def check(root):
    rows=json.loads((root/'records.json').read_text());summary=json.loads((root/'summary.json').read_text());identity=json.loads((root/'identity.json').read_text())
    assert [r['idx'] for r in rows]==identity['selected'];assert len({r['idx'] for r in rows})==len(rows)
    counts={k:0 for k in ['exact','json_format','json_correct','regex_correct']};gains=[];losses=[];redacted=[]
    for r in rows:
        gold=('negative','positive')[r['gold']];s=r['scores'];text=r['output']
        if text is None:
            assert r.get('redaction')=='source_overlap_8_words';redacted.append(r['idx'])
        else:
            assert hashlib.sha256(text.encode()).hexdigest()==r['output_sha256']
            hits=[m.group(0).lower() for m in re.finditer(r'positive|negative',text,re.I) if (m.start()==0 or not (text[m.start()-1].isalnum() or text[m.start()-1]=='_')) and (m.end()==len(text) or not (text[m.end()].isalnum() or text[m.end()]=='_'))]
            first=hits[0] if hits else None
            try:
                pairs=json.loads(text,object_pairs_hook=lambda x:x)
                valid=isinstance(pairs,list) and len(pairs)==1 and isinstance(pairs[0],tuple) and pairs[0][0]=='sentiment' and pairs[0][1] in ('negative','positive')
            except (ValueError,TypeError):valid=False
            assert s=={'exact':text.strip()=='{"sentiment": "'+gold+'"}','json_format':valid,'json_correct':bool(valid and pairs[0][1]==gold),'first_label':first,'regex_correct':first==gold,'unique_label':next(iter(set(hits))) if len(set(hits))==1 else None,'hit_count':len(hits),'distinct_labels':sorted(set(hits))}
            assert (r['official_extracted'].lower() if r['official_extracted']!='[invalid]' else None)==first
        assert 0<r['input_length']<=512 and 0<r['generated_length']<=32
        for k in counts:counts[k]+=s[k]
        if s['regex_correct'] and not s['exact']:gains.append(r['idx'])
        if s['exact'] and not s['regex_correct']:losses.append(r['idx'])
    assert summary['n']==len(rows) and summary['counts']==counts and summary['paired']=={'regex_gain':gains,'regex_loss':losses}
    assert summary['tokens']=={'input':sum(r['input_length'] for r in rows),'generated_including_eos':sum(r['generated_length'] for r in rows)}
    assert summary['length_stops']==sum(r['stop']=='length' for r in rows)
    assert summary['human']['error_rates'] is None and summary['human']['n_completed']==0
    return {'status':'passed','n':len(rows),'fully_reparsed':len(rows)-len(redacted),'redacted_ids':redacted,'redacted_boundary':'only derived arithmetic checked for redacted rows; full local audit separately saved','counts':counts,'script_sha256':digest(Path(__file__)),'inputs':{n:digest(root/n) for n in ['records.json','summary.json','identity.json']},'human_review':'not performed by this script'}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--out',type=Path);a=p.parse_args();r=check(a.root);text=json.dumps(r,indent=2);print(text)
    if a.out:a.out.write_text(text+'\n')
