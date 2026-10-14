import datetime,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def save(p,v):
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,default=str)+'\n')
def load(p):return json.loads(p.read_text())
def sha(s):return hashlib.sha256(s.encode()).hexdigest()
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def frozen_inputs():
    return {n:digest(ROOT/n) for n in ['run.py','spec.py','prepare.py','common.py','select.py','analyze.py','audit.py','config.json','plan.json','resources.lock.json','official_sst2.yaml','PROTOCOL.md']}
def summarize(rows):
    names=list(dict.fromkeys(r['condition'] for r in rows));ids={r['idx'] for r in rows if r['condition']==names[0]}
    out={}
    for name in names:
        rs=[r for r in rows if r['condition']==name];assert len(rs)==len(ids) and {r['idx'] for r in rs}==ids
        out[name]={'n':len(rs),'correct':sum(r['correct'] for r in rs),'accuracy':sum(r['correct'] for r in rs)/len(rs),'candidate_input_tokens':sum(c['input_tokens'] for r in rs for c in r['candidates']),'context_tokens_range':[min(r['context_tokens'] for r in rs),max(r['context_tokens'] for r in rs)]}
    return out
