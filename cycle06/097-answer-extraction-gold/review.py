"""Build an unfilled local human packet; validate actual returned annotations."""
import argparse,json,hashlib,html,datetime,re
from pathlib import Path
from prepare import ROOT,digest,prepare

def build(cache,results,out):
    import pyarrow.parquet as pq
    lock=prepare(cache);source=cache/'glue'/lock['glue']['revision']/'sst2/validation-00000-of-00001.parquet'
    data={r['idx']:r for r in pq.read_table(source).to_pylist()};rows={r['idx']:r for r in json.loads((results/'records.json').read_text())}
    sample=json.loads((ROOT/'review_sample.json').read_text());out.mkdir(parents=True,exist_ok=False)
    packet={'topic':97,'sample_sha256':digest(ROOT/'review_sample.json'),'records_sha256':digest(results/'records.json'),'reviewer':'','completed_at':'','attestation':False,'items':[]}
    sections=[]
    for n,s in enumerate(sample['items']):
        row=rows[s['idx']];raw=data[s['idx']];assert s['sentence_sha256']==hashlib.sha256(raw['sentence'].encode()).hexdigest()
        item={'idx':s['idx'],'output_sha256':row['output_sha256'],'decision':None,'content':None,'format':None,'disputed_reference':None,'reason':''};packet['items'].append(item)
        sections.append(f'<section><h2>{n+1} / 16 · ID {s["idx"]}</h2><p>影评</p><blockquote>{html.escape(raw["sentence"])}</blockquote><p>数据集参考：{("negative","positive")[s["gold"]]}</p><p>模型完整输出</p><pre>{html.escape(row["output"])}</pre><p>输出结束原因：{row["stop"]}</p>'+''.join(f'<label>{label}<select data-i="{n}" data-key="{key}"><option value="">请选择</option>'+''.join(f'<option value="{v}">{t}</option>' for v,t in options)+'</select></label>' for key,label,options in [('decision','回答表达的最终立场',[('negative','negative'),('positive','positive'),('ambiguous','矛盾/不确定立场'),('no_answer','没有给出分类')]),('content','内容是否符合参考标签',[('correct','正确'),('incorrect','错误'),('uncertain','无法判定')]),('format','是否遵守仅一个 sentiment 键的 JSON 格式',[('compliant','遵守'),('noncompliant','未遵守')]),('disputed_reference','是否认为数据集标签值得复核',[('no','否'),('yes','是')])])+f'<label>判断理由（必填）<textarea data-i="{n}" data-key="reason"></textarea></label></section>')
    serialized=json.dumps(packet,ensure_ascii=False).replace('<','\\u003c')
    doc='''<!doctype html><meta charset="utf-8"><title>097 真人审核入口</title><style>body{max-width:880px;margin:40px auto;padding:0 24px;font:17px/1.7 system-ui;background:#f5f7fa;color:#172c3d}section{background:white;padding:24px;margin:24px 0;border:1px solid #cbd8e2;border-radius:12px}pre,blockquote{white-space:pre-wrap;background:#eef3f7;padding:18px}label{display:block;margin:14px 0}select,input,textarea{display:block;width:100%;box-sizing:border-box;font:inherit;padding:9px}button{padding:14px 24px;font:inherit;background:#146979;color:white;border:0;border-radius:6px}#status{color:#ad4b00}</style><h1>097 · 16 条真实输出人工审核</h1><p>请阅读影评、参考标签及完整回答，逐条填写。自动评分结果未显示。允许不确定与标签争议；不要猜测模型本来想说什么。输出只有一个情感词也可表达立场，但不一定满足 JSON 格式。代码围栏、多余键、自然语言均不符合本次格式要求；重复键不合法。否定、引用、矛盾表达请依语义判断，并写理由。</p><p>内容判断以给定参考为准；若参考可能错误，请另标争议并解释。此处不要求你替换数据集标签。未明确回答、立场矛盾时选择 uncertain。所有字段为空起步，程序不会代填。表单暂存于本浏览器，导出后请把结果交回此任务。</p><label>审核人（真实称呼或你选择的代号）<input id="reviewer"></label>'''+''.join(sections)+'''<label><input id="attest" type="checkbox" style="width:auto;display:inline">我已亲自阅读并完成上述全部判断，未让模型代填。</label><button id="export">导出审核结果 JSON</button><p id="status"></p><script>const packet='''+serialized+''';const key='ai097-review-'+packet.records_sha256;const inputs=[...document.querySelectorAll('[data-key]')];const saved=localStorage.getItem(key);if(saved){try{const old=JSON.parse(saved);if(old.records_sha256===packet.records_sha256){packet.items=old.items;packet.reviewer=old.reviewer;document.getElementById('reviewer').value=old.reviewer;inputs.forEach(e=>e.value=packet.items[+e.dataset.i][e.dataset.key]??'')}}catch(e){}}function sync(){inputs.forEach(e=>packet.items[+e.dataset.i][e.dataset.key]=e.value||null);packet.reviewer=document.getElementById('reviewer').value.trim();localStorage.setItem(key,JSON.stringify(packet))}inputs.forEach(e=>e.addEventListener('change',sync));document.getElementById('reviewer').addEventListener('input',sync);document.getElementById('export').onclick=()=>{sync();const status=document.getElementById('status');if(!packet.reviewer||!document.getElementById('attest').checked||packet.items.some(r=>['decision','content','format','disputed_reference','reason'].some(k=>!r[k]))){status.textContent='仍有空项、缺少审核人或未勾选真人声明，请补齐。';return}packet.attestation=true;packet.completed_at=new Date().toISOString();const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(packet,null,2)],{type:'application/json'}));a.download='human_review_completed_097.json';a.click();URL.revokeObjectURL(a.href);status.textContent='已导出。请将文件交回本任务；导出不等于已完成实验验收。'};</script>'''
    (out/'review.html').write_text(doc);(out/'human_review_blank.json').write_text(json.dumps(packet,indent=2)+'\n')
    print(json.dumps({'html':str(out/'review.html'),'items':len(packet['items']),'human_status':'pending'}))

def validate(file,results):
    p=json.loads(file.read_text());sample=json.loads((ROOT/'review_sample.json').read_text());rows={r['idx']:r for r in json.loads((results/'records.json').read_text())}
    assert p['sample_sha256']==digest(ROOT/'review_sample.json') and p['records_sha256']==digest(results/'records.json'),'packet identity mismatch'
    assert p['attestation'] is True and p['reviewer'].strip(),'actual human attestation and reviewer required'
    assert datetime.datetime.fromisoformat(p['completed_at'].replace('Z','+00:00')).tzinfo is not None
    assert [x['idx'] for x in p['items']]==[x['idx'] for x in sample['items']]
    counts={k:{'false_positive':0,'false_negative':0,'n':0,'excluded':0,'label_mismatch':0,'label_n':0,'by_class':{str(g):{'n':0,'errors':0} for g in (0,1)}} for k in ['exact','json_correct','regex_correct']}
    for item in p['items']:
        row=rows[item['idx']];assert item['output_sha256']==row['output_sha256']
        assert item['decision'] in ['negative','positive','ambiguous','no_answer'] and item['content'] in ['correct','incorrect','uncertain']
        assert item['format'] in ['compliant','noncompliant'] and item['disputed_reference'] in ['yes','no'] and item['reason'].strip()
        if item['decision'] in ['negative','positive'] and item['disputed_reference']=='no':
            assert item['content']==('correct' if item['decision']==('negative','positive')[row['gold']] else 'incorrect'),'resolve inconsistent human decision/content'
        if item['decision'] in ['ambiguous','no_answer']:assert item['content']=='uncertain'
        for k,c in counts.items():
            if item['content']=='uncertain' or item['disputed_reference']=='yes':c['excluded']+=1;continue
            h=item['content']=='correct';a=row['scores'][k];c['n']+=1;c['false_positive']+=int(a and not h);c['false_negative']+=int(h and not a);cl=c['by_class'][str(row['gold'])];cl['n']+=1;cl['errors']+=int(a!=h)
            label=row['scores']['first_label'] if k=='regex_correct' else None
            if k=='regex_correct':c['label_n']+=1;c['label_mismatch']+=int(label!=item['decision'])
    for c in counts.values():
        c['error_rate']=(c['false_positive']+c['false_negative'])/c['n'] if c['n'] else None
        c['weighted_64_error_rate']=sum(sample['class_sizes'][g]/64*v['errors']/v['n'] for g,v in c['by_class'].items()) if all(v['n']==8 for v in c['by_class'].values()) else None
    return {'status':'completed supplied annotation file; actual human provenance must be confirmed by task owner','review_sha256':digest(file),'reviewer':p['reviewer'],'completed_at':p['completed_at'],'metrics':counts,'format_disagreements':[r['idx'] for r in p['items'] if (r['format']=='compliant')!=rows[r['idx']]['scores']['json_format']]}
if __name__=='__main__':
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='mode',required=True);b=s.add_parser('build');b.add_argument('--cache',type=Path,required=True);b.add_argument('--results',type=Path,required=True);b.add_argument('--out',type=Path,required=True);v=s.add_parser('validate');v.add_argument('--file',type=Path,required=True);v.add_argument('--results',type=Path,required=True);a=p.parse_args()
    if a.mode=='build':build(a.cache,a.results,a.out)
    else:print(json.dumps(validate(a.file,a.results),indent=2))
