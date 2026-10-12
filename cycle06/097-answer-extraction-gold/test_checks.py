"""Software-only fixtures/mutations, never human annotations or benchmark rows."""
import copy,json,tempfile,shutil
from pathlib import Path
from scoring import score
from audit import check
from review import validate
ROOT=Path(__file__).resolve().parent

def main():
    fixtures=[('{"sentiment": "positive"}',1,True,True,'positive'),('{"sentiment":"positive"}',1,False,True,'positive'),('{"sentiment":"positive","sentiment":"negative"}',1,False,False,'positive'),('not positive; negative',0,False,False,'positive'),('positively',1,False,False,None),('NEGATIVE',0,False,False,'negative')]
    for text,gold,exact,fmt,label in fixtures:
        s=score(text,gold);assert (s['exact'],s['json_format'],s['first_label'])==(exact,fmt,label)
    mutations=['score','hash','missing','duplicate','summary','official']
    for kind in mutations:
        with tempfile.TemporaryDirectory() as t:
            d=Path(t)
            for n in ['records.json','summary.json','identity.json']:shutil.copy2(ROOT/'results'/n,d/n)
            rows=json.loads((d/'records.json').read_text());summary=json.loads((d/'summary.json').read_text())
            if kind=='score':rows[0]['scores']['regex_correct']=not rows[0]['scores']['regex_correct']
            if kind=='hash':rows[0]['output_sha256']='0'*64
            if kind=='missing':rows.pop()
            if kind=='duplicate':rows.append(copy.deepcopy(rows[0]))
            if kind=='summary':summary['counts']['exact']+=1
            if kind=='official':rows[0]['official_extracted']='corrupt'
            (d/'records.json').write_text(json.dumps(rows));(d/'summary.json').write_text(json.dumps(summary))
            try:check(d)
            except AssertionError:pass
            else:raise AssertionError('Mutation escaped: '+kind)
    with tempfile.TemporaryDirectory() as t:
        p=Path(t)/'blank.json';p.write_text(json.dumps({'sample_sha256':__import__('audit').digest(ROOT/'review_sample.json'),'records_sha256':__import__('audit').digest(ROOT/'results/records.json'),'reviewer':'','attestation':False}))
        try:validate(p,ROOT/'results')
        except AssertionError as e:assert 'human' in str(e)
        else:raise AssertionError('Blank human review accepted')
    print(json.dumps({'passed':13,'parser_fixtures':6,'mutation_rejections':6,'empty_human_rejection':1,'human_annotations_created':0}))
if __name__=='__main__':main()
