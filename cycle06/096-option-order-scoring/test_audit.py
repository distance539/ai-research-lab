"""Mutation checks: demonstrate the audit rejects corrupted scientific records."""
import copy,json,tempfile,unittest
from pathlib import Path
from audit import check
ROOT=Path(__file__).resolve().parent

class AuditMutations(unittest.TestCase):
    def probe(self,mutate):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            for n in ['records.json','identity.json','summary.json']:(p/n).write_bytes((ROOT/'results'/n).read_bytes())
            rows=json.loads((p/'records.json').read_text());mutate(rows);(p/'records.json').write_text(json.dumps(rows))
            with self.assertRaises(AssertionError):check(p,ROOT)
    def test_gold_mapping(self):self.probe(lambda r:r[0].update(position_gold=1-r[0]['position_gold']))
    def test_score(self):self.probe(lambda r:r[0]['candidates'][0].update(score=999))
    def test_missing(self):self.probe(lambda r:r.pop())
    def test_duplicate(self):self.probe(lambda r:r.append(copy.deepcopy(r[0])))
    def test_token_count(self):self.probe(lambda r:r[0]['candidates'][0]['continuation_ids'].append(0))
    def test_prediction(self):self.probe(lambda r:r[0]['predictions']['sum'].update(semantic=1-r[0]['predictions']['sum']['semantic']))

if __name__=='__main__':unittest.main()
