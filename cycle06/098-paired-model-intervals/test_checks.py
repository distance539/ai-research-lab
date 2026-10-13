"""Meaningful corruption tests for pair identity, scores and independent audit."""
import copy,json,pathlib,tempfile,unittest
from run import load_rows
from audit import audit
ROOT=pathlib.Path(__file__).resolve().parent
class Checks(unittest.TestCase):
 def setUp(self):
  self.rows=json.loads((ROOT/'cached_predictions.json').read_text());self.tmp=tempfile.TemporaryDirectory();self.p=pathlib.Path(self.tmp.name)/'rows.json'
 def tearDown(self):self.tmp.cleanup()
 def reject(self,rows):
  self.p.write_text(json.dumps(rows))
  with self.assertRaises(AssertionError):load_rows(self.p)
 def test_duplicate(self):self.reject(self.rows+[self.rows[0]])
 def test_missing_pair(self):self.reject(self.rows[1:])
 def test_prompt_identity(self):self.rows[0]['prompt_sha256']='0'*64;self.reject(self.rows)
 def test_gold_identity(self):self.rows[0]['gold']=1-self.rows[0]['gold'];self.reject(self.rows)
 def test_score(self):self.rows[0]['joint']=not self.rows[0]['joint'];self.reject(self.rows)
 def test_shuffle_safe(self):
  self.p.write_text(json.dumps(list(reversed(self.rows))));self.assertEqual(load_rows(self.p),load_rows(ROOT/'cached_predictions.json'))
 def test_independent(self):self.assertEqual(audit(ROOT/'cached_predictions.json',ROOT/'results/summary.json')['status'],'passed')
 def test_wrong_summary(self):
  s=json.loads((ROOT/'results/summary.json').read_text());s['A_correct']-=1;self.p.write_text(json.dumps(s))
  with self.assertRaises(AssertionError):audit(ROOT/'cached_predictions.json',self.p)
if __name__=='__main__':unittest.main()
