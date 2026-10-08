"""Mutation tests: the independent audit must reject corrupt evidence."""
import copy
import json
import unittest
from prepare import ROOT
from audit import audit_records

class AuditTests(unittest.TestCase):
    def setUp(self):
        self.records=json.loads((ROOT/'results/records.json').read_text())
        self.summary=json.loads((ROOT/'results/summary.json').read_text())
        self.plan=json.loads((ROOT/'plan.json').read_text())
        self.selected=self.plan['development']

    def check(self):
        return audit_records(self.records,self.summary,self.plan,self.selected)

    def test_valid(self):
        self.check()

    def test_missing_row(self):
        self.records.pop()
        with self.assertRaises(AssertionError):self.check()

    def test_duplicate_identity(self):
        self.records[1]=copy.deepcopy(self.records[0])
        with self.assertRaises(AssertionError):self.check()

    def test_wrong_prediction(self):
        self.records[0]['prediction']=1-self.records[0]['prediction']
        with self.assertRaises(AssertionError):self.check()

    def test_wrong_score(self):
        self.records[0]['candidates'][0]['score']+=0.1
        with self.assertRaises(AssertionError):self.check()

    def test_wrong_demo_order(self):
        row=next(r for r in self.records if r['condition']=='A_forward')
        row['demo_ids']=row['demo_ids'][::-1]
        with self.assertRaises(AssertionError):self.check()

    def test_wrong_denominator(self):
        self.summary['independent_units']=320
        with self.assertRaises(AssertionError):self.check()

    def test_wrong_pair_count(self):
        self.summary['paired_contrasts']['A_reverse minus A_forward']['gains']+=1
        with self.assertRaises(AssertionError):self.check()

if __name__=='__main__':unittest.main()
