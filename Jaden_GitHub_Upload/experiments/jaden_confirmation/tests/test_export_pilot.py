import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import pandas as pd
from export_pilot_predictions import convert


class ExportTest(unittest.TestCase):
    def source(self):
        return pd.DataFrame([
            {'ckpt':'p6144','seed':0,'model':m,'feature':f,'layer':l,'trace_id':t,
             'problem_id':'a','y':y,'p':p,'cfg_sha':'same'}
            for m,f,l in [('hgb','topk50',20),('lr','len',-999)]
            for t,y,p in [('a0',0,.2),('a1',1,.8)]])
    def call(self, d):
        return convert(d,'p6144',0,'hgb','topk50',20,'lr','len',-999)
    def test_correct_orientation(self):
        out,_=self.call(self.source())
        self.assertEqual(out.y_correct.tolist(),[1,0])
        self.assertAlmostEqual(out.probe_score.iloc[0],.8)
    def test_missing_counterpart_fails(self):
        with self.assertRaisesRegex(ValueError,'identical trace'):
            self.call(self.source().iloc[:-1])
    def test_config_mismatch_fails(self):
        d=self.source();d.loc[0,'cfg_sha']='different'
        with self.assertRaisesRegex(ValueError,'configuration'):
            self.call(d)

if __name__=='__main__':unittest.main()
