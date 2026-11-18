import sys
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from tigrna_model.features import feature_matrix,feature_names,sequence_from_parts
from tigrna_model.dataio import load_experiment_workbook
from tigrna_model.score import score_records


class SubmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.all_rows=pd.read_csv(ROOT/'data/processed/all_experimental_rows.csv')
        cls.training=pd.read_csv(ROOT/'data/processed/training_dataset.csv')
        cls.bundle=joblib.load(ROOT/'models/tigrna_efficiency_model_293T.joblib')

    def test_sequence_validation(self):
        self.assertEqual(sequence_from_parts('g','gacattggaggtacccgt','a'),'GGACATTGGAGGTACCCGTA')
        for parts in [('GG','A'*18,'C'),('G','A'*17,'C'),('G','N'*18,'C')]:
            with self.assertRaises(ValueError): sequence_from_parts(*parts)

    def test_features(self):
        x,names=feature_matrix(['GGACATTGGAGGTACCCGTA'])
        self.assertEqual(x.shape,(1,482))
        self.assertEqual(len(set(names)),482)
        self.assertTrue(np.isfinite(x).all())
        self.assertEqual(len(feature_names()),482)

    def test_source_parser_and_labels(self):
        raw=load_experiment_workbook(next((ROOT/'data/raw').glob('*.xlsx')))
        saved=self.all_rows.copy()
        # CSV readers represent empty semicolon-string cells as NA; they are not
        # missing numeric labels and must be compared as equivalent empty strings.
        for col in ['wt_replicates','optimized_replicates']:
            raw[col]=raw[col].fillna('')
            saved[col]=saved[col].fillna('')
        pd.testing.assert_frame_equal(raw,saved,check_dtype=False,atol=1e-12,rtol=1e-12)
        self.assertEqual(len(raw),154)
        self.assertEqual(int(raw.primary_label_available.sum()),125)
        self.assertTrue(raw.loc[raw.optimized_n.eq(0),'optimized_efficiency'].isna().all())

    def test_missing_not_zero(self):
        self.assertEqual(len(self.all_rows)-len(self.training),29)
        self.assertTrue(self.training.optimized_n.gt(0).all())
        self.assertTrue(self.training.optimized_efficiency.between(0,1).all())

    def test_293t_and_n2a_scope(self):
        a=self.training[self.training.cell_line.eq('293T')]
        b=self.training[self.training.cell_line.eq('N2a')]
        self.assertEqual((len(a),a.locus_group.nunique()),(98,35))
        self.assertEqual((len(b),b.locus_group.nunique()),(27,1))
        self.assertEqual(set(self.bundle['training_cell_lines']),{'293T'})
        self.assertEqual(len(self.bundle['training_sequences']),98)

    def test_group_folds_do_not_overlap(self):
        groups=self.training.locus_group.to_numpy()
        for train,test in GroupKFold(5).split(self.training,groups=groups):
            self.assertFalse(set(groups[train])&set(groups[test]))
            self.assertFalse(set(self.training.iloc[train].sequence_20)&set(self.training.iloc[test].sequence_20))

    def test_model_parameters(self):
        self.assertEqual(self.bundle['model_family'],'svr_rbf')
        self.assertEqual(self.bundle['model_params'],{'model__C':.1,'model__epsilon':.02,'model__gamma':'scale'})

    def test_all_154_historical_predictions(self):
        expected=pd.read_csv(ROOT/'examples/historical_154_predictions.csv')
        result=score_records(expected[['id','tam5','target','tam3']],self.bundle)
        for col in ['predicted_efficiency_pct','prediction_interval_90_low_pct','prediction_interval_90_high_pct','score_0_100','batch_rank_0_100']:
            np.testing.assert_allclose(result[col].astype(float),expected[col].astype(float),atol=1e-8,rtol=1e-8,equal_nan=True,err_msg=col)

    def test_singleton_rank_and_bounds(self):
        result=score_records(pd.read_csv(ROOT/'examples/candidates.csv').iloc[:1],self.bundle)
        self.assertTrue(pd.isna(result.batch_rank_0_100.iloc[0]))
        self.assertTrue(result.predicted_efficiency_pct.between(0,100).all())
        self.assertTrue(result.prediction_interval_90_low_pct.le(result.predicted_efficiency_pct).all())
        self.assertTrue(result.prediction_interval_90_high_pct.ge(result.predicted_efficiency_pct).all())


if __name__=='__main__': unittest.main()
