import pandas as pd
import unittest

def findAllRecIDs(df):
        """For each recommendation system, puts all of the recommendation IDs into a set"""
        rec_set = set()
        for ind in range(5,15):
            rec_set.update(df.iloc[:,ind])
        return rec_set

class CheckIDs(unittest.TestCase):
    """Unit test for testing the equality of all three sets of recommendation IDs for bm25, kwd, and spctr TSVs"""
    @classmethod
    def setUpClass(cls):
        # Read in TSVs
        bm25_df = pd.read_csv('dhq-recs-zfill-bm25.tsv', sep='\t')
        kwd_df = pd.read_csv('dhq-recs-zfill-kwd.tsv', sep='\t')
        spctr_df = pd.read_csv('dhq-recs-zfill-spctr.tsv', sep='\t')
        # Grab the Article IDs into a set
        cls.bm25_ids = set(bm25_df['Article ID'].tolist())
        cls.kwd_ids = set(kwd_df['Article ID'].tolist())
        cls.spctr_ids = set(spctr_df['Article ID'].tolist())
        # Calculate the sets for all of the recommendation IDs
        cls.bm25_rec_ids = findAllRecIDs(bm25_df)
        cls.kwd_rec_ids = findAllRecIDs(kwd_df)
        cls.spctr_rec_ids = findAllRecIDs(spctr_df)

    # Each test is a set difference, so if set_A_diff_set_B results in an ID, then that ID is in A and not B. 
    def test_bm25_diff_spctr(self):
        self.assertEqual(self.bm25_ids - self.spctr_ids, set())

    def test_bm25_diff_kwd(self):
        self.assertEqual(self.bm25_ids - self.kwd_ids, set())

    def test_kwd_diff_bm25(self):
        self.assertEqual(self.kwd_ids - self.bm25_ids, set())

    def test_kwd_diff_spctr(self):
        self.assertEqual(self.kwd_ids - self.spctr_ids, set())

    def test_spctr_diff_bm25(self):
        self.assertEqual(self.spctr_ids - self.bm25_ids, set())

    def test_spctr_diff_kwd(self):
        self.assertEqual(self.spctr_ids - self.kwd_ids, set())
    
    # Each test here is also a set difference, but this is checking to make sure that every ID that is recommended also has a row in its respective TSV. 
    def test_bm25_rec_id_diff(self):
        self.assertEqual(self.bm25_rec_ids - self.bm25_ids, set())
    
    def test_kwd_rec_id_diff(self):
        self.assertEqual(self.kwd_rec_ids - self.kwd_ids, set())
    
    def test_spctr_rec_id_diff(self):
        self.assertEqual(self.spctr_rec_ids - self.spctr_ids, set())

if __name__ == '__main__':
    unittest.main()