import csv
import unittest


def findAllRecIDs(file_path):
    """For each recommendation system, puts all of the recommendation IDs into a set
    from a CSV file."""
    rec_set = set()
    with open(file_path, newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter="\t")
        next(reader)
        for row in reader:
            for ind in range(5, 15):
                rec_set.add(row[ind])
    return rec_set


def get_article_ids(file_path):
    """Extracts article IDs from the specified column of a TSV file into a set."""
    ids_set = set()
    with open(file_path, newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter="\t")
        header = next(reader)
        article_id_index = header.index("Article ID")
        for row in reader:
            ids_set.add(row[article_id_index])
    return ids_set


class CheckIDs(unittest.TestCase):
    """Unit test for testing the equality of all three sets of recommendation IDs for
    bm25, kwd, and spctr TSVs"""

    @classmethod
    def setUpClass(cls):
        # Read in TSVs
        bm25_path = "dhq-recs-zfill-bm25.tsv"
        kwd_path = "dhq-recs-zfill-kwd.tsv"
        spctr_path = "dhq-recs-zfill-spctr.tsv"

        # Grab the Article IDs into a set
        cls.bm25_ids = get_article_ids(bm25_path)
        cls.kwd_ids = get_article_ids(kwd_path)
        cls.spctr_ids = get_article_ids(spctr_path)

        # Calculate the sets for all of the recommendation IDs
        cls.bm25_rec_ids = findAllRecIDs(bm25_path)
        cls.kwd_rec_ids = findAllRecIDs(kwd_path)
        cls.spctr_rec_ids = findAllRecIDs(spctr_path)

    # Each test is a set difference, so if set_A_diff_set_B results in an ID, then that
    # ID is in A and not B.
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

    # Each test here is also a set difference, but this is checking to make sure that
    # every ID that is recommended also has a row in its respective TSV.
    def test_bm25_rec_id_diff(self):
        self.assertEqual(self.bm25_rec_ids - self.bm25_ids, set())

    def test_kwd_rec_id_diff(self):
        self.assertEqual(self.kwd_rec_ids - self.kwd_ids, set())

    def test_spctr_rec_id_diff(self):
        self.assertEqual(self.spctr_rec_ids - self.spctr_ids, set())


if __name__ == "__main__":
    unittest.main()
