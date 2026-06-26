import unittest

import numpy as np

from primalbedtools.compare import (
    MultipleSequenceAlignment,
    calc_levenshtein_distance,
    calc_mismatch_degree,
    calc_primer_hamming,
)


class TestMSAGetItem(unittest.TestCase):
    def setUp(self):
        """
        Set up test MSA with 3 sequences of length 4.
        """
        self.names = ["seq1", "seq2", "seq3"]
        self.sequences = np.array([list("ACGT"), list("TGCA"), list("AATT")])
        self.msa = MultipleSequenceAlignment(names=self.names, sequences=self.sequences)

    def test_getitem_by_name(self):
        """Test accessing sequence by name string."""
        seq = self.msa["seq2"]
        expected = np.array(["T", "G", "C", "A"])
        np.testing.assert_array_equal(seq, expected)

    def test_getitem_by_index(self):
        """Test accessing sequence by integer index."""
        seq = self.msa[1]
        expected = np.array(["T", "G", "C", "A"])
        np.testing.assert_array_equal(seq, expected)

    def test_getitem_name_not_found(self):
        """Test KeyError raised for non-existent name."""
        with self.assertRaises(KeyError):
            self.msa["seq99"]

    def test_getitem_index_out_of_range(self):
        """Test IndexError raised for out-of-range index."""
        with self.assertRaises(IndexError):
            self.msa[99]

    def test_getitem_negative_index(self):
        """Test negative integer indexing."""
        seq = self.msa[-1]
        expected = np.array(["A", "A", "T", "T"])
        np.testing.assert_array_equal(seq, expected)
        np.testing.assert_array_equal(seq, expected)


class TestPrimerExtract(unittest.TestCase):
    def setUp(self):
        """
        Set up test MSA with 3 sequences of length 4.
        """
        self.names = ["seq1", "seq2", "seq3"]
        self.sequences = np.array(
            [list("ACGTGGGT"), list("TGCAGGGT"), list("AATTGGGT")]
        )
        self.msa = MultipleSequenceAlignment(names=self.names, sequences=self.sequences)

    def test_extract_index(self):
        result = self.msa.extract_primer(key=0, start=2, stop=5)
        self.assertEqual(result, list("GTGG"))

    def test_extract_primer_by_name(self):
        """Test extracting primer by sequence name."""
        result = self.msa.extract_primer(key=0, start=2, stop=5)
        self.assertEqual(result, list("GTGG"))

    def test_extract_primer_single_base(self):
        """Test extracting single base (start == stop)."""
        result = self.msa.extract_primer("seq2", 4, 4)
        self.assertEqual(result, ["G"])

    def test_extract_primer_full_length(self):
        """Test extracting full sequence."""
        result = self.msa.extract_primer(0, 0, 7)
        self.assertEqual(result, list("ACGTGGGT"))

    def test_extract_primer_negative_index(self):
        """Test negative row index raises IndexError."""
        with self.assertRaises(IndexError) as cm:
            self.msa.extract_primer(-1, 0, 5)
        self.assertIn("Row index -1 out of bounds", str(cm.exception))

    def test_extract_primer_index_too_large(self):
        """Test row index beyond range raises IndexError."""
        with self.assertRaises(IndexError) as cm:
            self.msa.extract_primer(5, 0, 5)
        self.assertIn("Row index 5 out of bounds", str(cm.exception))
        self.assertIn("valid: 0 to 2", str(cm.exception))

    def test_extract_primer_negative_start(self):
        """Test negative start position raises ValueError."""
        with self.assertRaises(ValueError) as cm:
            self.msa.extract_primer(0, -1, 5)
        self.assertIn("Start position -1 must be non-negative", str(cm.exception))

    def test_extract_primer_invalid_name(self):
        """Test invalid sequence name raises KeyError."""
        with self.assertRaises(KeyError) as cm:
            self.msa.extract_primer("seq_Z", 0, 5)
        self.assertIn("Sequence 'seq_Z' not found", str(cm.exception))

    def test_getitem_by_name(self):
        """Test dictionary-style access by name."""
        result = self.msa["seq3"]
        np.testing.assert_array_equal(result, self.sequences[2])

    def test_getitem_by_index(self):
        """Test dictionary-style access by index."""
        result = self.msa[2]
        np.testing.assert_array_equal(result, self.sequences[2])

    def test_getitem_invalid_name(self):
        """Test KeyError for invalid name in __getitem__."""
        with self.assertRaises(KeyError) as cm:
            self.msa["unknown"]
        self.assertIn("Sequence 'unknown' not found", str(cm.exception))


class TestCalcHammingDistance(unittest.TestCase):
    def setUp(self):
        """
        Set up test MSA with 3 sequences of length 4.
        """
        self.names = ["seq1", "seq2", "seq3"]
        self.sequences = np.array(
            [list("ACGTGGGT"), list("TGCAGGGT"), list("AATTGGGT")]
        )
        self.msa = MultipleSequenceAlignment(names=self.names, sequences=self.sequences)

    def test_hamming_distance_gap(self):
        """
        Test hamming distance - gap
        """
        primer = np.array(list("ACGTGGG-"))
        result = calc_primer_hamming(seq1=self.msa["seq1"], seq2=primer)
        self.assertEqual(result, 1)

    def test_hamming_distance_diff(self):
        """
        Test hamming distance - 1 snp diff
        """
        primer = np.array(list("ACGTGGGA"))
        result = calc_primer_hamming(seq1=self.msa["seq1"], seq2=primer)
        self.assertEqual(result, 1)

    def test_hamming_distance_diff2(self):
        """
        Test hamming distance - 1 snp diff
        """
        primer = np.array(list("ACYTGGGA"))
        result = calc_primer_hamming(seq1=self.msa["seq1"], seq2=primer)
        self.assertEqual(result, 2)

    def test_hamming_distance_diffall(self):
        """
        Test hamming distance - 1 snp diff
        """
        primer = np.array(list("TCCCACGT"))
        result = calc_primer_hamming(seq1=self.msa["seq1"], seq2=primer)
        self.assertEqual(result, 5)


class TestCalcLevDistance(unittest.TestCase):
    def setUp(self):
        """
        Set up test MSA with 3 sequences of length 4.
        """
        self.names = ["seq1"]
        self.sequences = np.array([list("ACGTGGGT")])
        self.msa = MultipleSequenceAlignment(names=self.names, sequences=self.sequences)

    def test_equal(self):
        """
        Test same sequence
        """
        primer = np.array(list("ACGTGGGT"))
        result = calc_levenshtein_distance(seq1=self.msa["seq1"], seq2=primer)
        self.assertEqual(result[0], 0)

    def test_diff_one(self):
        """
        Test one difference
        """
        primer = np.array(list("ACGTGGGA"))
        result = calc_levenshtein_distance(seq1=self.msa["seq1"], seq2=primer)
        self.assertEqual(result[0], 1)

    def test_diff_del(self):
        """
        Test deletion x 2
        """
        primer = np.array(list("ACGGGG"))
        result = calc_levenshtein_distance(seq1=self.msa["seq1"], seq2=primer)
        self.assertEqual(result[0], 2)

    def test_diff_insert(self):
        """
        Test insertion (extra A at the end)
        """
        primer = np.array(list("ACGTGGGAA"))
        result = calc_levenshtein_distance(seq1=self.msa["seq1"], seq2=primer)
        self.assertEqual(result[0], 2)

    def test_diff_all(self):
        """
        Test completely different
        """
        primer = np.array(list("YYYYYYYY"))
        result = calc_levenshtein_distance(seq1=self.msa["seq1"], seq2=primer)
        self.assertEqual(result[0], 8)

    def test_gap(self):
        """
        Test completely different
        """
        primer = np.array(list("--------"))
        result = calc_levenshtein_distance(seq1=self.msa["seq1"], seq2=primer)
        self.assertEqual(result[0], 8)

    def test_diff_empty(self):
        """
        Test empty comparison
        """
        primer = np.array(list(""))
        result = calc_levenshtein_distance(seq1=self.msa["seq1"], seq2=primer)
        self.assertEqual(result[0], 8)


class TestDiff(unittest.TestCase):
    """
    Test diffseq()
    """

    def setUp(self):
        """
        Set up test MSA to represent of sequence/primer alignment.
        """
        self.names = [
            "primer",
            "identical",
            "obliterated",
            "single_deletion",
            "three_prime_1",
            "three_prime_3",
            "centre_1",
        ]
        self.sequences = np.array(
            [
                list("ACGTGGGTATGATGCACGTGGGTATGATGC"),  # ref
                list("ACGTGGGTATGATGCACGTGGGTATGATGC"),  # identicaly - clean
                list("------------------------------"),  # obliterated - deadly
                list("ACGTGGGTATGATGCACGTGGGTATGATGX"),  # end deletion - deadly
                list("ACGTGGGTATGATGCACGTGGGTATGATGX"),  # 3' single diff - deadly
                list("ACGTGGGTATGATGCACGTGGGTATGAXXX"),  # 3' three diff - deadly
                list("ACGTGGGTATGATGCXCGTGGGTATGATGC"),  # middle single diff - moderate
                list(
                    "ACGTGGGTATGATGCXCGTGGGTATGATGC"
                ),  # middle consequetive diff - deadly
            ]
        )
        self.msa = MultipleSequenceAlignment(names=self.names, sequences=self.sequences)

    def test_diff_identical(self):
        primer = np.array(["ACGTGGGTATGATGCACGTGGGTATGATGC"], dtype="<U1")
        sample = np.array(["ACGTGGGTATGATGCACGTGGGTATGATGC"], dtype="<U1")
        result = calc_mismatch_degree(primer, sample)
        self.assertEqual(result["status"], "identical")

    def test_diff_obliterated(self):
        primer = np.array(["ACGTGGGT"], dtype="<U1")
        sample = np.array([""], dtype="<U1")
        result = calc_mismatch_degree(primer, sample)
        self.assertEqual(result["status"], "deadly")

    def test_diff_single_deletion(self):
        primer = np.array(["ACGTGGGT"], dtype="<U1")
        sample = np.array(["ACGTGGG"], dtype="<U1")
        result = calc_mismatch_degree(primer, sample)
        self.assertEqual(result["status"], "deadly")

    def test_diff_three_prime_1(self):
        primer = np.array(["ACGTGGGT"], dtype="<U1")
        sample = np.array(["ACGTGGGC"], dtype="<U1")
        result = calc_mismatch_degree(primer, sample)
        self.assertEqual(result["status"], "deadly")

    def test_diff_three_prime_3(self):
        primer = np.array(["ACGTGGGT"], dtype="<U1")
        sample = np.array(["ACGTGCCC"], dtype="<U1")
        result = calc_mismatch_degree(primer, sample)
        self.assertEqual(result["status"], "deadly")

    def test_diff_centre(self):
        primer = np.array(["ACGTGGGTATGATGC"], dtype="<U1")
        sample = np.array(["ACGTGGGCATGATGC"], dtype="<U1")
        result = calc_mismatch_degree(primer, sample)
        self.assertEqual(result["status"], "mild")

    def test_diff_mild2(self):
        primer = np.array(["ACGTGGGTATGATGC"], dtype="<U1")
        sample = np.array(["ACGTGGCCATGATGC"], dtype="<U1")
        result = calc_mismatch_degree(primer, sample)
        self.assertEqual(result["status"], "mild")

    def test_diff_mild3(self):
        primer = np.array(["ACGTGGGTATGATGC"], dtype="<U1")
        sample = np.array(["TGGCCATGATGC"], dtype="<U1")
        result = calc_mismatch_degree(primer, sample)
        self.assertEqual(result["status"], "mild")

    def test_diff_deadly_10(self):
        primer = np.array(["ACGTGGGTATGATGCACGTGGGTATGATGC"], dtype="<U1")
        sample = np.array(["GGGGGGGTATGATGCACGTGGGTATGATGC"], dtype="<U1")
        result = calc_mismatch_degree(primer, sample)
        self.assertEqual(result["status"], "deadly")
