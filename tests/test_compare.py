import pathlib
import unittest

from primalbedtools.bedfiles import (
    PrimerClass,
    lr_string_to_strand_char,
    primer_class_str_to_enum,
    strand_char_to_primer_class_str,
)

TEST_BEDFILE = pathlib.Path(__file__).parent / "inputs/test.bed"
TEST_V2_BEDFILE = pathlib.Path(__file__).parent / "inputs/test.v2.bed"
TEST_WEIGHTS_BEDFILE = pathlib.Path(__file__).parent / "inputs/test.weights.bed"
TEST_WEIGHTS_BEDFILE = pathlib.Path(__file__).parent / "inputs/test.weights.bed"
TEST_ATTRIBUTES_BEDFILE = pathlib.Path(__file__).parent / "inputs/test.attributes.bed"
TEST_PROBE_BEDFILE = pathlib.Path(__file__).parent / "inputs/test.probe.bed"


class TestValidationFuncs(unittest.TestCase):
    def test_string_to_strand_char(self):
        # Check expected
        self.assertEqual(lr_string_to_strand_char("LEFT"), "+")
        self.assertEqual(lr_string_to_strand_char("RIGHT"), "-")

        # Check unexpected
        with self.assertRaises(ValueError):
            lr_string_to_strand_char("")

    def test_primer_class_str_to_enum(self):
        self.assertEqual(primer_class_str_to_enum("LEFT"), PrimerClass.LEFT)
        self.assertEqual(primer_class_str_to_enum("RIGHT"), PrimerClass.RIGHT)
        self.assertEqual(primer_class_str_to_enum("PROBE"), PrimerClass.PROBE)

        # Check unexpected
        with self.assertRaises(ValueError):
            primer_class_str_to_enum("")

    def test_strand_char_to_primer_class_str(self):
        self.assertEqual(strand_char_to_primer_class_str("+"), "LEFT")
        self.assertEqual(strand_char_to_primer_class_str("-"), "RIGHT")
        # Check unexpected
        with self.assertRaises(ValueError):
            strand_char_to_primer_class_str("")
