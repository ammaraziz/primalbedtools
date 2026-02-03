"""
Extracted from artic-network run report script
https://raw.githubusercontent.com/artic-network/amplicon-nf/refs/heads/main/modules/local/generate_run_report/templates/generate_run_report.py
"""

from dataclasses import dataclass
from typing import Dict

# from primalbedtools.fasta import read_fasta


def extend_ambiguous_base(base: str) -> str:
    """
    Extend an ambiguous base to its possible bases.
    """
    ambiguous = {
        "A": "A",
        "C": "C",
        "G": "G",
        "T": "T",
        "R": "AG",
        "Y": "CT",
        "S": "GC",
        "W": "AT",
        "K": "GT",
        "M": "AC",
        "B": "CGT",
        "D": "AGT",
        "H": "ACT",
        "V": "ACG",
        "N": "ACGT",
        "-": "-",
    }
    return ambiguous.get(base.upper(), base)


def calc_primer_hamming(seq1, seq2) -> int:
    """
    Calculate the hamming distance between two sequences of equal length. Ignores N.

    :param seq1: The primer sequence in 5' to 3' orientation.
    :param seq2: The primer sequence in 5' to 3' orientation.
    :return: The number of mismatches between the two sequences.
    """
    dif = 0
    for seq1b, seq2b in zip(seq1[::-1], seq2[::-1]):
        seq1b_exp = set(extend_ambiguous_base(seq1b))
        seq2b_exp = set(extend_ambiguous_base(seq2b))
        if not seq1b_exp & seq2b_exp and (seq1b != "N" and seq2b != "N"):
            dif += 1
    return dif


# ============================================================================
# GAP HANDLING FOR MISMATCH ANALYSIS
# ============================================================================


def check_for_end_on_gap(ref_index_to_msa: Dict[int, int], ref_index) -> bool:
    """
    Check if a slice of a mapping array ends on a gap.

    Returns True if the slice ends on a gap, False otherwise.

    Example scenario:
    5' AGAGTGTGGGGGTAGTGTTACG > MPXV_142_LEFT_1 170931:170953
    TTTTTTTTATAGAGTGTGGGGGTAGTGTTACG-----GAT >MT903345
    TTTTTTTTATAGAGTGT-GGGGTAGTGTTACGGATATCTGAT >KJ642613.1
    ^ 173598: MSA
    ^ 170953: ref

    In this example, the slice of the primer ends on a gap. So slicing the array with
    - array[:, ref_to_msa[170931]:ref_to_msa[170953]] will return "GGGGTAGTGTTACG-----"
    - fix_end_on_gap() will return indexes to slice the array without the gap
    """
    exclusive_msa_end = ref_index_to_msa[ref_index]
    inclusive_msa_end = ref_index_to_msa[ref_index - 1]
    return exclusive_msa_end - inclusive_msa_end != 1


def fix_end_on_gap(ref_index_to_msa: Dict[int, int], ref_index) -> int:
    """
    Returns the MSA index of the non-inclusive end of a slice with the gap removed.
    """
    return ref_index_to_msa[ref_index - 1] + 1


# ============================================================================
# TEXT FORMATTING FOR MISMATCH VISUALIZATION
# ============================================================================


@dataclass
class PlotlyText:
    """
    A class to hold the text for a plotly heatmap.
    """

    primer_name: str
    primer_seq: str
    genome_seq: str

    def __init__(
        self,
        primer_name: str,
        primer_seq: str,
        genome_seq: str,
    ):
        self.primer_name = primer_name
        self.primer_seq = primer_seq
        self.genome_seq = genome_seq

    def format_str(self) -> str:
        # Build cigar string showing matches (|) and mismatches (.)
        cigar = []
        for p, g in zip(self.primer_seq[::-1], self.genome_seq[::-1]):
            if p == g:
                cigar.append("|")
            else:
                cigar.append(".")
        cigar = "".join(cigar)[::-1]
        return f"5'{self.primer_seq}: {self.primer_name}\n5'{cigar}\n5'{self.genome_seq[-len(self.primer_seq) :]}"


# ============================================================================
# MISMATCH ANALYSIS EXAMPLE USAGE
# ============================================================================


def main():
    """Demonstrate the mismatch analysis functionality."""

    # Example sequences for mismatch analysis
    primer_seq = "AGAGTGTGGGGGTAGTGTTACG"
    genome_seq1 = "AGAGTGTGGGGGTAGTGTTACG"  # Perfect match
    genome_seq2 = "AGAGTGTGGGGGTAGTGTTATG"  # 1 mismatch at position -3
    genome_seq3 = "AGAGTGTGGGGGTAGTGTTGCG"  # 1 mismatch at position -4

    print("=" * 60)
    print("MISMATCH ANALYSIS DEMONSTRATION")
    print("=" * 60)

    # Test Hamming distance calculation
    print(f"\nPrimer: {primer_seq}")
    print(f"Genome 1: {genome_seq1}")
    print(f"  Mismatches: {calc_primer_hamming(primer_seq, genome_seq1)}")

    print(f"\nGenome 2: {genome_seq2}")
    print(f"  Mismatches: {calc_primer_hamming(primer_seq, genome_seq2)}")

    print(f"\nGenome 3: {genome_seq3}")
    print(f"  Mismatches: {calc_primer_hamming(primer_seq, genome_seq3)}")

    # Test text formatting
    print("\n" + "=" * 60)
    print("MISMATCH VISUALIZATION")
    print("=" * 60)

    text = PlotlyText(
        primer_name="MPXV_142_LEFT_1", primer_seq=primer_seq, genome_seq=genome_seq2
    )
    print(f"\n{text.format_str()}")

    # Test with ambiguous bases
    print("\n" + "=" * 60)
    print("AMBIGUOUS BASE HANDLING")
    print("=" * 60)

    primer_ambig = "AGAGTGTGGGGGTAGTGTTACG"
    genome_ambig = "AGAGTGTGGGGGTAGTGTTAYG"  # Y = C or T

    print(f"\nPrimer: {primer_ambig}")
    print(f"Genome (Y=C/T): {genome_ambig}")
    print(f"  Mismatches: {calc_primer_hamming(primer_ambig, genome_ambig)}")
    print("  (Y matches T, so no mismatch counted at that position)")

    # Test gap detection
    print("\n" + "=" * 60)
    print("GAP DETECTION")
    print("=" * 60)

    # Simulated mapping: reference index -> MSA index
    ref_to_msa = {
        170930: 173595,
        170931: 173596,
        170932: 173597,
        170933: 173598,  # Gap starts here
        170934: 173604,  # Skip of 6 indicates gap
        170935: 173605,
    }

    # Check if position 170934 ends on a gap
    ends_on_gap = check_for_end_on_gap(ref_to_msa, 170934)
    print(f"\nPosition 170934 ends on gap: {ends_on_gap}")

    if ends_on_gap:
        fixed_index = fix_end_on_gap(ref_to_msa, 170934)
        print(f"Fixed MSA index: {fixed_index}")


if __name__ == "__main__":
    main()
