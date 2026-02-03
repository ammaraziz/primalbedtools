from pathlib import Path

from primalbedtools.fasta import read_fasta


def align(fasta: Path, reference: str) -> psa.PairwiseAlignment:
    """
    Given a
    """
    msa = read_fasta(fasta)
