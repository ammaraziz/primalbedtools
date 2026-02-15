# from primalbedtools.fasta import read_fasta
# from primalbedtools.scheme import Scheme
# from primalbedtools.bedfiles import group_by_amplicon_number

from dataclasses import dataclass
from pathlib import Path

import numpy as np

"""
Input: msa, reference, bed file

MSA needs to be a "core genome" alignment - no inserts are allowed. Therefore only augur align and nextclade msa are accepted.

Steps:
    1. Read in bed file using Scheme()
    2. group primers by amplicon group_by_amplicon_number()
    3. For each primer in each amplicon group:
        - calc # of mismatches
        - 
        - append results to dict indexed by group number
    4. rate each primer by the follownig rules:
        - 
"""

ALL_DNA = {
    "A": "A",
    "C": "C",
    "G": "G",
    "T": "T",
    "M": "AC",
    "R": "AG",
    "W": "AT",
    "S": "CG",
    "Y": "CT",
    "K": "GT",
    "V": "ACG",
    "H": "ACT",
    "D": "AGT",
    "B": "CGT",
}


def extend_ambiguous_base(base: str) -> list[str]:
    """Return list of all possible sequences given an ambiguous DNA input"""
    return [*ALL_DNA.get(base, "N")]


@dataclass
class MultipleSequenceAlignment:
    """
    Multiple Sequence Alignment container.
    """

    names: list[str]
    sequences: np.ndarray

    def __post_init__(self):
        """Build name-to-index mapping after initialization."""
        self._name_to_idx = {name: i for i, name in enumerate(self.names)}

    def __getitem__(self, key: str | int) -> np.ndarray:
        """Enable msa['seq_name'] or msa[0] syntax."""
        if isinstance(key, str):
            if key not in self._name_to_idx:
                raise KeyError(f"Sequence '{key}' not found")
            return self.sequences[self._name_to_idx[key]]
        else:
            return self.sequences[key]

    def get_sequence(self, name: str) -> np.ndarray:
        """Get sequence array by name."""
        try:
            idx = self.names.index(name)
        except ValueError:
            raise KeyError(f"Sequence '{name}' not found")
        return self.sequences[idx]

    def get_index(self, name: str) -> int:
        """Get index of a sequence name."""
        try:
            return self.names.index(name)
        except ValueError:
            raise KeyError(f"Sequence '{name}' not found")

    def _check_row_index(self, index: int) -> None:
        """Validate row index bounds."""
        n_rows = self.sequences.shape[0]
        if not 0 <= index < n_rows:
            raise IndexError(
                f"Row index {index} out of bounds for MSA with {n_rows} sequences "
                f"(valid: 0 to {n_rows - 1})"
            )

    def _check_column_bounds(self, start: int, stop: int) -> None:
        """Validate column slice bounds."""
        n_cols = self.sequences.shape[1]

        if start < 0:
            raise ValueError(f"Start position {start} must be non-negative")
        if stop >= n_cols:
            raise IndexError(
                f"Stop position {stop} out of bounds for alignment length {n_cols} "
                f"(valid: 0 to {n_cols - 1})"
            )
        if start > stop:
            raise ValueError(f"Start {start} must be <= stop {stop}")

    def extract_primer(self, key: int | str, start: int, stop: int) -> list[str]:
        """
        Extract primer sequence from a specific row by name or index.
        """
        if isinstance(key, str):
            index = self.get_index(key)
        else:
            index = key
            self._check_row_index(index)
        # Validate slice bounds
        self._check_column_bounds(start, stop)

        return self.sequences[index][start : stop + 1].tolist()


def read_msa(fasta_file: Path) -> MultipleSequenceAlignment:
    """
    Read a fasta file and return a 2D numpy array of characters.
    Rows are sequences, columns are alignment positions.
    """
    seq_dict: dict[str, list[str]] = {}
    sequences = {}

    with open(fasta_file) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                seq_name = line[1:].split()[0]
                if seq_name in seq_dict:
                    raise ValueError(f"Duplicate sequence name: {seq_name}")
                seq_dict[seq_name] = []
            else:
                seq_dict[seq_name].append(line)

    # Extract names and joined sequences
    names = list(seq_dict.keys())
    seq_list = ["".join(seq_dict[name]) for name in names]

    # Validate equal lengths
    seq_lengths = [len(s) for s in seq_list]
    if len(set(seq_lengths)) > 1:
        raise ValueError(
            f"Sequences have different lengths: {set(seq_lengths)}. "
            "Input must be a valid MSA."
        )

    # Create 2D numpy array
    sequences = np.array([list(seq) for seq in seq_list])

    return MultipleSequenceAlignment(names=names, sequences=sequences)


def calc_levenshtein_distance(seq1: np.array, seq2: np.array) -> int:
    """Calculate standard Levenshtein distance between two sequences."""
    D = np.zeros((len(seq1) + 1, len(seq2) + 1), dtype=int)
    D[0, 1:] = np.arange(1, len(seq2) + 1)
    D[1:, 0] = np.arange(1, len(seq1) + 1)

    for i in range(1, len(seq1) + 1):
        for j in range(1, len(seq2) + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            D[i, j] = min(
                D[i - 1, j] + 1,  # deletion
                D[i, j - 1] + 1,  # insertion
                D[i - 1, j - 1] + cost,  # substitution
            )

    return D[-1, -1], D


def calc_mismatch_degree(primer: np.array, query: np.array):
    """
    Calculate the degree of mismatches from the difflab optcodes

    rules for computing degrees of changes:

    deadly:
        - from 3' bases 1-3 affect massive, beyond 4 bases diminishing effects [Huang et al., 2024]
        - beyond 3', >11 mismatches blocks amplification
    mild:
        - from centre to 5', up to 8 mismatches causes mild [Huang et al., 2024]
        - >6 from middle to 5' mainly due to Tm changes

    Citation: https://doi.org/10.3390/genes15020215
    """
    print(primer)
    for p, q in zip(primer, query):
        print(p, q)


#     primer = "".join(primer[0])
#     query = "".join(query[0])
#     danger_zone = range(len(primer) - 3, len(primer))
#     primer_length = len(primer)

#     sm = difflib.SequenceMatcher(None, primer, query)
#     for tag, primer_start, primer_end, query_start, query_end in sm.get_opcodes():
#         print(tag, primer_start, primer_end, query_start, query_end)
#         qrange = range(query_start, query_end)
#         prange = range(primer_start, primer_end)
#         if tag == "equal":
#             if (
#                 min(qrange)
#                 == 0 & min(prange)
#                 == 0 & max(qrange)
#                 == primer_length & max(prange)
#                 == primer_length
#             ):
#                 return "identical"
#         if tag == "delete":
#             # first check if primer is obliterated - aka completely missing
#             if len(prange) == primer_length:
#                 return "deadly"
#             # within danger zone, this is deadly to primer binding
#             if set(prange).intersection(danger_zone):
#                 return "deadly"
#             # outside of danger zone, it's acceptable but within reason
#             # first quantify the number of deletions
#             # code goes here

#         if tag == "replace":
#             print("hit replace")
#             print(qrange)
#             # check if primer is completely different
#             if len(qrange) == primer_length:
#                 return "deadly"
#             if set(qrange).intersection(danger_zone):
#                 return "deadly"


def calc_primer_hamming(seq1, seq2) -> int:
    """
    Calculate the hamming distance between two sequences of equal length. Ignores N.
    :param seq1: The primer sequence in 5' to 3' orientation.
    :param seq2: The primer sequence in 5' to 3' orientation.
    :return: The number of mismatches between the two sequences.
    """
    if len(seq1) != len(seq2):
        raise ValueError(
            f"Sequence lenght must be identical. {seq1} is different from {seq2}"
        )

    dif = 0
    for seq1b, seq2b in zip(seq1[::-1], seq2[::-1]):
        seq1b_exp = set(extend_ambiguous_base(seq1b))
        seq2b_exp = set(extend_ambiguous_base(seq2b))

        if not seq1b_exp & seq2b_exp and (seq1b != "N" and seq2b != "N"):
            dif += 1
    return dif
