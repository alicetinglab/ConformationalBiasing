from __future__ import annotations

from colabdesign.af.alphafold.common import protein

import numpy as np
from typing import List

from colabdesign.shared.protein import pdb_to_string
from scipy.special import softmax

from Bio.Align import PairwiseAligner


# Set of valid amino acid one-letter codes
AA_ALPHABET = set("ACDEFGHIKLMNPQRSTVWY")
# ProteinMPNN amino acid order for sequence encoding
MPNN_AA_ORDER = "ARNDCQEGHILKMFPSTWYV"

# Dictionary mapping one-letter amino acid codes to three-letter codes
aa_code = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "E": "GLU",
    "Q": "GLN",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}


def find_mismatch_positions(seq1, seq2):
    """
    Find positions where two sequences differ.

    Args:
        seq1: First sequence
        seq2: Second sequence

    Returns:
        List of indices where sequences mismatch
    """
    mm = []
    for idx, (c1, c2) in enumerate(zip(seq1, seq2)):
        if c1 != c2:
            mm.append(idx)
    return mm


def n_mismatches(seq1, seq2):
    """
    Count the number of mismatches between two sequences.

    Args:
        seq1: First sequence
        seq2: Second sequence

    Returns:
        Number of mismatches
    """
    return len(find_mismatch_positions(seq1, seq2))


def get_mpnn_seq(pdb_fp, chain):
    """
    Extract amino acid sequence from PDB file in ProteinMPNN format.

    Args:
        pdb_fp: Path to PDB file
        chain: Chain identifier

    Returns:
        Amino acid sequence string
    """
    pdb_str = pdb_to_string(pdb_fp, chains=[chain], models=[1])
    protein_obj = protein.from_pdb_string(pdb_str, chain_id=chain)

    return "".join([MPNN_AA_ORDER[i] for i in protein_obj.aatype])


def make_consensus_sequence(sequences: List[str]) -> str:
    """
    Create a consensus sequence from multiple aligned sequences.

    Args:
        sequences: List of sequences to align

    Returns:
        Consensus sequence string
    """
    aligner = setup_aligner()

    template_seq = sequences[0]

    for seq in sequences[1:]:
        alignment = aligner.align(template_seq, seq)[0]
        seq1, seq2 = alignment[0], alignment[1]

        template_seq = ""
        for a, b in zip(seq1, seq2):
            if a in MPNN_AA_ORDER:
                template_seq += a
            elif b in MPNN_AA_ORDER:
                template_seq += b
            else:
                template_seq += "*"

    return template_seq


def setup_aligner():
    """
    Configure and return a PairwiseAligner for sequence alignment.

    Returns:
        Configured PairwiseAligner object
    """
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    aligner.match_score = 2
    aligner.mismatch_score = -0.5

    return aligner


def alignment_to_mapping(alignment, allow_mismatches=True):
    """
    Convert sequence alignment to position mapping between sequences.

    Args:
        alignment: Pairwise alignment result
        allow_mismatches: Whether to include mismatched positions in mapping

    Returns:
        Dictionary mapping positions from first to second sequence
    """
    mapping = {}

    current_a_idx = 0
    current_b_idx = 0
    n_mismatches = 0

    for i, (a, b) in enumerate(zip(alignment[0], alignment[1])):
        if a != "-" and b != "-":
            if allow_mismatches or a == b:
                mapping[current_a_idx] = current_b_idx
            else:
                n_mismatches += 1

            current_a_idx += 1
            current_b_idx += 1

        elif a != "-":
            current_a_idx += 1
        elif b != "-":
            current_b_idx += 1

    if allow_mismatches and n_mismatches > 0:
        print(f"WARNING: {n_mismatches} mismatches included in alignment!")

    return mapping


def mapping_to_sequence(scaffold_seq, target_seq, mapping, fill=True, fill_char="A"):
    """
    Create a scorable sequence by mapping scaffold sequence to target positions.

    Args:
        scaffold_seq: Source sequence to map
        target_seq: Target sequence for gap filling
        mapping: Position mapping dictionary
        fill: Whether to fill gaps with target sequence
        fill_char: Character to use for filling invalid positions

    Returns:
        Mapped sequence string
    """
    seq = ["-"] * (np.max(list(mapping.values())) + 1)

    for i, c in enumerate(scaffold_seq):
        if i in mapping:
            seq[mapping[i]] = c

    seq = "".join(seq)
    seq = [c for c in seq.strip("-")]  # remove leading/trailing hyphens
    # fill in gaps with target seq
    for i in range(len(seq)):
        if seq[i] == "-":
            seq[i] = target_seq[i]

    if fill:
        filled = 0
        for i, val in enumerate(seq):
            if val not in AA_ALPHABET:
                seq[i] = fill_char
                filled += 1

        if filled > 0:
            print(f"WARNING: Filled in {filled} missing AAs with {fill_char}")

    else:
        if "-" in seq:
            raise ValueError("Incomplete Mapping Provided!")
        if "*" in seq or "?" in seq:
            raise ValueError(
                "Incomplete sequences provided (wild-card characters present)"
            )

    return "".join(seq)


def mpnn_score(seq, model, redesign_aa_pos=None, return_indiv=False):
    """
    Score a sequence using ProteinMPNN model.


    Args:
        seq: Amino acid sequence to score
        model: ProteinMPNN model
        redesign_aa_pos: Specific positions to score (optional)
        return_indiv: Whether to return individual position scores

    Returns:
        Log probability score(s)
    """
    L = len(seq)
    ar_mask = 1 - np.eye(L)
    outputs = model.score(seq=seq, ar_mask=ar_mask)
    pssm = softmax(outputs["logits"], -1)
    probs = np.squeeze(pssm[outputs["S"] == 1])
    if redesign_aa_pos is not None:
        probs = [probs[i] for i in redesign_aa_pos]

    if return_indiv:
        return np.log(probs)
    else:
        return np.log(probs).sum()
