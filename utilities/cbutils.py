from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


import numpy as np
from typing import List

from scipy.special import softmax

from Bio.Align import PairwiseAligner

# Additional imports for new functions
import gemmi
import pandas as pd


# Set of valid amino acid one-letter codes
AA_ALPHABET = set("ACDEFGHIKLMNPQRSTVWY")
# Additional constants for new functions
AA_ALPHABET_STR = "ARNDCQEGHILKMFPSTWYV"

# Standard 20 amino acids (3-letter)
STANDARD_AA3 = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
}

# Common PDB residue variants & PTMs mapped to canonical residues.
# Extend as needed for your datasets.
NONCANONICAL_TO_CANONICAL = {
    # Selenium substitutions
    "MSE": "MET",  # Selenomethionine
    "SEC": "CYS",  # Selenocysteine (U) → treat as CYS
    "CSE": "CYS",  # Alternate code seen for Sec
    # Phosphorylations
    "PTR": "TYR",  # Phosphotyrosine
    "TPO": "THR",  # Phosphothreonine
    "SEP": "SER",  # Phosphoserine
    # Cys oxidations / variants
    "CSO": "CYS",  # Cys sulfinic acid
    "CSD": "CYS",
    "CME": "CYS",
    "CYM": "CYS",  # Deprotonated cysteine
    "CYX": "CYS",  # Disulfide-bonded form
    # His protonation states
    "HID": "HIS",
    "HIE": "HIS",
    "HIP": "HIS",
    # Acid/base tautomers
    "ASH": "ASP",
    "GLH": "GLU",
    "LYN": "LYS",
    # Hydroxyproline
    "HYP": "PRO",
    # Methyl-lysines (common ones)
    "MLY": "LYS",
    "M3L": "LYS",
    # Rare noncanonical → closest canonical
    "PYL": "LYS",  # Pyrrolysine (O) → treat as Lys
    # Catch-alls sometimes seen for modified Asn/Gln
    "MEN": "ASN",
    "MEQ": "GLN",
}

ONE_LETTER = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}

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


def canonicalize_resname(resname3: str) -> str:
    """
    Convert a 3-letter residue name to its canonical form.

    Handles common non-canonical residues (e.g., modified residues, PTMs)
    by mapping them to their parent canonical amino acid. For example:
    - MSE (selenomethionine) -> MET
    - PTR (phosphotyrosine) -> TYR
    - CSE (selenocysteine) -> CYS

    Parameters
    ----------
    resname3 : str
        3-letter residue code to canonicalize

    Returns
    -------
    str
        Canonical 3-letter residue code. If the input is unknown and not
        mapped to a canonical residue, returns the uppercased input.

    Notes
    -----
    The complete mapping of non-canonical to canonical residues is defined
    in the NONCANONICAL_TO_CANONICAL dictionary.
    """
    r = (resname3 or "").upper()
    if r in STANDARD_AA3:
        return r
    if r in NONCANONICAL_TO_CANONICAL:
        return NONCANONICAL_TO_CANONICAL[r]
    return r  # leave as-is; caller can choose to skip if not in STANDARD_AA3


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


def make_consensus_sequence(sequences):
    """takes input sequences for many pdbs and generates a consensus sequence,
    preferring by default earlier sequences in the input list if there are
    mismatches
    if any noncanonical AAs are encountered, adds * character
    """
    aligner = setup_aligner()
    template_seq = sequences[0]
    for seq in sequences[1:]:
        alignment = aligner.align(template_seq, seq)[0]
        seq1, seq2 = alignment[0], alignment[1]
        template_seq = ""
        for a, b in zip(seq1, seq2):
            if a in AA_ALPHABET:
                template_seq += a
            elif b in AA_ALPHABET:
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


def alignment_to_mapping(alignment, allow_mismatches=False):
    """
    Convert an alignment object to a dictionary that maps
    sequence A index -> sequence B index

    - allow_mismatches: whether aligned regions where AA residues
    don't match should be mapped
    """
    mapping = {}
    current_a_idx = 0
    current_b_idx = 0
    n_mismatches = 0

    # iterate along sequences and match up indices, skipping gaps
    for i, (a, b) in enumerate(zip(alignment[0], alignment[1])):
        if a != "-" and b != "-":
            if a == b or allow_mismatches:
                mapping[current_a_idx] = current_b_idx
            if a != b:
                n_mismatches += 1
            current_a_idx += 1
            current_b_idx += 1
        elif a != "-":
            current_a_idx += 1
        elif b != "-":
            current_b_idx += 1

    if n_mismatches > 0:
        if allow_mismatches:
            print(
                f"WARNING: {n_mismatches} positions with AA mismatch included in alignment!"
            )
        else:
            print(
                f"WARNING: {n_mismatches} positions with AA mismatch not included in alignment!"
            )

    return mapping


def mapping_to_sequence(scaffold_seq, target_seq, mapping, fill=True, fill_char="A"):
    """Use mapping dictionary to convert an input sequence to a sequence
    that can be scored on a target structure
    - scaffold seq: consensus sequence
    - target seq: sequence matching to a pdb
    - mapping: generated dict from alignment that maps scaffold seq indices
    to target seq indices
    """

    # create temp seq and start filling using scaffold seq and mapping dict
    seq = ["-"] * len(
        target_seq
    )  # CHECK: can mapping result in values > len(target_seq)?
    for i, c in enumerate(scaffold_seq):
        if i in mapping:
            seq[mapping[i]] = c

    # fill any missing values with target seq
    for i in range(len(seq)):
        if seq[i] == "-":
            seq[i] = target_seq[i]

    # in edge case where some positions are unfilled, either fill or throw error
    if fill:
        filled = 0
        for i, val in enumerate(seq):
            if val not in AA_ALPHABET:
                seq[i] = fill_char
                filled += 1
        if filled > 0:
            print(f"WARNING: Filled in {filled} missing AAs with {fill_char}")
    else:
        for i, val in enumerate(seq):
            if val not in AA_ALPHABET:
                raise ValueError(f"Invalid character ({val}) in position {i}, {seq}")

    return "".join(seq)


def mpnn_score(seq, model, return_indiv=False):
    """use AR mask and seq to calculate PMPNN log likelihood"""
    L = len(seq)
    ar_mask = 1 - np.eye(L)
    outputs = model.score(seq=seq, ar_mask=ar_mask)
    pssm = softmax(outputs["logits"], -1)
    probs = np.squeeze(pssm[outputs["S"] == 1])

    if return_indiv:
        return np.log(probs)
    else:
        return np.log(probs).sum()


def norm_scale(inp):
    """helper to standardize scores"""
    mean_val = np.mean(inp)
    std_val = np.std(inp)
    if std_val == 0:
        return np.zeros_like(inp)
    return (inp - mean_val) / std_val


def add_scaled_outputs(df, model: str):
    """helper function to scale outputs and combine"""
    df[f"{model}_state1_scaled"] = norm_scale(df[f"{model}_state1"])
    df[f"{model}_state2_scaled"] = norm_scale(df[f"{model}_state2"])
    df[f"{model}_state1_bias"] = (
        df[f"{model}_state1_scaled"] - df[f"{model}_state2_scaled"]
    )
    df[f"{model}_state2_bias"] = (
        df[f"{model}_state2_scaled"] - df[f"{model}_state1_scaled"]
    )


##############################################
# PDB helper functions
##############################################
def list_chain_residues(
    structure_path: str,
    chain_id: str,
    *,
    include_noncanonical=False,
    model_index: int = 0,
) -> List[Dict[str, Union[str, int]]]:
    """
    Read a PDB/mmCIF with gemmi and list residues in a specified chain.
    - Converts common noncanonical/modified residues to their canonical amino acids.
    - Positions include seq number and insertion code.
    - By default, skips residues that are not canonical or mapped to canonical.
      Set include_noncanonical=True to keep any unmapped residues.

    Returns a list of dicts like:
      {"resname_raw": "PTR", "resname": "TYR", "one_letter": "Y",
       "seqnum": 123, "icode": "", "chain": "A"}
    """
    st = gemmi.read_structure(structure_path)
    if model_index < 0 or model_index >= len(st):
        raise IndexError(
            f"Model index {model_index} out of range (structure has {len(st)} models)."
        )

    model = st[model_index]

    # Try to find the exact chain ID; gemmi stores polymer chains in model.walk_chains()
    target_chain = None
    for ch in model:
        if ch.name == chain_id:
            target_chain = ch
            break
    if target_chain is None:
        # Also check entity-based iteration (e.g., for mmCIF)
        for ch in model.chains:
            if ch.name == chain_id:
                target_chain = ch
                break
    if target_chain is None:
        raise ValueError(
            f"Chain '{chain_id}' not found in model {model_index} of {structure_path!r}."
        )

    out: List[Dict[str, Union[str, int]]] = []
    for (
        res
    ) in target_chain.get_polymer():  # polymer residues only (skips ligands/solvent)
        raw3 = res.name.upper()
        can3 = canonicalize_resname(raw3)

        # Decide whether to keep
        is_canonical = can3 in STANDARD_AA3
        if not is_canonical and not include_noncanonical:
            # skip unknown/unmapped non-amino-acid residues (e.g., caps ACE/NME, ligands)
            continue

        one = ONE_LETTER[can3]
        seqnum = res.seqid.num
        icode = (
            res.seqid.icode if res.seqid.icode != "\x00" else ""
        )  # handle insertion codes, empty if none

        out.append(
            {
                "resname_raw": raw3,  # as in file
                "resname": can3,  # canonicalized 3-letter
                "one_letter": one,
                "seqnum": seqnum,
                "icode": icode,
                "chain": chain_id,
            }
        )

    return out


def convert_reslist_to_seq(
    res_list: List[Dict[str, Union[str, int]]], pad_negative: bool = False
) -> str:
    """
    Convert a list of residue information dictionaries to a linear amino acid sequence.

    Builds a sequence from residue number 1 to the maximum residue number,
    handling gaps, insertion codes, and optionally negative residue numbers.

    Parameters
    ----------
    res_list : List[Dict[str, Union[str, int]]]
        List of residue dictionaries, each containing:
        - resname: 3-letter residue code
        - one_letter: 1-letter residue code
        - seqnum: Residue sequence number
        - icode: Insertion code (if any)
    pad_negative : bool, optional
        If True, shifts negative residue numbers to positive by adding an offset,
        default False

    Returns
    -------
    str
        Linear amino acid sequence where:
        - '*' represents missing residues
        - For residues with same sequence number (insertions),
          prefers the one without insertion code

    Notes
    -----
    The function handles several edge cases:
    1. Gaps in numbering are filled with '*'
    2. Multiple residues at same position (insertions) are resolved by
       preferring the one without insertion code
    3. Negative sequence numbers can be shifted positive with pad_negative=True
    """
    if not res_list:
        return ""

    # Determine length (max seqnum). Ignore non-positive seqnums for the linear sequence.
    if pad_negative:
        min_seqnum = min(
            (r["seqnum"] for r in res_list if isinstance(r.get("seqnum"), int)),
            default=0,
        )
        if min_seqnum <= 0:
            pad_val = 1 - min_seqnum

            res_list = [
                {
                    **r,
                    "seqnum": r["seqnum"] + pad_val,
                }
                for r in res_list
            ]

    max_seqnum = max(
        (r["seqnum"] for r in res_list if isinstance(r.get("seqnum"), int)), default=0
    )
    if max_seqnum <= 0:
        return ""

    seq = ["*"] * max_seqnum
    chosen_icode = {}  # seqnum -> icode used

    for r in res_list:
        seqnum = r.get("seqnum")
        if not isinstance(seqnum, int) or seqnum <= 0 or seqnum > max_seqnum:
            continue

        # Determine one-letter (use provided if present, else map from canonical 3-letter)
        one = (r.get("one_letter") or "").strip()
        if not one:
            can3 = (r.get("resname") or "").upper()
            one = ONE_LETTER[can3]

        icode = r.get("icode", "") or ""
        cur = seq[seqnum - 1]

        if cur == "*":
            # nothing placed yet at this position
            seq[seqnum - 1] = one
            chosen_icode[seqnum] = icode
        else:
            # Something is already there; resolve by preferring empty icode
            prev_icode = chosen_icode.get(seqnum, "")
            if prev_icode and not icode:
                # replace a letter that came from an inserted residue with the main one
                seq[seqnum - 1] = one
                chosen_icode[seqnum] = icode
            # else: keep the first seen

    return "".join(seq)


def get_chain_seq(pdb: str, chain: str) -> str:
    """
    Extract the complete amino acid sequence from a specific chain in a PDB file.

    Parameters
    ----------
    pdb : str
        Path to the PDB file
    chain : str
        Chain identifier (e.g., 'A')

    Returns
    -------
    str
        Complete amino acid sequence including gaps ('*' for missing residues)
        and padded negative residue numbers if present
    """
    return convert_reslist_to_seq(list_chain_residues(pdb, chain), pad_negative=True)


def get_chain_seq_for_scoring(pdb: str, chain: str) -> str:
    """
    Get the scorable sequence from a chain by removing unresolved residues.

    Parameters
    ----------
    pdb : str
        Path to the PDB file
    chain : str
        Chain identifier (e.g., 'A')

    Returns
    -------
    str
        Amino acid sequence containing only resolved residues
        (unresolved '*' positions removed)
    """
    return get_chain_seq(pdb, chain).replace("*", "")


def norm_scale(inp):
    """helper to standardize scores"""
    mean_val = np.mean(inp)
    std_val = np.std(inp)
    if std_val == 0:
        return np.zeros_like(inp)
    return (inp - mean_val) / std_val


def add_scaled_outputs(df: pd.DataFrame, model: str, state1_col: str, state2_col: str):
    """helper function to scale outputs and combine"""
    df[f"{model}_state1_scaled"] = norm_scale(df[f"{model}_{state1_col}"])
    df[f"{model}_state2_scaled"] = norm_scale(df[f"{model}_{state2_col}"])
    df[f"{model}_state1_bias"] = (
        df[f"{model}_state1_scaled"] - df[f"{model}_state2_scaled"]
    )
    df[f"{model}_state2_bias"] = (
        df[f"{model}_state2_scaled"] - df[f"{model}_state1_scaled"]
    )
