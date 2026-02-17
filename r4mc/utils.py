"""Utility functions and RDKit wrapper."""

import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# Try to import RDKit; gracefully degrade if not available
RDKIT_AVAILABLE = False
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, Lipinski
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit import DataStructs

    RDKIT_AVAILABLE = True
    logger.info("RDKit available - full functionality enabled")
except ImportError:
    logger.warning(
        "RDKit not available - some functionality will be limited. "
        "Install with: pip install rdkit"
    )


def check_rdkit() -> bool:
    """Check if RDKit is available."""
    return RDKIT_AVAILABLE


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """
    Canonicalize SMILES string.

    Returns None if invalid or RDKit unavailable.
    """
    if not RDKIT_AVAILABLE:
        return smiles  # Return as-is if RDKit unavailable

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def is_valid_smiles(smiles: str) -> bool:
    """Check if SMILES is valid."""
    if not RDKIT_AVAILABLE:
        # Without RDKit, do basic check
        return bool(smiles and len(smiles) > 0 and not smiles.isspace())

    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except Exception:
        return False


def get_murcko_scaffold(smiles: str) -> Optional[str]:
    """Get Murcko scaffold for a molecule."""
    if not RDKIT_AVAILABLE:
        return None

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)
    except Exception:
        return None


def compute_ecfp4_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048):
    """Compute ECFP4 fingerprint."""
    if not RDKIT_AVAILABLE:
        return None

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    except Exception:
        return None


def compute_tanimoto_similarity(smiles1: str, smiles2: str) -> Optional[float]:
    """Compute Tanimoto similarity between two molecules."""
    if not RDKIT_AVAILABLE:
        return None

    fp1 = compute_ecfp4_fingerprint(smiles1)
    fp2 = compute_ecfp4_fingerprint(smiles2)

    if fp1 is None or fp2 is None:
        return None

    return DataStructs.TanimotoSimilarity(fp1, fp2)


def compute_molecular_properties(smiles: str) -> dict:
    """
    Compute molecular properties.

    Returns dict with MW, logP, TPSA, HBD, HBA, rotatable bonds.
    If RDKit unavailable, returns empty dict.
    """
    if not RDKIT_AVAILABLE:
        return {}

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}

        return {
            "mw": Descriptors.MolWt(mol),
            "logp": Descriptors.MolLogP(mol),
            "tpsa": Descriptors.TPSA(mol),
            "hbd": Lipinski.NumHDonors(mol),
            "hba": Lipinski.NumHAcceptors(mol),
            "rotatable_bonds": Lipinski.NumRotatableBonds(mol),
        }
    except Exception:
        return {}


def check_lipinski_rule_of_five(properties: dict) -> bool:
    """
    Check if molecule satisfies Lipinski's Rule of Five.

    Requires properties dict from compute_molecular_properties.
    """
    if not properties:
        return True  # Default to pass if we can't check

    violations = 0
    if properties.get("mw", 0) > 500:
        violations += 1
    if properties.get("logp", 0) > 5:
        violations += 1
    if properties.get("hbd", 0) > 5:
        violations += 1
    if properties.get("hba", 0) > 10:
        violations += 1

    return violations <= 1


def check_property_constraints(properties: dict, constraints: dict) -> bool:
    """Check if properties satisfy given constraints."""
    if not properties:
        return True  # Default to pass if we can't check

    checks = [
        constraints.get("mw_min", 0) <= properties.get("mw", 300) <= constraints.get("mw_max", 600),
        constraints.get("logp_min", -5)
        <= properties.get("logp", 2)
        <= constraints.get("logp_max", 5),
        properties.get("tpsa", 70) <= constraints.get("tpsa_max", 140),
        properties.get("hbd", 2) <= constraints.get("hbd_max", 5),
        properties.get("hba", 5) <= constraints.get("hba_max", 10),
        properties.get("rotatable_bonds", 5) <= constraints.get("rotatable_bonds_max", 10),
    ]

    return all(checks)


def batch_compute_fingerprints(smiles_list: List[str]) -> List[Optional[object]]:
    """Compute fingerprints for a batch of SMILES."""
    return [compute_ecfp4_fingerprint(s) for s in smiles_list]


def compute_internal_diversity(smiles_list: List[str], sample_size: int = 1000) -> float:
    """
    Compute average pairwise Tanimoto distance (1 - similarity).

    For large sets, samples randomly.
    Returns 0.0 if RDKit unavailable or computation fails.
    """
    if not RDKIT_AVAILABLE or len(smiles_list) < 2:
        return 0.0

    import random

    # Sample if too large
    if len(smiles_list) > sample_size:
        smiles_list = random.sample(smiles_list, sample_size)

    fps = [compute_ecfp4_fingerprint(s) for s in smiles_list]
    fps = [fp for fp in fps if fp is not None]

    if len(fps) < 2:
        return 0.0

    similarities = []
    n = len(fps)
    max_comparisons = min(n * (n - 1) // 2, 10000)  # Limit comparisons
    comparison_count = 0

    for i in range(n):
        for j in range(i + 1, n):
            if comparison_count >= max_comparisons:
                break
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            similarities.append(sim)
            comparison_count += 1
        if comparison_count >= max_comparisons:
            break

    if not similarities:
        return 0.0

    avg_similarity = sum(similarities) / len(similarities)
    return 1.0 - avg_similarity  # Return diversity (distance)
