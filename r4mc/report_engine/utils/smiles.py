"""SMILES standardization utilities."""
import logging
from typing import Optional
from rdkit import Chem

logger = logging.getLogger(__name__)


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """
    Canonicalize a SMILES string using RDKit.

    Args:
        smiles: Input SMILES string

    Returns:
        Canonical SMILES or None if invalid
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception as e:
        logger.debug(f"Failed to canonicalize SMILES '{smiles}': {e}")
        return None


def is_valid_smiles(smiles: str) -> bool:
    """
    Check if a SMILES string is valid.

    Args:
        smiles: Input SMILES string

    Returns:
        True if valid, False otherwise
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False


def smiles_to_mol(smiles: str) -> Optional[Chem.Mol]:
    """
    Convert SMILES to RDKit Mol object.

    Args:
        smiles: Input SMILES string

    Returns:
        RDKit Mol object or None if invalid
    """
    try:
        return Chem.MolFromSmiles(smiles)
    except:
        return None
