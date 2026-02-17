"""Diversity and scaffold analysis metrics."""
import pandas as pd
from typing import List, Optional
import logging
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

logger = logging.getLogger(__name__)


def get_bemis_murcko_scaffold(smiles: str) -> Optional[str]:
    """
    Extract Bemis-Murcko scaffold from a SMILES string.

    Args:
        smiles: Input SMILES string

    Returns:
        Scaffold SMILES or None if invalid
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)
    except Exception as e:
        logger.debug(f"Failed to extract scaffold from '{smiles}': {e}")
        return None


def compute_diversity_metrics(run_data) -> pd.DataFrame:
    """
    Compute diversity metrics per episode.

    Args:
        run_data: RunData object with molecules

    Returns:
        DataFrame with diversity metrics per episode
    """
    from ..io.model import RunData

    if not isinstance(run_data, RunData):
        raise ValueError("run_data must be a RunData object")

    results = []

    for ep_mols in run_data.molecules:
        # Unique SMILES
        unique_smiles = set(ep_mols.smiles)

        # Extract scaffolds
        scaffolds = []
        for smi in ep_mols.smiles:
            scaffold = get_bemis_murcko_scaffold(smi)
            if scaffold:
                scaffolds.append(scaffold)

        unique_scaffolds = set(scaffolds)

        # Calculate metrics
        n_molecules = len(ep_mols.smiles)
        n_unique = len(unique_smiles)
        n_scaffolds = len(scaffolds)
        n_unique_scaffolds = len(unique_scaffolds)

        # Diversity ratios
        uniqueness_ratio = n_unique / n_molecules if n_molecules > 0 else 0.0
        scaffold_diversity = n_unique_scaffolds / n_scaffolds if n_scaffolds > 0 else 0.0

        results.append({
            'episode': ep_mols.episode,
            'arm': ep_mols.arm,
            'n_molecules': n_molecules,
            'n_unique': n_unique,
            'n_scaffolds': n_scaffolds,
            'n_unique_scaffolds': n_unique_scaffolds,
            'uniqueness_ratio': uniqueness_ratio,
            'scaffold_diversity': scaffold_diversity
        })

    return pd.DataFrame(results)


def compute_rediscovery_metrics(run_data) -> pd.DataFrame:
    """
    Track rediscovery of molecules across episodes.

    Args:
        run_data: RunData object with molecules

    Returns:
        DataFrame with rediscovery metrics
    """
    from ..io.model import RunData

    if not isinstance(run_data, RunData):
        raise ValueError("run_data must be a RunData object")

    results = []
    seen_smiles = set()

    # Sort by episode
    sorted_mols = sorted(run_data.molecules, key=lambda x: x.episode)

    for ep_mols in sorted_mols:
        ep_smiles = set(ep_mols.smiles)

        # Count rediscovered molecules
        rediscovered = ep_smiles & seen_smiles
        novel = ep_smiles - seen_smiles

        n_total = len(ep_smiles)
        n_rediscovered = len(rediscovered)
        n_novel = len(novel)

        rediscovery_rate = n_rediscovered / n_total if n_total > 0 else 0.0
        novelty_rate = n_novel / n_total if n_total > 0 else 0.0

        results.append({
            'episode': ep_mols.episode,
            'arm': ep_mols.arm,
            'n_total': n_total,
            'n_rediscovered': n_rediscovered,
            'n_novel': n_novel,
            'rediscovery_rate': rediscovery_rate,
            'novelty_rate': novelty_rate,
            'cumulative_unique': len(seen_smiles | ep_smiles)
        })

        # Update seen set
        seen_smiles.update(ep_smiles)

    return pd.DataFrame(results)
