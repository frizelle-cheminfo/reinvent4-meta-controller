"""Nearest neighbor search for MolPort purchasability mapping."""
import pandas as pd
import logging
from typing import List, Dict, Optional
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from .index import MolPortIndex

logger = logging.getLogger(__name__)


def find_nearest_neighbors(
    smiles_list: List[str],
    molport_index: MolPortIndex,
    top_k: int = 1,
    min_similarity: float = 0.0
) -> pd.DataFrame:
    """
    Find nearest MolPort neighbors for query molecules.

    Args:
        smiles_list: List of query SMILES
        molport_index: MolPort fingerprint index
        top_k: Number of nearest neighbors to return per query
        min_similarity: Minimum Tanimoto similarity threshold

    Returns:
        DataFrame with nearest neighbor mappings
    """
    results = []

    for query_smiles in smiles_list:
        try:
            # Generate fingerprint for query
            mol = Chem.MolFromSmiles(query_smiles)
            if mol is None:
                logger.debug(f"Invalid SMILES: {query_smiles}")
                continue

            query_fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, molport_index.fp_radius, nBits=molport_index.fp_nbits
            )

            # Compute similarities to all MolPort molecules
            similarities = []
            for i, molport_fp in enumerate(molport_index.fingerprints):
                sim = DataStructs.TanimotoSimilarity(query_fp, molport_fp)
                if sim >= min_similarity:
                    similarities.append((i, sim))

            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)

            # Get top-k
            for i, (idx, sim) in enumerate(similarities[:top_k]):
                results.append({
                    'query_smiles': query_smiles,
                    'molport_smiles': molport_index.smiles[idx],
                    'molport_id': molport_index.ids[idx],
                    'tanimoto_similarity': sim,
                    'rank': i + 1
                })

        except Exception as e:
            logger.debug(f"Failed to find neighbors for {query_smiles}: {e}")
            continue

    return pd.DataFrame(results)


def map_run_to_molport(
    run_data,
    molport_index: MolPortIndex,
    top_k: int = 1,
    min_similarity: float = 0.7
) -> pd.DataFrame:
    """
    Map all molecules from a run to MolPort purchasable chemistry.

    Args:
        run_data: RunData object
        molport_index: MolPort fingerprint index
        top_k: Number of nearest neighbors per molecule
        min_similarity: Minimum Tanimoto similarity threshold

    Returns:
        DataFrame with purchasability mappings
    """
    from ..io.model import RunData

    if not isinstance(run_data, RunData):
        raise ValueError("run_data must be a RunData object")

    # Collect all unique SMILES
    all_smiles = set()
    for ep_mols in run_data.molecules:
        all_smiles.update(ep_mols.smiles)

    logger.info(f"Mapping {len(all_smiles)} unique molecules to MolPort...")

    # Find nearest neighbors
    mapping_df = find_nearest_neighbors(
        list(all_smiles),
        molport_index,
        top_k=top_k,
        min_similarity=min_similarity
    )

    logger.info(f"Found {len(mapping_df)} MolPort mappings (min similarity: {min_similarity})")

    return mapping_df


def compute_purchasability_metrics(run_data, molport_mapping: pd.DataFrame) -> pd.DataFrame:
    """
    Compute purchasability metrics per episode.

    Args:
        run_data: RunData object
        molport_mapping: DataFrame from map_run_to_molport

    Returns:
        DataFrame with purchasability metrics per episode
    """
    from ..io.model import RunData

    if not isinstance(run_data, RunData):
        raise ValueError("run_data must be a RunData object")

    # Build mapping dict
    purchasable_smiles = set(molport_mapping['query_smiles'].unique())

    results = []

    for ep_mols in run_data.molecules:
        ep_smiles = set(ep_mols.smiles)

        # Count purchasable molecules
        n_total = len(ep_smiles)
        n_purchasable = len(ep_smiles & purchasable_smiles)
        purchasability_rate = n_purchasable / n_total if n_total > 0 else 0.0

        # Get average similarity for purchasable molecules
        ep_mapping = molport_mapping[molport_mapping['query_smiles'].isin(ep_smiles)]
        avg_similarity = ep_mapping['tanimoto_similarity'].mean() if len(ep_mapping) > 0 else None

        results.append({
            'episode': ep_mols.episode,
            'arm': ep_mols.arm,
            'n_total': n_total,
            'n_purchasable': n_purchasable,
            'purchasability_rate': purchasability_rate,
            'avg_similarity': avg_similarity
        })

    return pd.DataFrame(results)
