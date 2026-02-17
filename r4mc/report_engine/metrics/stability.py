"""Top-N stability metrics for tracking molecule set overlap."""
import pandas as pd
from typing import List, Set
import logging

logger = logging.getLogger(__name__)


def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """
    Calculate Jaccard similarity between two sets.

    Args:
        set1: First set
        set2: Second set

    Returns:
        Jaccard similarity (0-1)
    """
    if len(set1) == 0 and len(set2) == 0:
        return 1.0

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    if union == 0:
        return 0.0

    return intersection / union


def compute_top_n_stability(run_data, top_n: int = 50) -> pd.DataFrame:
    """
    Compute Top-N stability across episodes using Jaccard overlap.

    Args:
        run_data: RunData object with molecules
        top_n: Number of top molecules to consider

    Returns:
        DataFrame with episode pairs and Jaccard similarity scores
    """
    from ..io.model import RunData

    if not isinstance(run_data, RunData):
        raise ValueError("run_data must be a RunData object")

    results = []

    # Sort molecules by episode
    sorted_mols = sorted(run_data.molecules, key=lambda x: x.episode)

    for i in range(len(sorted_mols) - 1):
        ep1 = sorted_mols[i]
        ep2 = sorted_mols[i + 1]

        # Get top-N SMILES from each episode (by score)
        if len(ep1.scores) == 0 or len(ep2.scores) == 0:
            continue

        # Sort by score (descending) and take top-N
        ep1_df = pd.DataFrame({
            'smiles': ep1.smiles,
            'score': ep1.scores
        })
        ep2_df = pd.DataFrame({
            'smiles': ep2.smiles,
            'score': ep2.scores
        })

        ep1_top = set(ep1_df.nlargest(min(top_n, len(ep1_df)), 'score')['smiles'])
        ep2_top = set(ep2_df.nlargest(min(top_n, len(ep2_df)), 'score')['smiles'])

        # Calculate Jaccard similarity
        jaccard = jaccard_similarity(ep1_top, ep2_top)

        results.append({
            'episode_1': ep1.episode,
            'episode_2': ep2.episode,
            'arm_1': ep1.arm,
            'arm_2': ep2.arm,
            'top_n': top_n,
            'jaccard_similarity': jaccard,
            'n_overlap': len(ep1_top & ep2_top),
            'n_union': len(ep1_top | ep2_top)
        })

    return pd.DataFrame(results)


def compute_stability_summary(stability_df: pd.DataFrame) -> dict:
    """
    Compute summary statistics for stability metrics.

    Args:
        stability_df: DataFrame from compute_top_n_stability

    Returns:
        Dict with summary statistics
    """
    if len(stability_df) == 0:
        return {
            'mean_jaccard': 0.0,
            'median_jaccard': 0.0,
            'min_jaccard': 0.0,
            'max_jaccard': 0.0,
            'std_jaccard': 0.0
        }

    return {
        'mean_jaccard': stability_df['jaccard_similarity'].mean(),
        'median_jaccard': stability_df['jaccard_similarity'].median(),
        'min_jaccard': stability_df['jaccard_similarity'].min(),
        'max_jaccard': stability_df['jaccard_similarity'].max(),
        'std_jaccard': stability_df['jaccard_similarity'].std()
    }
