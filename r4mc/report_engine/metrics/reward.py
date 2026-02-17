"""Reward decomposition and analysis metrics."""
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def compute_reward_decomposition(run_data) -> pd.DataFrame:
    """
    Decompose reward components per episode.

    Args:
        run_data: RunData object with molecules

    Returns:
        DataFrame with reward decomposition per episode
    """
    from ..io.model import RunData

    if not isinstance(run_data, RunData):
        raise ValueError("run_data must be a RunData object")

    results = []

    for ep_mols in run_data.molecules:
        # Base score statistics
        if len(ep_mols.scores) > 0:
            score_mean = sum(ep_mols.scores) / len(ep_mols.scores)
            score_max = max(ep_mols.scores)
            score_min = min(ep_mols.scores)
            score_std = (
                sum((x - score_mean) ** 2 for x in ep_mols.scores) / len(ep_mols.scores)
            ) ** 0.5
        else:
            score_mean = None
            score_max = None
            score_min = None
            score_std = None

        # Novelty statistics (if available)
        if ep_mols.novelty is not None and len(ep_mols.novelty) > 0:
            novelty_mean = sum(ep_mols.novelty) / len(ep_mols.novelty)
            novelty_max = max(ep_mols.novelty)
            novelty_min = min(ep_mols.novelty)
        else:
            novelty_mean = None
            novelty_max = None
            novelty_min = None

        results.append({
            'episode': ep_mols.episode,
            'arm': ep_mols.arm,
            'score_mean': score_mean,
            'score_max': score_max,
            'score_min': score_min,
            'score_std': score_std,
            'novelty_mean': novelty_mean,
            'novelty_max': novelty_max,
            'novelty_min': novelty_min
        })

    return pd.DataFrame(results)


def compute_novelty_score_frontier(run_data) -> pd.DataFrame:
    """
    Compute novelty vs score frontier per episode.

    Args:
        run_data: RunData object with molecules

    Returns:
        DataFrame with novelty-score pairs per episode
    """
    from ..io.model import RunData

    if not isinstance(run_data, RunData):
        raise ValueError("run_data must be a RunData object")

    results = []

    for ep_mols in run_data.molecules:
        if ep_mols.novelty is not None and len(ep_mols.novelty) > 0:
            for i, (score, novelty) in enumerate(zip(ep_mols.scores, ep_mols.novelty)):
                results.append({
                    'episode': ep_mols.episode,
                    'arm': ep_mols.arm,
                    'smiles': ep_mols.smiles[i] if i < len(ep_mols.smiles) else None,
                    'score': score,
                    'novelty': novelty
                })

    return pd.DataFrame(results)


def compute_arm_performance(run_data) -> pd.DataFrame:
    """
    Compute performance metrics per arm.

    Args:
        run_data: RunData object with molecules

    Returns:
        DataFrame with performance metrics per arm
    """
    from ..io.model import RunData

    if not isinstance(run_data, RunData):
        raise ValueError("run_data must be a RunData object")

    arm_stats = {}

    for ep_mols in run_data.molecules:
        arm = ep_mols.arm

        if arm not in arm_stats:
            arm_stats[arm] = {
                'episodes': [],
                'scores': [],
                'novelties': []
            }

        arm_stats[arm]['episodes'].append(ep_mols.episode)
        arm_stats[arm]['scores'].extend(ep_mols.scores)

        if ep_mols.novelty is not None:
            arm_stats[arm]['novelties'].extend(ep_mols.novelty)

    results = []

    for arm, stats in arm_stats.items():
        scores = stats['scores']
        novelties = stats['novelties']

        results.append({
            'arm': arm,
            'n_episodes': len(stats['episodes']),
            'n_molecules': len(scores),
            'score_mean': sum(scores) / len(scores) if scores else 0.0,
            'score_max': max(scores) if scores else 0.0,
            'novelty_mean': sum(novelties) / len(novelties) if novelties else None
        })

    return pd.DataFrame(results)
