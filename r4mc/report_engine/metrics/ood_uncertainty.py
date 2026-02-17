"""OOD and uncertainty metrics over time."""
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def compute_ood_metrics(run_data) -> pd.DataFrame:
    """
    Compute OOD (Out-of-Distribution) metrics per episode.

    Args:
        run_data: RunData object with molecules

    Returns:
        DataFrame with OOD metrics per episode
    """
    from ..io.model import RunData

    if not isinstance(run_data, RunData):
        raise ValueError("run_data must be a RunData object")

    results = []

    for ep_mols in run_data.molecules:
        if ep_mols.ood_flags is not None and len(ep_mols.ood_flags) > 0:
            n_total = len(ep_mols.ood_flags)
            n_ood = sum(ep_mols.ood_flags)
            ood_rate = n_ood / n_total if n_total > 0 else 0.0
        else:
            n_total = len(ep_mols.smiles)
            n_ood = None
            ood_rate = None

        results.append({
            'episode': ep_mols.episode,
            'arm': ep_mols.arm,
            'n_total': n_total,
            'n_ood': n_ood,
            'ood_rate': ood_rate
        })

    return pd.DataFrame(results)


def compute_uncertainty_metrics(run_data) -> pd.DataFrame:
    """
    Compute uncertainty metrics per episode.

    Args:
        run_data: RunData object with molecules

    Returns:
        DataFrame with uncertainty metrics per episode
    """
    from ..io.model import RunData

    if not isinstance(run_data, RunData):
        raise ValueError("run_data must be a RunData object")

    results = []

    for ep_mols in run_data.molecules:
        if ep_mols.uncertainty is not None and len(ep_mols.uncertainty) > 0:
            uncertainty_mean = sum(ep_mols.uncertainty) / len(ep_mols.uncertainty)
            uncertainty_max = max(ep_mols.uncertainty)
            uncertainty_min = min(ep_mols.uncertainty)
            uncertainty_std = (
                sum((x - uncertainty_mean) ** 2 for x in ep_mols.uncertainty) / len(ep_mols.uncertainty)
            ) ** 0.5
        else:
            uncertainty_mean = None
            uncertainty_max = None
            uncertainty_min = None
            uncertainty_std = None

        results.append({
            'episode': ep_mols.episode,
            'arm': ep_mols.arm,
            'uncertainty_mean': uncertainty_mean,
            'uncertainty_max': uncertainty_max,
            'uncertainty_min': uncertainty_min,
            'uncertainty_std': uncertainty_std
        })

    return pd.DataFrame(results)


def detect_regime_changes(ood_df: pd.DataFrame, threshold: float = 0.2) -> pd.DataFrame:
    """
    Detect regime changes based on OOD rate shifts.

    Args:
        ood_df: DataFrame from compute_ood_metrics
        threshold: Minimum change in OOD rate to consider a regime change

    Returns:
        DataFrame with detected regime changes
    """
    if len(ood_df) < 2 or 'ood_rate' not in ood_df.columns:
        return pd.DataFrame(columns=['episode', 'ood_rate_change', 'regime_change'])

    ood_df = ood_df.sort_values('episode')
    regime_changes = []

    for i in range(1, len(ood_df)):
        prev_rate = ood_df.iloc[i - 1]['ood_rate']
        curr_rate = ood_df.iloc[i]['ood_rate']

        if prev_rate is not None and curr_rate is not None:
            change = abs(curr_rate - prev_rate)
            is_regime_change = change >= threshold

            regime_changes.append({
                'episode': ood_df.iloc[i]['episode'],
                'ood_rate_change': change,
                'regime_change': is_regime_change
            })

    return pd.DataFrame(regime_changes)
