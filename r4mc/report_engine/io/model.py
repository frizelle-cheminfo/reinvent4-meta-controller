"""Data models for REINVENT meta-controller runs."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import pandas as pd


@dataclass
class EpisodeMolecules:
    """Molecules generated in an episode."""
    episode: int
    arm: str
    smiles: List[str]
    scores: List[float]
    uncertainty: Optional[List[float]] = None
    novelty: Optional[List[float]] = None
    ood_flags: Optional[List[bool]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        data = {
            'episode': self.episode,
            'arm': self.arm,
            'smiles': self.smiles,
            'score': self.scores,
        }
        if self.uncertainty:
            data['uncertainty'] = self.uncertainty
        if self.novelty:
            data['novelty'] = self.novelty
        if self.ood_flags:
            data['ood'] = self.ood_flags
        return pd.DataFrame(data)


@dataclass
class EpisodeMetrics:
    """Aggregated metrics for an episode."""
    episode: int
    arm: str
    ood_rate: Optional[float] = None
    uncertainty_mean: Optional[float] = None
    diversity: Optional[float] = None
    novelty_mean: Optional[float] = None
    reward: Optional[float] = None
    best_score: Optional[float] = None
    n_molecules: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ControllerEvent:
    """A controller decision event."""
    episode: int
    arm_chosen: str
    reasons: List[str]
    ucb_scores: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SeedBankSnapshot:
    """Seed bank state at an episode."""
    episode: int
    seeds: List[str]
    scores: Optional[List[float]] = None
    source_episodes: Optional[List[int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunData:
    """Complete data from a meta-controller run."""
    run_name: str
    run_dir: str
    molecules: List[EpisodeMolecules] = field(default_factory=list)
    metrics: List[EpisodeMetrics] = field(default_factory=list)
    events: List[ControllerEvent] = field(default_factory=list)
    seed_banks: List[SeedBankSnapshot] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def get_molecules_df(self) -> pd.DataFrame:
        """Get all molecules as a DataFrame."""
        if not self.molecules:
            return pd.DataFrame()
        return pd.concat([m.to_dataframe() for m in self.molecules], ignore_index=True)

    def get_metrics_df(self) -> pd.DataFrame:
        """Get all metrics as a DataFrame."""
        if not self.metrics:
            return pd.DataFrame()
        return pd.DataFrame([vars(m) for m in self.metrics])

    def get_events_df(self) -> pd.DataFrame:
        """Get all events as a DataFrame."""
        if not self.events:
            return pd.DataFrame()
        data = []
        for e in self.events:
            row = {
                'episode': e.episode,
                'arm_chosen': e.arm_chosen,
                'reasons': ','.join(e.reasons)
            }
            if e.ucb_scores:
                row.update({f'ucb_{k}': v for k, v in e.ucb_scores.items()})
            data.append(row)
        return pd.DataFrame(data)
