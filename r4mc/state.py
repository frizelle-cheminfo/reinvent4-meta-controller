"""State management for the meta-controller."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class MoleculeRecord(BaseModel):
    """Record of a generated molecule."""

    smiles: str
    score: float
    episode: int
    arm_id: str
    seed_id: Optional[str] = None
    properties: Dict[str, Any] = Field(default_factory=dict)


class SeedRecord(BaseModel):
    """Record in the seed bank."""

    seed_id: str
    smiles: str
    score: float
    scaffold: Optional[str] = None
    added_episode: int
    times_used: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EpisodeMetrics(BaseModel):
    """Metrics computed for an episode."""

    valid_pct: float = 0.0
    unique_pct: float = 0.0
    internal_diversity: float = 0.0
    scaffold_diversity: float = 0.0
    rediscovery_rate: float = 0.0
    novelty: float = 0.0
    ood_rate: float = 0.0
    uncertainty_mean: float = 0.0
    topk_gain: float = 0.0
    hit_yield: float = 0.0
    property_pass_rate: float = 0.0
    best_score: float = 0.0
    best_smiles: str = ""


class EpisodeRecord(BaseModel):
    """Record of a completed episode."""

    episode_num: int
    arm_id: str
    prior_filename: str
    regime: str
    reason: List[str] = Field(default_factory=list)
    metrics: EpisodeMetrics
    output_dir: Path
    checkpoint_path: Optional[Path] = None
    success: bool = True
    error_message: Optional[str] = None


class ArmStatistics(BaseModel):
    """Statistics for an arm (for bandit selection)."""

    arm_id: str
    episodes_run: int = 0
    total_quality: float = 0.0
    mean_quality: float = 0.0
    successes: int = 0
    failures: int = 0
    blacklisted: bool = False
    blacklist_reason: Optional[str] = None


class ControllerState(BaseModel):
    """Persistent state of the controller."""

    campaign_name: str
    seed: int
    current_episode: int = 0
    episodes_history: List[EpisodeRecord] = Field(default_factory=list)
    arm_stats: Dict[str, ArmStatistics] = Field(default_factory=dict)
    seed_bank: List[SeedRecord] = Field(default_factory=list)
    best_molecules: List[MoleculeRecord] = Field(default_factory=list)
    best_score: float = 0.0
    all_generated_smiles: List[str] = Field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> "ControllerState":
        """Load state from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    def save(self, path: Path) -> None:
        """Save state to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.model_dump(mode="python"), f, indent=2, default=str)

    def get_arm_stats(self, arm_id: str) -> ArmStatistics:
        """Get or create statistics for an arm."""
        if arm_id not in self.arm_stats:
            self.arm_stats[arm_id] = ArmStatistics(arm_id=arm_id)
        return self.arm_stats[arm_id]

    def update_arm_stats(self, arm_id: str, quality: float, success: bool) -> None:
        """Update statistics after an episode."""
        stats = self.get_arm_stats(arm_id)
        stats.episodes_run += 1
        if success:
            stats.successes += 1
            stats.total_quality += quality
            stats.mean_quality = stats.total_quality / stats.successes
        else:
            stats.failures += 1

        # Blacklist if too many consecutive failures
        if stats.failures >= 3 and stats.successes == 0:
            stats.blacklisted = True
            stats.blacklist_reason = "Too many consecutive failures"

    def add_to_seed_bank(self, seeds: List[SeedRecord], max_size: int = 100) -> None:
        """Add seeds to the bank, maintaining max size."""
        self.seed_bank.extend(seeds)
        # Sort by score and keep top N
        self.seed_bank.sort(key=lambda s: s.score, reverse=True)
        self.seed_bank = self.seed_bank[:max_size]

    def record_episode(self, record: EpisodeRecord) -> None:
        """Record a completed episode."""
        self.episodes_history.append(record)
        self.current_episode += 1
