"""Configuration models for the meta-controller."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field


class ScoringProfile(BaseModel):
    """Default scoring profile parameters."""

    potency_weight: float = 0.5
    novelty_weight: float = 0.2
    similarity_weight: float = 0.1
    uncertainty_penalty: float = 0.1
    diversity_weight: float = 0.1


class PropertyConstraints(BaseModel):
    """Molecular property constraints."""

    mw_min: float = 200.0
    mw_max: float = 600.0
    logp_min: float = -2.0
    logp_max: float = 5.0
    tpsa_max: float = 140.0
    hbd_max: int = 5
    hba_max: int = 10
    rotatable_bonds_max: int = 10


class DiversitySettings(BaseModel):
    """Diversity filter settings."""

    bucket_similarity: float = 0.6
    penalty_strength: float = 0.5
    use_scaffold_memory: bool = True


class ThresholdConfig(BaseModel):
    """Thresholds for controller decision logic."""

    ood_high: float = 0.3
    uncertainty_spike: float = 0.4
    diversity_collapse: float = 0.1
    rediscovery_high: float = 0.4
    property_pass_low: float = 0.3
    stagnation_episodes: int = 3
    hit_threshold: float = 0.6


class EpisodeConfig(BaseModel):
    """Configuration for episode execution."""

    steps: int = 500
    batch_size: int = 100
    learning_rate: float = 0.0001
    sigma: float = 80
    randomize_smiles: bool = True


class ControllerConfig(BaseModel):
    """Main controller configuration."""

    campaign_name: str
    seed: int = 42
    priors_dir: Path = Path("./priors")
    reinvent_bin: Optional[Path] = None
    output_dir: Path = Path("./runs")

    # Episode settings
    episode: EpisodeConfig = Field(default_factory=EpisodeConfig)

    # Scoring and constraints
    scoring: ScoringProfile = Field(default_factory=ScoringProfile)
    properties: PropertyConstraints = Field(default_factory=PropertyConstraints)
    diversity: DiversitySettings = Field(default_factory=DiversitySettings)
    thresholds: ThresholdConfig = Field(default_factory=ThresholdConfig)

    # Bandit parameters
    bandit_ucb_c: float = 1.5
    bandit_thompson_alpha: float = 1.0
    bandit_thompson_beta: float = 1.0

    # Seed bank
    seed_bank_size: int = 100
    support_set_file: Optional[Path] = None

    # Output parsing
    output_csv_name: str = "results.csv"
    output_csv_smiles_col: str = "SMILES"
    output_csv_score_col: str = "Score"

    @classmethod
    def load(cls, path: Path) -> "ControllerConfig":
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def save(self, path: Path) -> None:
        """Save config to YAML file."""
        # Convert Path objects to strings for YAML serialization
        data = self.model_dump(mode="python")
        data['priors_dir'] = str(data['priors_dir'])
        data['output_dir'] = str(data['output_dir'])
        if data.get('reinvent_bin'):
            data['reinvent_bin'] = str(data['reinvent_bin'])
        if data.get('support_set_file'):
            data['support_set_file'] = str(data['support_set_file'])

        with open(path, "w") as f:
            yaml.safe_dump(data, f, default_flow_style=False)


class ArmConfig(BaseModel):
    """Configuration for a single arm (prior + regime + settings)."""

    arm_id: str
    prior_filename: str
    generator_type: str  # reinvent, mol2mol, libinvent, linkinvent
    regime: str  # explore, exploit
    requires_seeds: bool
    scoring_profile: ScoringProfile = Field(default_factory=ScoringProfile)
    diversity_settings: DiversitySettings = Field(default_factory=DiversitySettings)
    tags: List[str] = Field(default_factory=list)
    template_name: str = "reinvent.toml.j2"

    # Override episode settings if needed
    episode_steps: Optional[int] = None
    episode_batch_size: Optional[int] = None
