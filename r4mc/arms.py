"""Arm catalog and management."""

import logging
from pathlib import Path
from typing import Dict, List

from .config import ArmConfig, DiversitySettings, ScoringProfile

logger = logging.getLogger(__name__)


def get_arm_catalog() -> List[ArmConfig]:
    """
    Define the catalog of all available arms.

    Each arm specifies a prior, generator type, regime, and default settings.
    """
    catalog = [
        # De novo REINVENT arms
        ArmConfig(
            arm_id="reinvent_explore",
            prior_filename="reinvent.prior",
            generator_type="reinvent",
            regime="explore",
            requires_seeds=False,
            scoring_profile=ScoringProfile(
                potency_weight=0.3,
                novelty_weight=0.4,
                similarity_weight=0.0,
                uncertainty_penalty=0.1,
                diversity_weight=0.2,
            ),
            diversity_settings=DiversitySettings(
                bucket_similarity=0.55, penalty_strength=0.6, use_scaffold_memory=True
            ),
            tags=["explore", "de-novo", "global"],
            template_name="reinvent.toml.j2",
        ),
        ArmConfig(
            arm_id="reinvent_exploit",
            prior_filename="reinvent.prior",
            generator_type="reinvent",
            regime="exploit",
            requires_seeds=False,
            scoring_profile=ScoringProfile(
                potency_weight=0.6,
                novelty_weight=0.1,
                similarity_weight=0.0,
                uncertainty_penalty=0.2,
                diversity_weight=0.1,
            ),
            diversity_settings=DiversitySettings(
                bucket_similarity=0.65, penalty_strength=0.4, use_scaffold_memory=False
            ),
            tags=["exploit", "de-novo"],
            template_name="reinvent.toml.j2",
        ),
        # Mol2Mol high similarity (small changes)
        ArmConfig(
            arm_id="mol2mol_high_sim_exploit",
            prior_filename="mol2mol_high_similarity.prior",
            generator_type="mol2mol",
            regime="exploit",
            requires_seeds=True,
            scoring_profile=ScoringProfile(
                potency_weight=0.7,
                novelty_weight=0.0,
                similarity_weight=0.2,
                uncertainty_penalty=0.1,
                diversity_weight=0.0,
            ),
            diversity_settings=DiversitySettings(
                bucket_similarity=0.7, penalty_strength=0.3, use_scaffold_memory=False
            ),
            tags=["exploit", "mol2mol", "high-similarity", "local-search"],
            template_name="mol2mol.toml.j2",
        ),
        # Mol2Mol medium similarity
        ArmConfig(
            arm_id="mol2mol_medium_sim_exploit",
            prior_filename="mol2mol_medium_similarity.prior",
            generator_type="mol2mol",
            regime="exploit",
            requires_seeds=True,
            scoring_profile=ScoringProfile(
                potency_weight=0.6,
                novelty_weight=0.1,
                similarity_weight=0.2,
                uncertainty_penalty=0.1,
                diversity_weight=0.0,
            ),
            diversity_settings=DiversitySettings(
                bucket_similarity=0.65, penalty_strength=0.4, use_scaffold_memory=False
            ),
            tags=["exploit", "mol2mol", "medium-similarity"],
            template_name="mol2mol.toml.j2",
        ),
        # Mol2Mol general similarity (explore)
        ArmConfig(
            arm_id="mol2mol_sim_explore",
            prior_filename="mol2mol_similarity.prior",
            generator_type="mol2mol",
            regime="explore",
            requires_seeds=True,
            scoring_profile=ScoringProfile(
                potency_weight=0.4,
                novelty_weight=0.3,
                similarity_weight=0.1,
                uncertainty_penalty=0.1,
                diversity_weight=0.1,
            ),
            diversity_settings=DiversitySettings(
                bucket_similarity=0.6, penalty_strength=0.5, use_scaffold_memory=True
            ),
            tags=["explore", "mol2mol", "similarity-guided"],
            template_name="mol2mol.toml.j2",
        ),
        # Mol2Mol scaffold (scaffold hopping)
        ArmConfig(
            arm_id="mol2mol_scaffold_explore",
            prior_filename="mol2mol_scaffold.prior",
            generator_type="mol2mol",
            regime="explore",
            requires_seeds=True,
            scoring_profile=ScoringProfile(
                potency_weight=0.3,
                novelty_weight=0.4,
                similarity_weight=0.0,
                uncertainty_penalty=0.2,
                diversity_weight=0.1,
            ),
            diversity_settings=DiversitySettings(
                bucket_similarity=0.5, penalty_strength=0.6, use_scaffold_memory=True
            ),
            tags=["explore", "mol2mol", "scaffold-hop"],
            template_name="mol2mol.toml.j2",
        ),
        # Mol2Mol scaffold generic
        ArmConfig(
            arm_id="mol2mol_scaffold_generic_explore",
            prior_filename="mol2mol_scaffold_generic.prior",
            generator_type="mol2mol",
            regime="explore",
            requires_seeds=True,
            scoring_profile=ScoringProfile(
                potency_weight=0.3,
                novelty_weight=0.4,
                similarity_weight=0.0,
                uncertainty_penalty=0.2,
                diversity_weight=0.1,
            ),
            diversity_settings=DiversitySettings(
                bucket_similarity=0.5, penalty_strength=0.6, use_scaffold_memory=True
            ),
            tags=["explore", "mol2mol", "scaffold-hop", "generic"],
            template_name="mol2mol.toml.j2",
        ),
        # Mol2Mol MMP (matched molecular pairs)
        ArmConfig(
            arm_id="mol2mol_mmp_exploit",
            prior_filename="mol2mol_mmp.prior",
            generator_type="mol2mol",
            regime="exploit",
            requires_seeds=True,
            scoring_profile=ScoringProfile(
                potency_weight=0.7,
                novelty_weight=0.0,
                similarity_weight=0.2,
                uncertainty_penalty=0.1,
                diversity_weight=0.0,
            ),
            diversity_settings=DiversitySettings(
                bucket_similarity=0.7, penalty_strength=0.3, use_scaffold_memory=False
            ),
            tags=["exploit", "mol2mol", "mmp-edits", "local-search"],
            template_name="mol2mol.toml.j2",
        ),
        # LibInvent (decoration)
        ArmConfig(
            arm_id="libinvent_exploit",
            prior_filename="libinvent.prior",
            generator_type="libinvent",
            regime="exploit",
            requires_seeds=True,
            scoring_profile=ScoringProfile(
                potency_weight=0.6,
                novelty_weight=0.1,
                similarity_weight=0.2,
                uncertainty_penalty=0.1,
                diversity_weight=0.0,
            ),
            diversity_settings=DiversitySettings(
                bucket_similarity=0.65, penalty_strength=0.4, use_scaffold_memory=False
            ),
            tags=["exploit", "libinvent", "decorate"],
            template_name="libinvent.toml.j2",
        ),
        # LinkInvent (linker design)
        ArmConfig(
            arm_id="linkinvent_explore",
            prior_filename="linkinvent.prior",
            generator_type="linkinvent",
            regime="explore",
            requires_seeds=True,
            scoring_profile=ScoringProfile(
                potency_weight=0.4,
                novelty_weight=0.3,
                similarity_weight=0.1,
                uncertainty_penalty=0.1,
                diversity_weight=0.1,
            ),
            diversity_settings=DiversitySettings(
                bucket_similarity=0.6, penalty_strength=0.5, use_scaffold_memory=True
            ),
            tags=["explore", "linkinvent", "link"],
            template_name="linkinvent.toml.j2",
        ),
        # PubChem prior (large diverse prior)
        ArmConfig(
            arm_id="pubchem_explore",
            prior_filename="pubchem_ecfp4_with_count_with_rank_reinvent4_dict_voc.prior",
            generator_type="reinvent",
            regime="explore",
            requires_seeds=False,
            scoring_profile=ScoringProfile(
                potency_weight=0.2,
                novelty_weight=0.5,
                similarity_weight=0.0,
                uncertainty_penalty=0.1,
                diversity_weight=0.2,
            ),
            diversity_settings=DiversitySettings(
                bucket_similarity=0.5, penalty_strength=0.7, use_scaffold_memory=True
            ),
            tags=["explore", "de-novo", "global", "pubchem"],
            template_name="reinvent.toml.j2",
        ),
        # Optional transformer priors (if available)
        ArmConfig(
            arm_id="libinvent_transformer_exploit",
            prior_filename="libinvent_transformer_pubchem.prior",
            generator_type="libinvent",
            regime="exploit",
            requires_seeds=True,
            scoring_profile=ScoringProfile(
                potency_weight=0.6,
                novelty_weight=0.1,
                similarity_weight=0.2,
                uncertainty_penalty=0.1,
                diversity_weight=0.0,
            ),
            diversity_settings=DiversitySettings(
                bucket_similarity=0.65, penalty_strength=0.4, use_scaffold_memory=False
            ),
            tags=["exploit", "libinvent", "decorate", "transformer"],
            template_name="libinvent.toml.j2",
        ),
        ArmConfig(
            arm_id="linkinvent_transformer_explore",
            prior_filename="linkinvent_transformer_pubchem.prior",
            generator_type="linkinvent",
            regime="explore",
            requires_seeds=True,
            scoring_profile=ScoringProfile(
                potency_weight=0.4,
                novelty_weight=0.3,
                similarity_weight=0.1,
                uncertainty_penalty=0.1,
                diversity_weight=0.1,
            ),
            diversity_settings=DiversitySettings(
                bucket_similarity=0.6, penalty_strength=0.5, use_scaffold_memory=True
            ),
            tags=["explore", "linkinvent", "link", "transformer"],
            template_name="linkinvent.toml.j2",
        ),
        # QSAR-guided REINVENT arms
        ArmConfig(
            arm_id="reinvent_qsar_explore",
            prior_filename="reinvent.prior",
            generator_type="reinvent",
            regime="explore",
            requires_seeds=False,
            scoring_profile=ScoringProfile(
                potency_weight=0.4,  # QSAR will be primary scorer
                novelty_weight=0.3,
                similarity_weight=0.0,
                uncertainty_penalty=0.1,
                diversity_weight=0.2,
            ),
            diversity_settings=DiversitySettings(
                bucket_similarity=0.55, penalty_strength=0.6, use_scaffold_memory=True
            ),
            tags=["explore", "de-novo", "qsar", "activity-guided"],
            template_name="reinvent_qsar.toml.j2",
        ),
        ArmConfig(
            arm_id="reinvent_qsar_exploit",
            prior_filename="reinvent.prior",
            generator_type="reinvent",
            regime="exploit",
            requires_seeds=False,
            scoring_profile=ScoringProfile(
                potency_weight=0.7,  # Focus on predicted activity
                novelty_weight=0.0,
                similarity_weight=0.0,
                uncertainty_penalty=0.2,
                diversity_weight=0.1,
            ),
            diversity_settings=DiversitySettings(
                bucket_similarity=0.65, penalty_strength=0.4, use_scaffold_memory=False
            ),
            tags=["exploit", "de-novo", "qsar", "activity-focused"],
            template_name="reinvent_qsar.toml.j2",
        ),
        # QSAR-guided Mol2Mol arms
        ArmConfig(
            arm_id="mol2mol_qsar_high_sim",
            prior_filename="mol2mol_high_similarity.prior",
            generator_type="mol2mol",
            regime="exploit",
            requires_seeds=True,
            scoring_profile=ScoringProfile(
                potency_weight=0.7,
                novelty_weight=0.0,
                similarity_weight=0.2,
                uncertainty_penalty=0.1,
                diversity_weight=0.0,
            ),
            diversity_settings=DiversitySettings(
                bucket_similarity=0.7, penalty_strength=0.3, use_scaffold_memory=False
            ),
            tags=["exploit", "mol2mol", "qsar", "high-similarity", "local-search"],
            template_name="mol2mol_qsar.toml.j2",
        ),
        ArmConfig(
            arm_id="mol2mol_qsar_similarity",
            prior_filename="mol2mol_similarity.prior",
            generator_type="mol2mol",
            regime="explore",
            requires_seeds=True,
            scoring_profile=ScoringProfile(
                potency_weight=0.5,
                novelty_weight=0.2,
                similarity_weight=0.1,
                uncertainty_penalty=0.1,
                diversity_weight=0.1,
            ),
            diversity_settings=DiversitySettings(
                bucket_similarity=0.6, penalty_strength=0.5, use_scaffold_memory=True
            ),
            tags=["explore", "mol2mol", "qsar", "similarity-guided"],
            template_name="mol2mol_qsar.toml.j2",
        ),
    ]

    return catalog


def filter_available_arms(
    catalog: List[ArmConfig], priors_dir: Path
) -> tuple[List[ArmConfig], Dict[str, str]]:
    """
    Filter arms based on which priors are actually available.

    Returns:
        - List of available arms
        - Dict of unavailable arm_id -> reason
    """
    available = []
    unavailable = {}

    for arm in catalog:
        prior_path = priors_dir / arm.prior_filename
        if prior_path.exists():
            available.append(arm)
            logger.info(f"✓ Arm '{arm.arm_id}' enabled (prior found: {arm.prior_filename})")
        else:
            unavailable[arm.arm_id] = f"Prior not found: {prior_path}"
            logger.warning(f"✗ Arm '{arm.arm_id}' disabled (prior missing: {prior_path})")

    return available, unavailable
