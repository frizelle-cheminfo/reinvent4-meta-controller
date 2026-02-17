"""Metrics computation for generated molecules."""

import logging
from typing import Dict, List, Optional, Set

import numpy as np

from .config import PropertyConstraints
from .state import EpisodeMetrics, MoleculeRecord
from .utils import (
    canonicalize_smiles,
    check_property_constraints,
    check_rdkit,
    compute_ecfp4_fingerprint,
    compute_internal_diversity,
    compute_molecular_properties,
    compute_tanimoto_similarity,
    get_murcko_scaffold,
    is_valid_smiles,
)

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculate metrics for an episode's generated molecules."""

    def __init__(
        self,
        property_constraints: PropertyConstraints,
        hit_threshold: float = 0.6,
        ood_similarity_threshold: float = 0.3,
        ood_uncertainty_threshold: float = 0.5,
    ):
        self.property_constraints = property_constraints
        self.hit_threshold = hit_threshold
        self.ood_similarity_threshold = ood_similarity_threshold
        self.ood_uncertainty_threshold = ood_uncertainty_threshold

    def compute_episode_metrics(
        self,
        generated_smiles: List[str],
        scores: List[float],
        support_set: Optional[List[str]] = None,
        prior_generated: Optional[List[str]] = None,
        reference_actives: Optional[List[str]] = None,
        campaign_best_score: float = 0.0,
        uncertainties: Optional[List[float]] = None,
    ) -> EpisodeMetrics:
        """
        Compute comprehensive metrics for an episode.

        Args:
            generated_smiles: List of generated SMILES
            scores: Corresponding scores
            support_set: Support set for novelty (seed bank + reference actives)
            prior_generated: All previously generated molecules (for rediscovery)
            reference_actives: Optional reference actives for rediscovery
            campaign_best_score: Best score so far in campaign
            uncertainties: Optional per-molecule uncertainty scores

        Returns:
            EpisodeMetrics object
        """
        if not generated_smiles:
            return EpisodeMetrics()

        # Validity
        valid_smiles = [s for s in generated_smiles if is_valid_smiles(s)]
        valid_pct = len(valid_smiles) / len(generated_smiles) * 100

        # Uniqueness
        unique_smiles = list(set(valid_smiles))
        unique_pct = len(unique_smiles) / len(valid_smiles) * 100 if valid_smiles else 0.0

        # Internal diversity (requires RDKit)
        internal_diversity = 0.0
        if check_rdkit() and unique_smiles:
            internal_diversity = compute_internal_diversity(unique_smiles)

        # Scaffold diversity
        scaffold_diversity = self._compute_scaffold_diversity(unique_smiles)

        # Rediscovery rate
        rediscovery_rate = self._compute_rediscovery_rate(
            unique_smiles, prior_generated, reference_actives
        )

        # Novelty
        novelty = self._compute_novelty(unique_smiles, support_set)

        # OOD rate
        ood_rate = self._compute_ood_rate(unique_smiles, support_set, uncertainties)

        # Uncertainty mean
        uncertainty_mean = np.mean(uncertainties) if uncertainties else 0.0

        # Top-k gain
        best_score = max(scores) if scores else 0.0
        topk_gain = max(0.0, best_score - campaign_best_score)

        # Hit yield
        hit_yield = self._compute_hit_yield(scores)

        # Property pass rate
        property_pass_rate = self._compute_property_pass_rate(unique_smiles)

        # Best molecule
        if scores:
            best_idx = np.argmax(scores)
            best_smiles = generated_smiles[best_idx]
        else:
            best_smiles = ""

        return EpisodeMetrics(
            valid_pct=valid_pct,
            unique_pct=unique_pct,
            internal_diversity=internal_diversity,
            scaffold_diversity=scaffold_diversity,
            rediscovery_rate=rediscovery_rate,
            novelty=novelty,
            ood_rate=ood_rate,
            uncertainty_mean=uncertainty_mean,
            topk_gain=topk_gain,
            hit_yield=hit_yield,
            property_pass_rate=property_pass_rate,
            best_score=best_score,
            best_smiles=best_smiles,
        )

    def _compute_scaffold_diversity(self, smiles_list: List[str]) -> float:
        """Compute scaffold diversity (unique scaffolds / total)."""
        if not check_rdkit() or not smiles_list:
            return 0.0

        scaffolds = set()
        for smi in smiles_list:
            scaffold = get_murcko_scaffold(smi)
            if scaffold:
                scaffolds.add(scaffold)

        return len(scaffolds) / len(smiles_list) if smiles_list else 0.0

    def _compute_rediscovery_rate(
        self,
        smiles_list: List[str],
        prior_generated: Optional[List[str]],
        reference_actives: Optional[List[str]],
    ) -> float:
        """Compute rate of rediscovering known molecules."""
        if not smiles_list:
            return 0.0

        # Canonicalize all
        canonical_smiles = {canonicalize_smiles(s) for s in smiles_list if s}
        canonical_smiles.discard(None)

        known_set = set()
        if prior_generated:
            known_set.update(canonicalize_smiles(s) for s in prior_generated if s)
        if reference_actives:
            known_set.update(canonicalize_smiles(s) for s in reference_actives if s)
        known_set.discard(None)

        if not known_set or not canonical_smiles:
            return 0.0

        rediscovered = canonical_smiles.intersection(known_set)
        return len(rediscovered) / len(canonical_smiles)

    def _compute_novelty(
        self, smiles_list: List[str], support_set: Optional[List[str]]
    ) -> float:
        """
        Compute novelty as average distance to nearest support molecule.

        If RDKit unavailable or no support set, returns 0.5 (neutral).
        """
        if not check_rdkit() or not support_set or not smiles_list:
            return 0.5

        # Sample if too large
        import random

        sample_smiles = (
            random.sample(smiles_list, min(100, len(smiles_list)))
            if len(smiles_list) > 100
            else smiles_list
        )

        novelties = []
        for smi in sample_smiles:
            fp = compute_ecfp4_fingerprint(smi)
            if fp is None:
                continue

            max_sim = 0.0
            for support_smi in support_set:
                sim = compute_tanimoto_similarity(smi, support_smi)
                if sim is not None and sim > max_sim:
                    max_sim = sim

            novelty = 1.0 - max_sim  # Distance
            novelties.append(novelty)

        return np.mean(novelties) if novelties else 0.5

    def _compute_ood_rate(
        self,
        smiles_list: List[str],
        support_set: Optional[List[str]],
        uncertainties: Optional[List[float]],
    ) -> float:
        """
        Compute out-of-distribution rate.

        A molecule is OOD if:
        - Similarity to support set < threshold OR
        - Uncertainty > threshold (if uncertainties provided)
        """
        if not smiles_list:
            return 0.0

        ood_count = 0

        for i, smi in enumerate(smiles_list):
            is_ood = False

            # Check uncertainty
            if uncertainties and i < len(uncertainties):
                if uncertainties[i] > self.ood_uncertainty_threshold:
                    is_ood = True

            # Check similarity to support (if RDKit available)
            if check_rdkit() and support_set and not is_ood:
                max_sim = 0.0
                fp = compute_ecfp4_fingerprint(smi)
                if fp is not None:
                    for support_smi in support_set[:100]:  # Sample support
                        sim = compute_tanimoto_similarity(smi, support_smi)
                        if sim is not None and sim > max_sim:
                            max_sim = sim
                    if max_sim < self.ood_similarity_threshold:
                        is_ood = True

            if is_ood:
                ood_count += 1

        return ood_count / len(smiles_list)

    def _compute_hit_yield(self, scores: List[float]) -> float:
        """Compute fraction of molecules above hit threshold."""
        if not scores:
            return 0.0
        hits = sum(1 for s in scores if s >= self.hit_threshold)
        return hits / len(scores)

    def _compute_property_pass_rate(self, smiles_list: List[str]) -> float:
        """Compute fraction passing property constraints."""
        if not check_rdkit() or not smiles_list:
            return 1.0  # Default to pass if can't check

        passed = 0
        constraints_dict = self.property_constraints.model_dump()

        for smi in smiles_list:
            props = compute_molecular_properties(smi)
            if props and check_property_constraints(props, constraints_dict):
                passed += 1

        return passed / len(smiles_list) if smiles_list else 1.0


def compute_episode_quality(metrics: EpisodeMetrics) -> float:
    """
    Compute episode quality score for bandit selection.

    Q = + topk_gain + hit_yield + diversity_score
        - ood_rate - uncertainty_spike - rediscovery_rate
    """
    quality = (
        metrics.topk_gain * 2.0  # Weight improvements highly
        + metrics.hit_yield
        + (metrics.internal_diversity + metrics.scaffold_diversity) / 2.0
        - metrics.ood_rate
        - metrics.uncertainty_mean
        - metrics.rediscovery_rate
    )

    return quality
