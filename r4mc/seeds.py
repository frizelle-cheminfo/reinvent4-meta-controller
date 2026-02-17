"""Seed bank management and selection policies."""

import logging
import random
from pathlib import Path
from typing import List, Optional

from .config import ArmConfig
from .state import SeedRecord
from .utils import check_rdkit, get_murcko_scaffold

logger = logging.getLogger(__name__)


class SeedSelector:
    """Handles seed selection from the seed bank."""

    def __init__(self, seed_bank: List[SeedRecord], random_seed: int = 42):
        self.seed_bank = seed_bank
        self.rng = random.Random(random_seed)

    def select_seeds_for_arm(
        self, arm: ArmConfig, n_seeds: int = 50, episode: int = 0
    ) -> List[SeedRecord]:
        """
        Select seeds for an arm based on its regime and requirements.

        Args:
            arm: The arm configuration
            n_seeds: Number of seeds to select
            episode: Current episode number

        Returns:
            List of selected seed records
        """
        if not self.seed_bank:
            logger.warning("Seed bank is empty, cannot select seeds")
            return []

        if arm.regime == "exploit":
            return self._select_exploit_seeds(n_seeds)
        else:  # explore
            return self._select_explore_seeds(n_seeds)

    def _select_exploit_seeds(self, n_seeds: int) -> List[SeedRecord]:
        """
        Select seeds for exploitation: top scoring seeds.

        Prioritizes high-scoring seeds that are within the applicability domain.
        """
        # Sort by score descending
        sorted_seeds = sorted(self.seed_bank, key=lambda s: s.score, reverse=True)

        # Take top N
        selected = sorted_seeds[:n_seeds]

        logger.info(
            f"Selected {len(selected)} exploit seeds (top scores: "
            f"{[f'{s.score:.3f}' for s in selected[:3]]})"
        )

        return selected

    def _select_explore_seeds(self, n_seeds: int) -> List[SeedRecord]:
        """
        Select seeds for exploration: diverse scaffolds.

        Uses scaffold clustering to pick representatives from different regions.
        """
        if not check_rdkit():
            # Fallback: random sampling
            logger.warning("RDKit unavailable, using random seed selection")
            return self.rng.sample(self.seed_bank, min(n_seeds, len(self.seed_bank)))

        # Cluster by scaffold
        scaffold_clusters = {}
        for seed in self.seed_bank:
            scaffold = seed.scaffold or get_murcko_scaffold(seed.smiles) or "unknown"
            if scaffold not in scaffold_clusters:
                scaffold_clusters[scaffold] = []
            scaffold_clusters[scaffold].append(seed)

        # Select representatives from each cluster
        selected = []
        clusters = list(scaffold_clusters.values())
        self.rng.shuffle(clusters)  # Randomize cluster order

        # Round-robin selection from clusters
        idx = 0
        while len(selected) < n_seeds and clusters:
            cluster = clusters[idx % len(clusters)]
            if cluster:
                # Pick best from this cluster
                best = max(cluster, key=lambda s: s.score)
                selected.append(best)
                cluster.remove(best)
            # Remove empty clusters
            clusters = [c for c in clusters if c]
            idx += 1

        logger.info(
            f"Selected {len(selected)} diverse seeds from {len(scaffold_clusters)} scaffold clusters"
        )

        return selected

    def export_seeds_to_file(self, seeds: List[SeedRecord], output_path: Path) -> None:
        """Export seeds to a SMILES file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for seed in seeds:
                # Write SMILES with optional ID
                f.write(f"{seed.smiles} {seed.seed_id}\n")

        logger.info(f"Exported {len(seeds)} seeds to {output_path}")


def load_reference_actives(path: Path) -> List[str]:
    """Load reference actives from SMILES file."""
    if not path.exists():
        logger.warning(f"Reference actives file not found: {path}")
        return []

    actives = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                # Split on whitespace, take first column as SMILES
                parts = line.split()
                if parts:
                    actives.append(parts[0])

    logger.info(f"Loaded {len(actives)} reference actives from {path}")
    return actives


def update_seed_bank(
    current_bank: List[SeedRecord],
    new_molecules: List[tuple[str, float]],
    episode: int,
    max_size: int = 100,
) -> List[SeedRecord]:
    """
    Update seed bank with new high-scoring molecules.

    Args:
        current_bank: Current seed bank
        new_molecules: List of (smiles, score) tuples
        episode: Current episode number
        max_size: Maximum seed bank size

    Returns:
        Updated seed bank
    """
    # Convert new molecules to seed records
    new_seeds = []
    for i, (smiles, score) in enumerate(new_molecules):
        scaffold = get_murcko_scaffold(smiles) if check_rdkit() else None
        seed = SeedRecord(
            seed_id=f"ep{episode}_mol{i}",
            smiles=smiles,
            score=score,
            scaffold=scaffold,
            added_episode=episode,
            times_used=0,
        )
        new_seeds.append(seed)

    # Combine and sort by score
    combined = current_bank + new_seeds
    combined.sort(key=lambda s: s.score, reverse=True)

    # Keep top N unique SMILES
    seen_smiles = set()
    deduplicated = []
    for seed in combined:
        if seed.smiles not in seen_smiles:
            deduplicated.append(seed)
            seen_smiles.add(seed.smiles)
            if len(deduplicated) >= max_size:
                break

    logger.info(
        f"Updated seed bank: {len(new_seeds)} new → {len(deduplicated)} total "
        f"(score range: {deduplicated[0].score:.3f} - {deduplicated[-1].score:.3f})"
    )

    return deduplicated


def initialize_seed_bank_from_actives(
    actives: List[str], episode: int = 0, default_score: float = 0.7
) -> List[SeedRecord]:
    """Initialize seed bank from reference actives."""
    seeds = []
    for i, smiles in enumerate(actives):
        scaffold = get_murcko_scaffold(smiles) if check_rdkit() else None
        seed = SeedRecord(
            seed_id=f"ref_active_{i}",
            smiles=smiles,
            score=default_score,
            scaffold=scaffold,
            added_episode=episode,
            times_used=0,
        )
        seeds.append(seed)

    logger.info(f"Initialized seed bank with {len(seeds)} reference actives")
    return seeds
