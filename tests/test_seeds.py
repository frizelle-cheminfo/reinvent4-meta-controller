"""Tests for seed bank management."""

import pytest

from reinvent_meta.arms import get_arm_catalog
from reinvent_meta.seeds import SeedSelector, update_seed_bank
from reinvent_meta.state import SeedRecord


def test_seed_selector_exploit():
    """Test exploit seed selection (top scoring)."""
    seeds = [
        SeedRecord(seed_id=f"s{i}", smiles=f"C{i}", score=i / 10.0, added_episode=0)
        for i in range(10)
    ]

    selector = SeedSelector(seeds, random_seed=42)
    catalog = get_arm_catalog()
    exploit_arm = next(arm for arm in catalog if arm.regime == "exploit" and arm.requires_seeds)

    selected = selector.select_seeds_for_arm(exploit_arm, n_seeds=5)

    # Should select top 5
    assert len(selected) == 5
    scores = [s.score for s in selected]
    assert scores == sorted(scores, reverse=True)  # Descending order
    assert selected[0].score == 0.9  # Highest score


def test_seed_selector_explore():
    """Test explore seed selection (diverse)."""
    seeds = [
        SeedRecord(
            seed_id=f"s{i}",
            smiles=f"C{i}",
            score=i / 10.0,
            scaffold=f"scaffold_{i % 3}",  # 3 scaffold groups
            added_episode=0,
        )
        for i in range(9)
    ]

    selector = SeedSelector(seeds, random_seed=42)
    catalog = get_arm_catalog()
    explore_arm = next(arm for arm in catalog if arm.regime == "explore" and arm.requires_seeds)

    selected = selector.select_seeds_for_arm(explore_arm, n_seeds=6)

    # Should have diverse scaffolds
    assert len(selected) > 0
    scaffolds = {s.scaffold for s in selected}
    assert len(scaffolds) > 1  # Multiple scaffolds


def test_update_seed_bank():
    """Test seed bank update."""
    current_bank = [
        SeedRecord(seed_id="s1", smiles="CCO", score=0.5, added_episode=0),
        SeedRecord(seed_id="s2", smiles="c1ccccc1", score=0.6, added_episode=0),
    ]

    new_molecules = [("CC(C)O", 0.8), ("CCCO", 0.7), ("c1ccccc1", 0.6)]  # One duplicate

    updated = update_seed_bank(current_bank, new_molecules, episode=1, max_size=5)

    # Check deduplication
    smiles_set = {s.smiles for s in updated}
    assert len(smiles_set) == len(updated)

    # Check top scores kept
    assert updated[0].score == 0.8
    assert len(updated) <= 5


def test_update_seed_bank_max_size():
    """Test that seed bank respects max size."""
    current_bank = [
        SeedRecord(seed_id=f"s{i}", smiles=f"C{i}", score=i / 20.0, added_episode=0)
        for i in range(10)
    ]

    new_molecules = [(f"N{i}", 0.8 + i / 100.0) for i in range(20)]

    updated = update_seed_bank(current_bank, new_molecules, episode=1, max_size=15)

    assert len(updated) == 15
    # Check highest scores kept
    assert all(s.score >= 0.8 for s in updated[:10])


def test_seed_selector_empty_bank():
    """Test seed selection with empty bank."""
    selector = SeedSelector([], random_seed=42)
    catalog = get_arm_catalog()
    arm = catalog[0]

    selected = selector.select_seeds_for_arm(arm, n_seeds=5)

    assert len(selected) == 0
