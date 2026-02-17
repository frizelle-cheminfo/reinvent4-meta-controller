"""Tests for arm catalog and filtering."""

import tempfile
from pathlib import Path

import pytest

from reinvent_meta.arms import filter_available_arms, get_arm_catalog


def test_get_arm_catalog():
    """Test that arm catalog returns expected arms."""
    catalog = get_arm_catalog()

    assert len(catalog) > 0
    assert any(arm.arm_id == "reinvent_explore" for arm in catalog)
    assert any(arm.arm_id == "mol2mol_mmp_exploit" for arm in catalog)

    # Check structure
    for arm in catalog:
        assert arm.arm_id
        assert arm.prior_filename
        assert arm.generator_type in ["reinvent", "mol2mol", "libinvent", "linkinvent"]
        assert arm.regime in ["explore", "exploit"]
        assert isinstance(arm.requires_seeds, bool)
        assert isinstance(arm.tags, list)


def test_filter_available_arms():
    """Test filtering arms by available priors."""
    catalog = get_arm_catalog()

    # Create temp directory with some priors
    with tempfile.TemporaryDirectory() as tmpdir:
        priors_dir = Path(tmpdir)

        # Create a couple of mock prior files
        (priors_dir / "reinvent.prior").touch()
        (priors_dir / "mol2mol_mmp.prior").touch()

        available, unavailable = filter_available_arms(catalog, priors_dir)

        # Should have 2-3 available (some arms share priors)
        assert len(available) >= 2
        assert len(unavailable) > 0

        # Check that the right arms are available
        available_ids = {arm.arm_id for arm in available}
        assert "reinvent_explore" in available_ids or "reinvent_exploit" in available_ids
        assert "mol2mol_mmp_exploit" in available_ids


def test_filter_no_priors():
    """Test filtering when no priors available."""
    catalog = get_arm_catalog()

    with tempfile.TemporaryDirectory() as tmpdir:
        priors_dir = Path(tmpdir)
        available, unavailable = filter_available_arms(catalog, priors_dir)

        assert len(available) == 0
        assert len(unavailable) == len(catalog)


def test_arm_tags():
    """Test that arms have appropriate tags."""
    catalog = get_arm_catalog()

    for arm in catalog:
        if arm.regime == "explore":
            assert "explore" in arm.tags
        else:
            assert "exploit" in arm.tags

        if "mmp" in arm.arm_id:
            assert "mmp-edits" in arm.tags or "mmp" in arm.tags
