"""Tests for report generation."""

import json
import tempfile
from pathlib import Path

import pytest

from reinvent_meta.reporting import ReportGenerator
from reinvent_meta.state import ControllerState, EpisodeMetrics, EpisodeRecord


def test_report_manifest():
    """Test manifest generation."""
    state = ControllerState(campaign_name="test", seed=42)

    # Add some episodes
    for i in range(3):
        state.episodes_history.append(
            EpisodeRecord(
                episode_num=i,
                arm_id=f"arm_{i}",
                prior_filename="test.prior",
                regime="explore",
                reason=["test"],
                metrics=EpisodeMetrics(best_score=0.5 + i * 0.1),
                output_dir=Path("/tmp"),
                success=True,
            )
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        generator = ReportGenerator(state, output_dir)

        manifest = generator._create_manifest()

        assert manifest["campaign_name"] == "test"
        assert manifest["total_episodes"] == 3
        assert len(manifest["episodes"]) == 3
        assert manifest["best_score"] == 0.7


def test_report_full_generation():
    """Test full report generation."""
    state = ControllerState(campaign_name="test", seed=42)

    # Add episodes
    for i in range(5):
        state.episodes_history.append(
            EpisodeRecord(
                episode_num=i,
                arm_id=f"arm_{i % 2}",
                prior_filename="test.prior",
                regime="explore" if i % 2 == 0 else "exploit",
                reason=["bandit-selection"],
                metrics=EpisodeMetrics(
                    best_score=0.5 + i * 0.05,
                    internal_diversity=0.4,
                    ood_rate=0.2,
                ),
                output_dir=Path("/tmp"),
                success=True,
            )
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        generator = ReportGenerator(state, output_dir)

        generator.generate_full_report()

        # Check files created
        assert (output_dir / "manifest.json").exists()
        assert (output_dir / "report.md").exists()
        assert (output_dir / "report.html").exists()

        # Check manifest is valid JSON
        with open(output_dir / "manifest.json") as f:
            manifest = json.load(f)
            assert manifest["campaign_name"] == "test"

        # Check markdown has content
        with open(output_dir / "report.md") as f:
            md_content = f.read()
            assert "Episode Timeline" in md_content
            assert "test" in md_content

        # Check HTML has content
        with open(output_dir / "report.html") as f:
            html_content = f.read()
            assert "<html>" in html_content
            assert "test" in html_content


def test_report_empty_episodes():
    """Test report generation with no episodes."""
    state = ControllerState(campaign_name="test", seed=42)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        generator = ReportGenerator(state, output_dir)

        # Should not crash with empty state
        generator.generate_full_report()

        assert (output_dir / "manifest.json").exists()
