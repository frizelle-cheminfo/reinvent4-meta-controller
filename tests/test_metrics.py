"""Tests for metrics computation."""

import pytest

from reinvent_meta.config import PropertyConstraints
from reinvent_meta.metrics import MetricsCalculator, compute_episode_quality
from reinvent_meta.state import EpisodeMetrics


def test_metrics_basic():
    """Test basic metrics computation."""
    calculator = MetricsCalculator(PropertyConstraints())

    smiles = ["CCO", "c1ccccc1", "CC(C)O", "CCO"]  # Includes duplicate
    scores = [0.5, 0.7, 0.6, 0.5]

    metrics = calculator.compute_episode_metrics(smiles, scores)

    assert metrics.valid_pct == 100.0
    assert metrics.unique_pct == 75.0  # 3 unique out of 4 valid
    assert metrics.best_score == 0.7
    assert 0 <= metrics.hit_yield <= 1.0


def test_metrics_invalid_smiles():
    """Test handling of invalid SMILES."""
    calculator = MetricsCalculator(PropertyConstraints())

    smiles = ["CCO", "INVALID", "c1ccccc1", ""]
    scores = [0.5, 0.7, 0.6, 0.8]

    metrics = calculator.compute_episode_metrics(smiles, scores)

    # Should handle invalid SMILES gracefully
    assert metrics.valid_pct < 100.0


def test_metrics_rediscovery():
    """Test rediscovery rate calculation."""
    calculator = MetricsCalculator(PropertyConstraints())

    smiles = ["CCO", "c1ccccc1", "CC(C)O"]
    scores = [0.5, 0.7, 0.6]
    prior = ["CCO"]  # One molecule rediscovered

    metrics = calculator.compute_episode_metrics(smiles, scores, prior_generated=prior)

    assert metrics.rediscovery_rate > 0.0


def test_metrics_topk_gain():
    """Test top-k gain calculation."""
    calculator = MetricsCalculator(PropertyConstraints())

    smiles = ["CCO", "c1ccccc1"]
    scores = [0.9, 0.7]

    metrics = calculator.compute_episode_metrics(smiles, scores, campaign_best_score=0.8)

    assert metrics.topk_gain == 0.1  # 0.9 - 0.8


def test_compute_episode_quality():
    """Test episode quality score computation."""
    metrics = EpisodeMetrics(
        topk_gain=0.1,
        hit_yield=0.5,
        internal_diversity=0.4,
        scaffold_diversity=0.6,
        ood_rate=0.2,
        uncertainty_mean=0.1,
        rediscovery_rate=0.1,
    )

    quality = compute_episode_quality(metrics)

    # Should be positive with these decent metrics
    assert quality > 0


def test_compute_episode_quality_negative():
    """Test episode quality with bad metrics."""
    metrics = EpisodeMetrics(
        topk_gain=0.0,
        hit_yield=0.1,
        internal_diversity=0.1,
        scaffold_diversity=0.1,
        ood_rate=0.8,
        uncertainty_mean=0.7,
        rediscovery_rate=0.6,
    )

    quality = compute_episode_quality(metrics)

    # Should be negative with these bad metrics
    assert quality < 0


def test_metrics_empty_input():
    """Test handling of empty input."""
    calculator = MetricsCalculator(PropertyConstraints())

    metrics = calculator.compute_episode_metrics([], [])

    assert metrics.valid_pct == 0.0
    assert metrics.best_score == 0.0
