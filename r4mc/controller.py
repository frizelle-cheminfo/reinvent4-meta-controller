"""Meta-controller with adaptive arm selection."""

import logging
import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from .arms import ArmConfig
from .config import ControllerConfig, ThresholdConfig
from .metrics import compute_episode_quality
from .state import ArmStatistics, ControllerState, EpisodeMetrics, EpisodeRecord

logger = logging.getLogger(__name__)


class MetaController:
    """
    Adaptive meta-controller that selects arms using hard rules + bandit logic.

    The controller combines:
    1. Hard safety rules (override bandit if conditions met)
    2. UCB or Thompson sampling for arm selection among eligible arms
    """

    def __init__(
        self,
        config: ControllerConfig,
        state: ControllerState,
        available_arms: List[ArmConfig],
    ):
        self.config = config
        self.state = state
        self.available_arms = available_arms
        self.thresholds = config.thresholds
        self.rng = random.Random(config.seed)

    def select_next_arm(self) -> Tuple[ArmConfig, List[str]]:
        """
        Select the next arm to run.

        Returns:
            - Selected arm
            - List of reasons for selection
        """
        # Apply hard rules first
        forced_arm, reasons = self._apply_hard_rules()
        if forced_arm:
            logger.info(f"Hard rule selected arm '{forced_arm.arm_id}': {reasons}")
            return forced_arm, reasons

        # Use bandit selection among all available arms
        eligible_arms = [arm for arm in self.available_arms if not self._is_blacklisted(arm)]

        if not eligible_arms:
            logger.error("No eligible arms available!")
            # Fallback: pick any available arm
            eligible_arms = self.available_arms

        arm = self._bandit_selection(eligible_arms)
        reasons = ["bandit-selection"]

        logger.info(f"Bandit selected arm '{arm.arm_id}'")
        return arm, reasons

    def _apply_hard_rules(self) -> Tuple[Optional[ArmConfig], List[str]]:
        """
        Apply hard safety rules to override bandit if necessary.

        Returns:
            - Arm to force (or None)
            - List of trigger reasons
        """
        if not self.state.episodes_history:
            # First episode: pick an explore arm
            explore_arms = [
                arm for arm in self.available_arms if arm.regime == "explore" and not arm.requires_seeds
            ]
            if explore_arms:
                return self.rng.choice(explore_arms), ["first-episode", "explore-de-novo"]
            return None, []

        # Get recent episode metrics
        recent_episodes = self.state.episodes_history[-3:]
        recent_metrics = [ep.metrics for ep in recent_episodes if ep.success]

        if not recent_metrics:
            return None, []

        latest_metrics = recent_metrics[-1]
        reasons = []

        # Rule 1: High OOD rate → prefer exploit/similarity/MMP arms
        if latest_metrics.ood_rate > self.thresholds.ood_high:
            reasons.append(f"high-ood-rate-{latest_metrics.ood_rate:.2f}")
            exploit_arms = [
                arm
                for arm in self.available_arms
                if arm.regime == "exploit"
                and ("mmp" in arm.tags or "high-similarity" in arm.tags or "local-search" in arm.tags)
                and not self._is_blacklisted(arm)
            ]
            if exploit_arms:
                return self.rng.choice(exploit_arms), reasons

        # Rule 2: High uncertainty spike → prefer exploit/similarity
        if latest_metrics.uncertainty_mean > self.thresholds.uncertainty_spike:
            reasons.append(f"uncertainty-spike-{latest_metrics.uncertainty_mean:.2f}")
            exploit_arms = [
                arm for arm in self.available_arms
                if arm.regime == "exploit" and arm.requires_seeds and not self._is_blacklisted(arm)
            ]
            if exploit_arms:
                return self.rng.choice(exploit_arms), reasons

        # Rule 3: Diversity collapse → prefer explore/scaffold/global
        if latest_metrics.internal_diversity < self.thresholds.diversity_collapse:
            reasons.append(f"diversity-collapse-{latest_metrics.internal_diversity:.2f}")
            explore_arms = [
                arm
                for arm in self.available_arms
                if arm.regime == "explore"
                and ("scaffold-hop" in arm.tags or "global" in arm.tags)
                and not self._is_blacklisted(arm)
            ]
            if explore_arms:
                return self.rng.choice(explore_arms), reasons

        # Rule 4: High rediscovery rate → prefer explore/scaffold
        if latest_metrics.rediscovery_rate > self.thresholds.rediscovery_high:
            reasons.append(f"high-rediscovery-{latest_metrics.rediscovery_rate:.2f}")
            explore_arms = [
                arm
                for arm in self.available_arms
                if arm.regime == "explore"
                and ("scaffold-hop" in arm.tags or "global" in arm.tags)
                and not self._is_blacklisted(arm)
            ]
            if explore_arms:
                return self.rng.choice(explore_arms), reasons

        # Rule 5: Property pass rate collapse → prefer similarity/MMP/libinvent
        if latest_metrics.property_pass_rate < self.thresholds.property_pass_low:
            reasons.append(f"property-filter-collapse-{latest_metrics.property_pass_rate:.2f}")
            constrained_arms = [
                arm
                for arm in self.available_arms
                if ("high-similarity" in arm.tags
                or "mmp" in arm.tags
                or "libinvent" in arm.tags)
                and not self._is_blacklisted(arm)
            ]
            if constrained_arms:
                return self.rng.choice(constrained_arms), reasons

        # Rule 6: Stagnation (no improvement) → try different regime
        if len(recent_metrics) >= self.thresholds.stagnation_episodes:
            improvements = [m.topk_gain for m in recent_metrics]
            if all(gain < 0.01 for gain in improvements):
                reasons.append("stagnation-no-improvement")
                # Get regime of last arm
                last_regime = self.state.episodes_history[-1].regime
                # Switch to opposite regime
                opposite_regime = "explore" if last_regime == "exploit" else "exploit"
                opposite_arms = [
                    arm for arm in self.available_arms
                    if arm.regime == opposite_regime and not self._is_blacklisted(arm)
                ]
                if opposite_arms:
                    return self.rng.choice(opposite_arms), reasons

        return None, []

    def _bandit_selection(self, arms: List[ArmConfig]) -> ArmConfig:
        """
        Select arm using UCB (Upper Confidence Bound).

        For each arm, compute:
            UCB = mean_quality + C * sqrt(log(total_episodes) / episodes_run)

        Where C is the exploration parameter.
        """
        total_episodes = self.state.current_episode + 1

        ucb_scores = []
        for arm in arms:
            stats = self.state.get_arm_stats(arm.arm_id)

            if stats.episodes_run == 0:
                # Unplayed arm: give infinite score to explore it
                ucb_score = float("inf")
            else:
                # UCB formula
                mean_q = stats.mean_quality
                exploration_bonus = self.config.bandit_ucb_c * np.sqrt(
                    np.log(total_episodes) / stats.episodes_run
                )
                ucb_score = mean_q + exploration_bonus

            ucb_scores.append((arm, ucb_score))
            logger.debug(
                f"Arm '{arm.arm_id}': UCB={ucb_score:.3f} "
                f"(mean_Q={stats.mean_quality:.3f}, n={stats.episodes_run})"
            )

        # Select arm with highest UCB score
        selected_arm = max(ucb_scores, key=lambda x: x[1])[0]
        return selected_arm

    def _is_blacklisted(self, arm: ArmConfig) -> bool:
        """Check if arm is blacklisted."""
        stats = self.state.get_arm_stats(arm.arm_id)
        return stats.blacklisted

    def record_episode_result(
        self, arm: ArmConfig, metrics: EpisodeMetrics, success: bool
    ) -> None:
        """
        Record episode result and update arm statistics.

        Args:
            arm: The arm that was run
            metrics: Computed metrics
            success: Whether episode succeeded
        """
        # Compute quality score
        quality = compute_episode_quality(metrics) if success else -1.0

        # Update arm stats
        self.state.update_arm_stats(arm.arm_id, quality, success)

        logger.info(
            f"Recorded episode result for '{arm.arm_id}': "
            f"success={success}, quality={quality:.3f}"
        )


def get_strategy_narrative(episodes: List[EpisodeRecord]) -> List[Dict]:
    """
    Extract strategy switches from episode history.

    Returns list of switch events with triggers and context.
    """
    switches = []

    for i, episode in enumerate(episodes):
        # Look for switch indicators in reasons
        reasons = episode.reason
        if not reasons or "bandit-selection" in reasons:
            continue

        # This was a hard-rule switch
        switch = {
            "episode": episode.episode_num,
            "from_arm": episodes[i - 1].arm_id if i > 0 else "none",
            "to_arm": episode.arm_id,
            "triggers": reasons,
            "metrics": {
                "ood_rate": episode.metrics.ood_rate,
                "diversity": episode.metrics.internal_diversity,
                "rediscovery": episode.metrics.rediscovery_rate,
                "best_score": episode.metrics.best_score,
            },
        }
        switches.append(switch)

    return switches
