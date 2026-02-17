"""Report generation for campaign analysis."""

import base64
import json
import logging
from io import BytesIO
from pathlib import Path
from typing import Dict, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from .controller import get_strategy_narrative
from .state import ControllerState, EpisodeRecord

matplotlib.use("Agg")  # Non-interactive backend

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate comprehensive campaign reports."""

    def __init__(self, state: ControllerState, output_dir: Path):
        self.state = state
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_full_report(self) -> None:
        """Generate all report artifacts."""
        logger.info(f"Generating report in {self.output_dir}")

        # Generate manifest
        manifest = self._create_manifest()
        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)
        logger.info(f"Created manifest: {manifest_path}")

        # Generate plots
        plots = self._generate_plots()
        logger.info(f"Generated {len(plots)} plots")

        # Generate markdown report
        md_report = self._generate_markdown_report(manifest, plots)
        md_path = self.output_dir / "report.md"
        with open(md_path, "w") as f:
            f.write(md_report)
        logger.info(f"Created markdown report: {md_path}")

        # Generate HTML report
        html_report = self._generate_html_report(manifest, plots)
        html_path = self.output_dir / "report.html"
        with open(html_path, "w") as f:
            f.write(html_report)
        logger.info(f"Created HTML report: {html_path}")

    def _create_manifest(self) -> Dict:
        """Create machine-readable manifest."""
        episodes_data = []
        for ep in self.state.episodes_history:
            episodes_data.append(
                {
                    "episode": ep.episode_num,
                    "arm_id": ep.arm_id,
                    "prior": ep.prior_filename,
                    "regime": ep.regime,
                    "reason": ep.reason,
                    "success": ep.success,
                    "metrics": ep.metrics.model_dump(),
                }
            )

        arm_stats_data = {
            arm_id: stats.model_dump() for arm_id, stats in self.state.arm_stats.items()
        }

        manifest = {
            "campaign_name": self.state.campaign_name,
            "total_episodes": self.state.current_episode,
            "best_score": self.state.best_score,
            "episodes": episodes_data,
            "arm_statistics": arm_stats_data,
            "seed_bank_size": len(self.state.seed_bank),
            "total_molecules_generated": len(self.state.all_generated_smiles),
        }

        return manifest

    def _generate_plots(self) -> Dict[str, str]:
        """Generate plots and return as base64-encoded PNGs."""
        plots = {}

        if not self.state.episodes_history:
            logger.warning("No episodes to plot")
            return plots

        successful_episodes = [ep for ep in self.state.episodes_history if ep.success]
        if not successful_episodes:
            logger.warning("No successful episodes to plot")
            return plots

        # Plot 1: Best score over episodes
        plots["best_score"] = self._plot_best_score(successful_episodes)

        # Plot 2: Diversity over episodes
        plots["diversity"] = self._plot_diversity(successful_episodes)

        # Plot 3: OOD rate over episodes
        plots["ood_rate"] = self._plot_ood_rate(successful_episodes)

        # Plot 4: Arm selection frequency
        plots["arm_frequency"] = self._plot_arm_frequency(successful_episodes)

        # Plot 5: Quality scores per arm
        plots["arm_quality"] = self._plot_arm_quality()

        return plots

    def _plot_best_score(self, episodes: List[EpisodeRecord]) -> str:
        """Plot best score progression."""
        fig, ax = plt.subplots(figsize=(10, 6))

        episode_nums = [ep.episode_num for ep in episodes]
        scores = [ep.metrics.best_score for ep in episodes]

        # Plot scores
        ax.plot(episode_nums, scores, marker="o", linewidth=2, markersize=6)

        # Also plot cumulative best
        cumulative_best = []
        best_so_far = 0.0
        for score in scores:
            best_so_far = max(best_so_far, score)
            cumulative_best.append(best_so_far)
        ax.plot(
            episode_nums, cumulative_best, linestyle="--", linewidth=2, label="Cumulative Best"
        )

        ax.set_xlabel("Episode", fontsize=12)
        ax.set_ylabel("Best Score", fontsize=12)
        ax.set_title("Best Score Progression", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        return self._fig_to_base64(fig)

    def _plot_diversity(self, episodes: List[EpisodeRecord]) -> str:
        """Plot diversity metrics."""
        fig, ax = plt.subplots(figsize=(10, 6))

        episode_nums = [ep.episode_num for ep in episodes]
        internal_div = [ep.metrics.internal_diversity for ep in episodes]
        scaffold_div = [ep.metrics.scaffold_diversity for ep in episodes]

        ax.plot(episode_nums, internal_div, marker="o", label="Internal Diversity", linewidth=2)
        ax.plot(episode_nums, scaffold_div, marker="s", label="Scaffold Diversity", linewidth=2)

        ax.set_xlabel("Episode", fontsize=12)
        ax.set_ylabel("Diversity", fontsize=12)
        ax.set_title("Diversity Over Episodes", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        return self._fig_to_base64(fig)

    def _plot_ood_rate(self, episodes: List[EpisodeRecord]) -> str:
        """Plot OOD rate."""
        fig, ax = plt.subplots(figsize=(10, 6))

        episode_nums = [ep.episode_num for ep in episodes]
        ood_rates = [ep.metrics.ood_rate for ep in episodes]
        uncertainty = [ep.metrics.uncertainty_mean for ep in episodes]

        ax.plot(episode_nums, ood_rates, marker="o", label="OOD Rate", linewidth=2)
        ax.plot(episode_nums, uncertainty, marker="s", label="Uncertainty", linewidth=2)

        ax.set_xlabel("Episode", fontsize=12)
        ax.set_ylabel("Rate", fontsize=12)
        ax.set_title("OOD Rate and Uncertainty", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        return self._fig_to_base64(fig)

    def _plot_arm_frequency(self, episodes: List[EpisodeRecord]) -> str:
        """Plot arm selection frequency."""
        fig, ax = plt.subplots(figsize=(10, 6))

        # Count arm usage
        arm_counts = {}
        for ep in episodes:
            arm_counts[ep.arm_id] = arm_counts.get(ep.arm_id, 0) + 1

        arms = list(arm_counts.keys())
        counts = list(arm_counts.values())

        ax.barh(arms, counts)
        ax.set_xlabel("Episodes Run", fontsize=12)
        ax.set_ylabel("Arm", fontsize=12)
        ax.set_title("Arm Selection Frequency", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="x")
        plt.tight_layout()

        return self._fig_to_base64(fig)

    def _plot_arm_quality(self) -> str:
        """Plot quality scores per arm."""
        fig, ax = plt.subplots(figsize=(10, 6))

        arms = []
        qualities = []

        for arm_id, stats in self.state.arm_stats.items():
            if stats.successes > 0:
                arms.append(arm_id)
                qualities.append(stats.mean_quality)

        if not arms:
            # No data
            ax.text(
                0.5, 0.5, "No arm statistics available", ha="center", va="center", fontsize=14
            )
        else:
            ax.barh(arms, qualities)
            ax.set_xlabel("Mean Quality Score", fontsize=12)
            ax.set_ylabel("Arm", fontsize=12)
            ax.set_title("Arm Performance (Mean Quality)", fontsize=14, fontweight="bold")
            ax.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()
        return self._fig_to_base64(fig)

    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 PNG."""
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        return img_base64

    def _generate_markdown_report(self, manifest: Dict, plots: Dict[str, str]) -> str:
        """Generate markdown report."""
        md = []
        md.append(f"# REINVENT4 Meta-Controller Report: {self.state.campaign_name}\n")
        md.append(f"**Total Episodes:** {manifest['total_episodes']}\n")
        md.append(f"**Best Score Achieved:** {manifest['best_score']:.4f}\n")
        md.append(f"**Total Molecules Generated:** {manifest['total_molecules_generated']}\n")
        md.append("\n---\n")

        # Episode timeline
        md.append("## Episode Timeline\n")
        md.append(
            "| Episode | Arm | Prior | Regime | Reason | Best Score | Diversity | OOD Rate |\n"
        )
        md.append("|---------|-----|-------|--------|--------|------------|-----------|----------|\n")

        for ep_data in manifest["episodes"]:
            reasons = ", ".join(ep_data["reason"]) if ep_data["reason"] else "bandit"
            metrics = ep_data["metrics"]
            md.append(
                f"| {ep_data['episode']} | {ep_data['arm_id']} | "
                f"{ep_data['prior']} | {ep_data['regime']} | {reasons} | "
                f"{metrics['best_score']:.3f} | {metrics['internal_diversity']:.3f} | "
                f"{metrics['ood_rate']:.3f} |\n"
            )

        md.append("\n---\n")

        # Strategy switches
        md.append("## Strategy Switches\n")
        switches = get_strategy_narrative(self.state.episodes_history)
        if switches:
            for switch in switches:
                md.append(f"### Episode {switch['episode']}\n")
                md.append(f"**Switch:** `{switch['from_arm']}` → `{switch['to_arm']}`\n\n")
                md.append(f"**Triggers:** {', '.join(switch['triggers'])}\n\n")
                md.append("**Metrics at switch:**\n")
                md.append(f"- OOD Rate: {switch['metrics']['ood_rate']:.3f}\n")
                md.append(f"- Diversity: {switch['metrics']['diversity']:.3f}\n")
                md.append(f"- Rediscovery: {switch['metrics']['rediscovery']:.3f}\n")
                md.append(f"- Best Score: {switch['metrics']['best_score']:.3f}\n\n")
        else:
            md.append("No strategy switches triggered (bandit-only selection).\n")

        md.append("\n---\n")

        # Molecule evolution
        md.append("## Molecule Evolution\n")
        md.append("### Top Molecules per Episode\n")

        for ep in self.state.episodes_history:
            if ep.success:
                md.append(f"**Episode {ep.episode_num}** ({ep.arm_id}):\n")
                md.append(f"- Best: `{ep.metrics.best_smiles}` (score: {ep.metrics.best_score:.3f})\n")
                md.append("\n")

        md.append("\n---\n")

        # Arm statistics
        md.append("## Arm Performance Summary\n")
        md.append("| Arm | Episodes Run | Mean Quality | Successes | Failures |\n")
        md.append("|-----|--------------|--------------|-----------|----------|\n")

        for arm_id, stats in manifest["arm_statistics"].items():
            md.append(
                f"| {arm_id} | {stats['episodes_run']} | "
                f"{stats['mean_quality']:.3f} | {stats['successes']} | {stats['failures']} |\n"
            )

        md.append("\n---\n")
        md.append("## Plots\n")
        md.append("See the HTML report for interactive visualizations.\n")

        return "".join(md)

    def _generate_html_report(self, manifest: Dict, plots: Dict[str, str]) -> str:
        """Generate HTML report with embedded plots."""
        html = []
        html.append("<!DOCTYPE html>\n<html>\n<head>\n")
        html.append(f"<title>REINVENT4 Meta-Controller Report: {self.state.campaign_name}</title>\n")
        html.append("<style>\n")
        html.append(
            """
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 1200px; margin: auto; background: white; padding: 30px;
                         box-shadow: 0 0 10px rgba(0,0,0,0.1); }
            h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
            h2 { color: #34495e; margin-top: 30px; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
            th { background-color: #3498db; color: white; }
            tr:nth-child(even) { background-color: #f2f2f2; }
            .plot { margin: 20px 0; text-align: center; }
            .plot img { max-width: 100%; height: auto; box-shadow: 0 0 5px rgba(0,0,0,0.2); }
            .metric-box { display: inline-block; margin: 10px; padding: 15px;
                          background: #ecf0f1; border-radius: 5px; }
            .metric-value { font-size: 24px; font-weight: bold; color: #2c3e50; }
            .metric-label { font-size: 12px; color: #7f8c8d; }
            code { background: #ecf0f1; padding: 2px 5px; border-radius: 3px; }
        """
        )
        html.append("</style>\n</head>\n<body>\n<div class='container'>\n")

        # Header
        html.append(f"<h1>REINVENT4 Meta-Controller Report</h1>\n")
        html.append(f"<h2>{self.state.campaign_name}</h2>\n")

        # Metrics boxes
        html.append("<div style='margin: 20px 0;'>\n")
        html.append("<div class='metric-box'>")
        html.append(f"<div class='metric-value'>{manifest['total_episodes']}</div>")
        html.append("<div class='metric-label'>Total Episodes</div>")
        html.append("</div>")

        html.append("<div class='metric-box'>")
        html.append(f"<div class='metric-value'>{manifest['best_score']:.4f}</div>")
        html.append("<div class='metric-label'>Best Score</div>")
        html.append("</div>")

        html.append("<div class='metric-box'>")
        html.append(f"<div class='metric-value'>{manifest['total_molecules_generated']}</div>")
        html.append("<div class='metric-label'>Molecules Generated</div>")
        html.append("</div>")

        html.append("<div class='metric-box'>")
        html.append(f"<div class='metric-value'>{manifest['seed_bank_size']}</div>")
        html.append("<div class='metric-label'>Seed Bank Size</div>")
        html.append("</div>")
        html.append("</div>\n")

        # Plots
        html.append("<h2>Visualizations</h2>\n")

        plot_titles = {
            "best_score": "Best Score Progression",
            "diversity": "Diversity Over Episodes",
            "ood_rate": "OOD Rate and Uncertainty",
            "arm_frequency": "Arm Selection Frequency",
            "arm_quality": "Arm Performance",
        }

        for plot_key, plot_b64 in plots.items():
            title = plot_titles.get(plot_key, plot_key)
            html.append(f"<div class='plot'>\n")
            html.append(f"<h3>{title}</h3>\n")
            html.append(f"<img src='data:image/png;base64,{plot_b64}' />\n")
            html.append("</div>\n")

        # Episode table
        html.append("<h2>Episode Timeline</h2>\n")
        html.append("<table>\n")
        html.append(
            "<tr><th>Episode</th><th>Arm</th><th>Regime</th><th>Reason</th>"
            "<th>Best Score</th><th>Diversity</th><th>OOD Rate</th></tr>\n"
        )

        for ep_data in manifest["episodes"]:
            reasons = ", ".join(ep_data["reason"]) if ep_data["reason"] else "bandit"
            metrics = ep_data["metrics"]
            html.append(
                f"<tr><td>{ep_data['episode']}</td><td><code>{ep_data['arm_id']}</code></td>"
                f"<td>{ep_data['regime']}</td><td>{reasons}</td>"
                f"<td>{metrics['best_score']:.3f}</td>"
                f"<td>{metrics['internal_diversity']:.3f}</td>"
                f"<td>{metrics['ood_rate']:.3f}</td></tr>\n"
            )

        html.append("</table>\n")

        # Footer
        html.append("<hr>\n")
        html.append(
            "<p style='text-align: center; color: #7f8c8d; font-size: 12px;'>"
            "Generated by REINVENT4 Meta-Controller</p>\n"
        )

        html.append("</div>\n</body>\n</html>")

        return "".join(html)
