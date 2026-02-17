"""Main report builder that orchestrates metrics and output generation."""
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from ..io.discover import RunDiscovery
from ..io.parsers import parse_run
from ..io.model import RunData
from ..metrics import stability, diversity, ood_uncertainty, reward
from ..molport.ingest import load_molport_to_dataframe
from ..molport.index import build_molport_index
from ..molport.nearest import map_run_to_molport, compute_purchasability_metrics
from ..utils.hashing import hash_directory
from ..utils.logging import setup_logger

logger = logging.getLogger(__name__)


class ReportBuilder:
    """Build behavioral reports for REINVENT4 meta-controller runs."""

    def __init__(
        self,
        run_dir: str,
        out_dir: str,
        molport_csv: Optional[str] = None,
        top_k_per_episode: int = 1000,
        molport_similarity_threshold: float = 0.7
    ):
        """
        Initialize report builder.

        Args:
            run_dir: Path to run directory
            out_dir: Path to output directory
            molport_csv: Optional path to MolPort CSV
            top_k_per_episode: Number of top molecules to keep per episode
            molport_similarity_threshold: Minimum Tanimoto similarity for MolPort mapping
        """
        self.run_dir = Path(run_dir)
        self.out_dir = Path(out_dir)
        self.molport_csv = molport_csv
        self.top_k_per_episode = top_k_per_episode
        self.molport_similarity_threshold = molport_similarity_threshold

        self.run_data: Optional[RunData] = None
        self.metrics: Dict[str, Any] = {}
        self.molport_mapping = None

    def discover_and_parse(self):
        """Discover files and parse run data."""
        logger.info(f"Discovering files in {self.run_dir}...")

        discovery = RunDiscovery(str(self.run_dir))
        discovery_result = discovery.discover()

        # Check for warnings/errors
        if discovery_result['diagnostics']['warnings']:
            logger.warning(f"Discovery warnings: {discovery_result['diagnostics']['warnings']}")
        if discovery_result['diagnostics']['errors']:
            logger.error(f"Discovery errors: {discovery_result['diagnostics']['errors']}")

        logger.info("Parsing run data...")
        self.run_data = parse_run(discovery_result, top_k_per_episode=self.top_k_per_episode)

        logger.info(f"Parsed {len(self.run_data.molecules)} episodes")

    def compute_metrics(self):
        """Compute all behavioral metrics."""
        if self.run_data is None:
            raise ValueError("Run data not loaded. Call discover_and_parse() first.")

        logger.info("Computing behavioral metrics...")

        # Stability metrics
        try:
            logger.info("Computing Top-N stability...")
            stability_df = stability.compute_top_n_stability(self.run_data, top_n=50)
            self.metrics['stability'] = {
                'top_50': stability_df,
                'summary': stability.compute_stability_summary(stability_df)
            }
        except Exception as e:
            logger.warning(f"Failed to compute stability metrics: {e}")
            self.metrics['stability'] = None

        # Diversity metrics
        try:
            logger.info("Computing diversity metrics...")
            diversity_df = diversity.compute_diversity_metrics(self.run_data)
            rediscovery_df = diversity.compute_rediscovery_metrics(self.run_data)
            self.metrics['diversity'] = {
                'diversity': diversity_df,
                'rediscovery': rediscovery_df
            }
        except Exception as e:
            logger.warning(f"Failed to compute diversity metrics: {e}")
            self.metrics['diversity'] = None

        # OOD and uncertainty metrics
        try:
            logger.info("Computing OOD and uncertainty metrics...")
            ood_df = ood_uncertainty.compute_ood_metrics(self.run_data)
            uncertainty_df = ood_uncertainty.compute_uncertainty_metrics(self.run_data)
            regime_changes = ood_uncertainty.detect_regime_changes(ood_df)
            self.metrics['ood_uncertainty'] = {
                'ood': ood_df,
                'uncertainty': uncertainty_df,
                'regime_changes': regime_changes
            }
        except Exception as e:
            logger.warning(f"Failed to compute OOD/uncertainty metrics: {e}")
            self.metrics['ood_uncertainty'] = None

        # Reward metrics
        try:
            logger.info("Computing reward metrics...")
            reward_df = reward.compute_reward_decomposition(self.run_data)
            frontier_df = reward.compute_novelty_score_frontier(self.run_data)
            arm_performance = reward.compute_arm_performance(self.run_data)
            self.metrics['reward'] = {
                'decomposition': reward_df,
                'frontier': frontier_df,
                'arm_performance': arm_performance
            }
        except Exception as e:
            logger.warning(f"Failed to compute reward metrics: {e}")
            self.metrics['reward'] = None

        logger.info("Behavioral metrics computed successfully")

    def compute_molport_mapping(self):
        """Compute MolPort purchasability mapping (optional)."""
        if self.molport_csv is None:
            logger.info("No MolPort CSV provided, skipping purchasability mapping")
            return

        if self.run_data is None:
            raise ValueError("Run data not loaded. Call discover_and_parse() first.")

        try:
            logger.info(f"Loading MolPort CSV from {self.molport_csv}...")
            molport_df = load_molport_to_dataframe(self.molport_csv, max_rows=100000)

            logger.info("Building MolPort fingerprint index...")
            cache_path = self.out_dir / "cache" / "molport_index.pkl"
            molport_index = build_molport_index(molport_df, cache_path=str(cache_path))

            logger.info("Mapping molecules to MolPort...")
            self.molport_mapping = map_run_to_molport(
                self.run_data,
                molport_index,
                min_similarity=self.molport_similarity_threshold
            )

            # Compute purchasability metrics
            purchasability_df = compute_purchasability_metrics(self.run_data, self.molport_mapping)
            self.metrics['purchasability'] = {
                'mapping': self.molport_mapping,
                'per_episode': purchasability_df
            }

            logger.info(f"MolPort mapping complete: {len(self.molport_mapping)} matches found")

        except Exception as e:
            logger.error(f"Failed to compute MolPort mapping: {e}")
            self.metrics['purchasability'] = None

    def save_tables(self):
        """Save metric tables to CSV files."""
        tables_dir = self.out_dir / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving tables to {tables_dir}...")

        for metric_name, metric_data in self.metrics.items():
            if metric_data is None:
                continue

            try:
                if isinstance(metric_data, dict):
                    for sub_name, df in metric_data.items():
                        if df is not None and hasattr(df, 'to_csv'):
                            filename = f"{metric_name}_{sub_name}.csv"
                            df.to_csv(tables_dir / filename, index=False)
                            logger.debug(f"Saved {filename}")
            except Exception as e:
                logger.warning(f"Failed to save table for {metric_name}: {e}")

    def generate_manifest(self):
        """Generate manifest with run metadata and file hashes."""
        manifest = {
            'timestamp': datetime.now().isoformat(),
            'run_dir': str(self.run_dir),
            'run_name': self.run_data.run_name if self.run_data else None,
            'n_episodes': len(self.run_data.molecules) if self.run_data else 0,
            'top_k_per_episode': self.top_k_per_episode,
            'molport_csv': self.molport_csv,
            'molport_similarity_threshold': self.molport_similarity_threshold,
            'metrics_computed': list(self.metrics.keys()),
            'diagnostics': self.run_data.diagnostics if self.run_data else {}
        }

        manifest_path = self.out_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Manifest saved to {manifest_path}")

    def generate_markdown_report(self):
        """Generate markdown report."""
        md_lines = [
            f"# REINVENT4 Meta-Controller Run Report",
            f"",
            f"**Run:** {self.run_data.run_name}",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Episodes:** {len(self.run_data.molecules)}",
            f"",
            f"## Summary",
            f"",
        ]

        # Add metrics summaries
        if self.metrics.get('stability'):
            summary = self.metrics['stability']['summary']
            md_lines.extend([
                f"### Top-N Stability",
                f"- Mean Jaccard similarity: {summary['mean_jaccard']:.3f}",
                f"- Median Jaccard similarity: {summary['median_jaccard']:.3f}",
                f""
            ])

        if self.metrics.get('reward'):
            arm_perf = self.metrics['reward']['arm_performance']
            md_lines.extend([
                f"### Arm Performance",
                f""
            ])
            for _, row in arm_perf.iterrows():
                md_lines.append(f"- **{row['arm']}**: {row['n_episodes']} episodes, "
                                f"mean score: {row['score_mean']:.3f}")
            md_lines.append("")

        if self.metrics.get('purchasability'):
            purch = self.metrics['purchasability']['per_episode']
            mean_rate = purch['purchasability_rate'].mean()
            md_lines.extend([
                f"### MolPort Purchasability",
                f"- Mean purchasability rate: {mean_rate:.1%}",
                f"- Total purchasable molecules: {self.molport_mapping['query_smiles'].nunique() if self.molport_mapping is not None else 0}",
                f""
            ])

        report_md_path = self.out_dir / "report.md"
        with open(report_md_path, 'w') as f:
            f.write("\n".join(md_lines))

        logger.info(f"Markdown report saved to {report_md_path}")

    def build(self):
        """Build complete report."""
        logger.info("Building report...")

        # Create output directory
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "cache").mkdir(exist_ok=True)
        (self.out_dir / "figures").mkdir(exist_ok=True)
        (self.out_dir / "tables").mkdir(exist_ok=True)
        (self.out_dir / "diagnostics").mkdir(exist_ok=True)

        # Step 1: Discover and parse
        self.discover_and_parse()

        # Step 2: Compute metrics
        self.compute_metrics()

        # Step 3: Compute MolPort mapping (optional)
        if self.molport_csv:
            self.compute_molport_mapping()

        # Step 4: Save outputs
        self.save_tables()
        self.generate_markdown_report()
        self.generate_manifest()

        logger.info(f"Report generation complete. Output saved to {self.out_dir}")

        return self.out_dir
