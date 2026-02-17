"""Episode execution and REINVENT4 integration."""

import logging
import random
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from jinja2 import Environment, FileSystemLoader, Template

from .arms import ArmConfig
from .config import ControllerConfig
from .seeds import SeedRecord, SeedSelector
from .state import SeedRecord as SeedRecordState

logger = logging.getLogger(__name__)


class EpisodeRunner:
    """Handles running a single REINVENT4 episode."""

    def __init__(self, config: ControllerConfig, dry_run: bool = False, lite: bool = False):
        self.config = config
        self.dry_run = dry_run
        self.lite = lite

        # Setup Jinja2 for template rendering
        template_dir = Path(__file__).parent / "templates"
        self.jinja_env = Environment(loader=FileSystemLoader(str(template_dir)))

    def run_episode(
        self,
        arm: ArmConfig,
        episode_num: int,
        seed_bank: List[SeedRecordState],
        checkpoint_path: Optional[Path] = None,
    ) -> Tuple[Path, bool, Optional[str]]:
        """
        Run a single episode.

        Args:
            arm: Arm configuration
            episode_num: Episode number
            seed_bank: Current seed bank
            checkpoint_path: Path to checkpoint to resume from

        Returns:
            - Episode output directory
            - Success flag
            - Error message (if failed)
        """
        # Create episode directory
        episode_dir = (
            self.config.output_dir
            / self.config.campaign_name
            / arm.arm_id
            / f"episode_{episode_num:03d}"
        )
        episode_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Running episode {episode_num} with arm '{arm.arm_id}' in {episode_dir}")

        # Check if this is a mol2mol arm that should use batched processing
        use_batched = arm.requires_seeds and "mol2mol" in arm.arm_id.lower()

        if use_batched:
            logger.info("Using batched seed processing for mol2mol arm")
            return self._run_batched_mol2mol(arm, episode_num, seed_bank, episode_dir)

        # Prepare seeds if required (non-batched)
        seed_file = None
        if arm.requires_seeds:
            seed_file = episode_dir / "seeds.smi"
            selector = SeedSelector(seed_bank, random_seed=self.config.seed + episode_num)
            # Reduce seeds in lite mode to avoid memory issues with Mol2Mol
            n_seeds = 10 if self.lite else 50
            selected_seeds = selector.select_seeds_for_arm(arm, n_seeds=n_seeds, episode=episode_num)

            if not selected_seeds:
                error_msg = f"No seeds available for arm '{arm.arm_id}'"
                logger.error(error_msg)
                return episode_dir, False, error_msg

            selector.export_seeds_to_file(selected_seeds, seed_file)

        # Render REINVENT4 config
        config_file = episode_dir / "reinvent_config.toml"
        try:
            self._render_config(
                arm, config_file, episode_dir, seed_file, checkpoint_path
            )
        except Exception as e:
            error_msg = f"Failed to render config: {e}"
            logger.error(error_msg)
            return episode_dir, False, error_msg

        # Run REINVENT4 (or dry-run simulation)
        if self.dry_run:
            logger.info("DRY RUN: Simulating REINVENT4 execution")
            success, error = self._simulate_reinvent_run(episode_dir)
        else:
            success, error = self._execute_reinvent(config_file, episode_dir)

        return episode_dir, success, error

    def _render_config(
        self,
        arm: ArmConfig,
        output_path: Path,
        episode_dir: Path,
        seed_file: Optional[Path],
        checkpoint_path: Optional[Path],
    ) -> None:
        """Render REINVENT4 TOML config from template."""
        template = self.jinja_env.get_template(arm.template_name)

        # Prepare template variables
        prior_path = self.config.priors_dir / arm.prior_filename

        # Episode settings (use arm overrides if present)
        steps = arm.episode_steps or self.config.episode.steps
        batch_size = arm.episode_batch_size or self.config.episode.batch_size

        # Further reduce batch size for mol2mol in lite mode to avoid OOM
        if self.lite and "mol2mol" in arm.arm_id.lower():
            batch_size = min(batch_size, 32)

        template_vars = {
            "prior_path": str(prior_path.absolute()),
            "output_dir": str(episode_dir.absolute()),
            "seed_file": str(seed_file.absolute()) if seed_file else None,
            "checkpoint_path": str(checkpoint_path.absolute()) if checkpoint_path else None,
            "steps": steps,
            "batch_size": batch_size,
            "learning_rate": self.config.episode.learning_rate,
            "sigma": self.config.episode.sigma,
            "randomize_smiles": self.config.episode.randomize_smiles,
            # Scoring
            "potency_weight": arm.scoring_profile.potency_weight,
            "novelty_weight": arm.scoring_profile.novelty_weight,
            "similarity_weight": arm.scoring_profile.similarity_weight,
            "uncertainty_penalty": arm.scoring_profile.uncertainty_penalty,
            "diversity_weight": arm.scoring_profile.diversity_weight,
            # Diversity
            "bucket_similarity": arm.diversity_settings.bucket_similarity,
            "penalty_strength": arm.diversity_settings.penalty_strength,
            "use_scaffold_memory": arm.diversity_settings.use_scaffold_memory,
            # Diversity filter params for REINVENT4
            "diversity_bucket_size": int(arm.diversity_settings.bucket_similarity * 100),  # Convert to count
            "diversity_minscore": 0.4,
            "diversity_minsimilarity": arm.diversity_settings.bucket_similarity,
            "diversity_penalty": arm.diversity_settings.penalty_strength,
            # Output and other settings
            "output_csv_prefix": self.config.output_csv_name.replace(".csv", ""),
            "max_score": 1.0,
            "max_steps": steps,
            "arm_name": arm.arm_id,
            # Properties
            "mw_min": self.config.properties.mw_min,
            "mw_max": self.config.properties.mw_max,
            "logp_min": self.config.properties.logp_min,
            "logp_max": self.config.properties.logp_max,
            "tpsa_max": self.config.properties.tpsa_max,
            "hbd_max": self.config.properties.hbd_max,
            "hba_max": self.config.properties.hba_max,
            # QSAR model path (for QSAR arms) - absolute paths required
            "qsar_model_path": str((self.config.output_dir.parent / "models/brd4_qsar/brd4_qsar_model.pkl").absolute()),
            "qsar_scorer_path": str((self.config.output_dir.parent / "scripts/qsar_scorer.py").absolute()),
        }

        # Render
        rendered = template.render(**template_vars)

        # Write
        with open(output_path, "w") as f:
            f.write(rendered)

        logger.info(f"Rendered config to {output_path}")

    def _execute_reinvent(
        self, config_file: Path, episode_dir: Path
    ) -> Tuple[bool, Optional[str]]:
        """Execute REINVENT4 via subprocess."""
        import shlex

        # Determine REINVENT4 binary
        if self.config.reinvent_bin:
            reinvent_cmd = str(self.config.reinvent_bin)
        else:
            # Try to find in PATH
            reinvent_cmd = "reinvent"

        # Build command - handle space-separated commands like "python /path/to/script.py"
        # Use just the filename since we're running from episode_dir
        config_filename = config_file.name
        if ' ' in reinvent_cmd:
            cmd = shlex.split(reinvent_cmd) + [config_filename]
        else:
            cmd = [reinvent_cmd, config_filename]

        # Execute
        log_file = episode_dir / "reinvent.log"
        try:
            with open(log_file, "w") as log_f:
                result = subprocess.run(
                    cmd,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    cwd=episode_dir,
                    timeout=3600,  # 1 hour timeout
                )

            if result.returncode != 0:
                error_msg = f"REINVENT4 failed with return code {result.returncode}"
                logger.error(error_msg)
                return False, error_msg

            logger.info("REINVENT4 completed successfully")
            return True, None

        except FileNotFoundError:
            error_msg = (
                f"REINVENT4 binary not found: {reinvent_cmd}. "
                "Please specify reinvent_bin in config or ensure REINVENT4 is in PATH."
            )
            logger.error(error_msg)
            return False, error_msg

        except subprocess.TimeoutExpired:
            error_msg = "REINVENT4 execution timed out"
            logger.error(error_msg)
            return False, error_msg

        except Exception as e:
            error_msg = f"REINVENT4 execution failed: {e}"
            logger.error(error_msg)
            return False, error_msg

    def _simulate_reinvent_run(self, episode_dir: Path) -> Tuple[bool, Optional[str]]:
        """
        Simulate REINVENT4 run for dry-run mode.

        Generates a realistic dummy output CSV.
        """
        output_file = episode_dir / self.config.output_csv_name

        # Generate fake molecules
        fake_smiles = self._generate_fake_molecules(100)
        fake_scores = [random.uniform(0.3, 0.9) for _ in fake_smiles]

        # Create DataFrame
        df = pd.DataFrame(
            {
                self.config.output_csv_smiles_col: fake_smiles,
                self.config.output_csv_score_col: fake_scores,
            }
        )

        # Save
        df.to_csv(output_file, index=False)

        logger.info(f"Generated dry-run output: {output_file}")
        return True, None

    def _generate_fake_molecules(self, n: int) -> List[str]:
        """Generate fake but valid SMILES for dry-run."""
        # Simple fake SMILES patterns
        templates = [
            "c1ccccc1",  # benzene
            "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # ibuprofen-like
            "CC(C)NCC(COc1ccccc1)O",  # propranolol-like
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # caffeine
            "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
        ]

        fake_smiles = []
        for i in range(n):
            # Pick random template and add some variation
            base = random.choice(templates)
            # Simple variation: add/remove methyls or modify
            if random.random() > 0.5:
                base = base.replace("C", "CC", 1)
            fake_smiles.append(base)

        return fake_smiles

    def _run_batched_mol2mol(
        self,
        arm: ArmConfig,
        episode_num: int,
        seed_bank: List[SeedRecordState],
        episode_dir: Path,
    ) -> Tuple[Path, bool, Optional[str]]:
        """
        Run mol2mol with batched seed processing to avoid OOM.

        Seeds are split into batches, each batch is run separately,
        then results are aggregated.
        """
        # Select seeds
        selector = SeedSelector(seed_bank, random_seed=self.config.seed + episode_num)
        n_seeds = 30 if self.lite else 50  # More seeds than simple approach, batched
        selected_seeds = selector.select_seeds_for_arm(arm, n_seeds=n_seeds, episode=episode_num)

        if not selected_seeds:
            error_msg = f"No seeds available for arm '{arm.arm_id}'"
            logger.error(error_msg)
            return episode_dir, False, error_msg

        # Determine batch size
        batch_size = 10 if self.lite else 15
        num_batches = (len(selected_seeds) + batch_size - 1) // batch_size

        logger.info(f"Processing {len(selected_seeds)} seeds in {num_batches} batches of ~{batch_size}")

        all_results = []

        # Process each batch
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(selected_seeds))
            batch_seeds = selected_seeds[start_idx:end_idx]

            logger.info(f"Batch {batch_idx + 1}/{num_batches}: Processing {len(batch_seeds)} seeds")

            # Create batch subdirectory
            batch_dir = episode_dir / f"batch_{batch_idx}"
            batch_dir.mkdir(parents=True, exist_ok=True)

            # Write seeds for this batch
            batch_seed_file = batch_dir / "seeds.smi"
            selector.export_seeds_to_file(batch_seeds, batch_seed_file)

            # Render config for this batch
            batch_config_file = batch_dir / "reinvent_config.toml"
            try:
                self._render_config(
                    arm, batch_config_file, batch_dir, batch_seed_file, None
                )
            except Exception as e:
                logger.error(f"Batch {batch_idx + 1} config render failed: {e}")
                continue

            # Run REINVENT4 for this batch
            if self.dry_run:
                success, error = self._simulate_reinvent_run(batch_dir)
            else:
                success, error = self._execute_reinvent(batch_config_file, batch_dir)

            if not success:
                logger.error(f"Batch {batch_idx + 1} failed: {error}")
                continue

            # Parse batch results
            base_name = self.config.output_csv_name.replace('.csv', '')
            batch_results_file = batch_dir / f"{base_name}_1.csv"

            if not batch_results_file.exists():
                batch_results_file = batch_dir / self.config.output_csv_name

            if batch_results_file.exists():
                try:
                    df = pd.read_csv(batch_results_file)
                    df['batch'] = batch_idx
                    df['batch_seed_idx'] = range(len(df))
                    all_results.append(df)
                    logger.info(f"Batch {batch_idx + 1}: Generated {len(df)} molecules")
                except Exception as e:
                    logger.error(f"Batch {batch_idx + 1} parsing failed: {e}")

        # Aggregate results
        if not all_results:
            error_msg = "No batches succeeded - all batches failed"
            logger.error(error_msg)
            return episode_dir, False, error_msg

        # Combine all batch results
        combined_df = pd.concat(all_results, ignore_index=True)

        # Save combined results in episode root
        output_file = episode_dir / f"{self.config.output_csv_name.replace('.csv', '')}_1.csv"
        combined_df.to_csv(output_file, index=False)

        logger.info(f"Batched processing complete: {len(combined_df)} total molecules from {len(all_results)} batches")
        logger.info(f"  Unique SMILES: {combined_df['SMILES'].nunique()}")
        logger.info(f"  Score range: {combined_df['Score'].min():.3f} - {combined_df['Score'].max():.3f}")

        return episode_dir, True, None

    def parse_episode_output(self, episode_dir: Path) -> Tuple[List[str], List[float]]:
        """
        Parse REINVENT4 output to extract generated molecules and scores.

        Returns:
            - List of SMILES
            - List of scores
        """
        # REINVENT4 appends stage number to output file (e.g., results_1.csv)
        base_name = self.config.output_csv_name.replace('.csv', '')
        output_file = episode_dir / f"{base_name}_1.csv"

        # Fallback to exact name if stage-numbered file doesn't exist
        if not output_file.exists():
            output_file = episode_dir / self.config.output_csv_name

        if not output_file.exists():
            logger.error(f"Output file not found: {output_file}")
            return [], []

        try:
            df = pd.read_csv(output_file)

            smiles_col = self.config.output_csv_smiles_col
            score_col = self.config.output_csv_score_col

            if smiles_col not in df.columns or score_col not in df.columns:
                logger.error(
                    f"Required columns not found. Expected: {smiles_col}, {score_col}. "
                    f"Found: {df.columns.tolist()}"
                )
                return [], []

            smiles = df[smiles_col].tolist()
            scores = df[score_col].tolist()

            logger.info(f"Parsed {len(smiles)} molecules from {output_file}")
            return smiles, scores

        except Exception as e:
            logger.error(f"Failed to parse output file: {e}")
            return [], []

    def find_latest_checkpoint(self, arm: ArmConfig) -> Optional[Path]:
        """Find the latest checkpoint for an arm."""
        arm_dir = self.config.output_dir / self.config.campaign_name / arm.arm_id

        if not arm_dir.exists():
            return None

        # Look for checkpoint files
        checkpoints = list(arm_dir.glob("**/checkpoint*.chkpt"))
        if not checkpoints:
            checkpoints = list(arm_dir.glob("**/model.chkpt"))

        if checkpoints:
            # Return most recent
            latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
            logger.info(f"Found checkpoint for '{arm.arm_id}': {latest}")
            return latest

        return None
