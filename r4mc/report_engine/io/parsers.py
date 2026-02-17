"""Parsers for different run formats."""

import pandas as pd
import json
import gzip
from pathlib import Path
from typing import List, Optional
import logging

from .model import EpisodeMolecules, EpisodeMetrics, ControllerEvent, RunData

logger = logging.getLogger(__name__)


class GenericCSVParser:
    """Parse generic CSV files with SMILES data."""

    SMILES_COLS = ['SMILES', 'smiles', 'Smiles', 'SMILE', 'smile']
    SCORE_COLS = ['Score', 'score', 'Reward', 'reward', 'Value', 'value']

    def parse_episode_molecules(self, csv_file: str, episode: int, arm: str, top_k: Optional[int] = None) -> Optional[EpisodeMolecules]:
        """Parse molecules from CSV file."""
        try:
            if csv_file.endswith('.gz'):
                df = pd.read_csv(csv_file, compression='gzip')
            else:
                df = pd.read_csv(csv_file)

            # Find SMILES column
            smiles_col = None
            for col in self.SMILES_COLS:
                if col in df.columns:
                    smiles_col = col
                    break

            if smiles_col is None:
                logger.warning(f"No SMILES column found in {csv_file}")
                return None

            # Find score column
            score_col = None
            for col in self.SCORE_COLS:
                if col in df.columns:
                    score_col = col
                    break

            if score_col is None:
                logger.warning(f"No score column found in {csv_file}")
                return None

            # Filter to top-k if requested
            if top_k and len(df) > top_k:
                df = df.nlargest(top_k, score_col)

            # Extract data
            smiles = df[smiles_col].tolist()
            scores = df[score_col].tolist()

            # Optional columns
            uncertainty = None
            if 'uncertainty' in df.columns or 'Uncertainty' in df.columns:
                unc_col = 'uncertainty' if 'uncertainty' in df.columns else 'Uncertainty'
                uncertainty = df[unc_col].tolist()

            novelty = None
            if 'novelty' in df.columns or 'Novelty' in df.columns:
                nov_col = 'novelty' if 'novelty' in df.columns else 'Novelty'
                novelty = df[nov_col].tolist()

            ood_flags = None
            if 'ood' in df.columns or 'OOD' in df.columns:
                ood_col = 'ood' if 'ood' in df.columns else 'OOD'
                ood_flags = df[ood_col].astype(bool).tolist()

            return EpisodeMolecules(
                episode=episode,
                arm=arm,
                smiles=smiles,
                scores=scores,
                uncertainty=uncertainty,
                novelty=novelty,
                ood_flags=ood_flags,
                metadata={'source_file': csv_file, 'n_rows': len(df)}
            )

        except Exception as e:
            logger.error(f"Error parsing {csv_file}: {e}")
            return None


class LogParser:
    """Parse controller log files."""

    def parse_controller_log(self, log_file: str) -> List[ControllerEvent]:
        """Parse controller decisions from log file."""
        events = []
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()

            current_episode = None
            for line in lines:
                # Look for episode markers
                if 'Episode' in line and any(x in line for x in ['1', '2', '3', '4', '5', '6', '7', '8', '9']):
                    import re
                    match = re.search(r'Episode\s+(\d+)', line)
                    if match:
                        current_episode = int(match.group(1))

                # Look for arm selection
                if current_episode and 'Selected arm:' in line:
                    parts = line.split('Selected arm:')
                    if len(parts) > 1:
                        arm_part = parts[1].strip()
                        arm_name = arm_part.split('(')[0].strip()

                        reasons = []
                        if 'Reasons:' in line:
                            reason_part = line.split('Reasons:')[1].strip()
                            reasons = [r.strip() for r in reason_part.split(',')]

                        events.append(ControllerEvent(
                            episode=current_episode,
                            arm_chosen=arm_name,
                            reasons=reasons
                        ))

        except Exception as e:
            logger.error(f"Error parsing log {log_file}: {e}")

        return events


def parse_run(discovery: dict, top_k_per_episode: int = 1000) -> RunData:
    """Parse run data from discovery."""
    run_data = RunData(
        run_name=discovery['run_name'],
        run_dir=discovery['run_dir'],
        diagnostics=discovery['diagnostics']
    )

    csv_parser = GenericCSVParser()
    log_parser = LogParser()

    # Parse episodes
    for ep_num, ep_data in sorted(discovery['episodes'].items()):
        arm = ep_data['arm']

        # Parse molecules
        for csv_file in ep_data['csv_files']:
            molecules = csv_parser.parse_episode_molecules(csv_file, ep_num, arm, top_k=top_k_per_episode)
            if molecules:
                run_data.molecules.append(molecules)
                break  # Take first valid CSV per episode

    # Parse controller events from logs
    for log_file in discovery['log_files']:
        events = log_parser.parse_controller_log(log_file)
        run_data.events.extend(events)

    logger.info(f"Parsed {len(run_data.molecules)} episode molecule sets")
    logger.info(f"Parsed {len(run_data.events)} controller events")

    return run_data
