"""Run directory discovery and validation."""

import os
import glob
from pathlib import Path
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class RunDiscovery:
    """Discover files in a run directory."""

    def __init__(self, run_dir: str):
        self.run_dir = Path(run_dir)
        if not self.run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")

    def discover(self) -> Dict[str, Any]:
        """Discover all relevant files."""
        discovery = {
            'run_dir': str(self.run_dir),
            'run_name': self.run_dir.name,
            'episodes': {},
            'config_files': [],
            'log_files': [],
            'seed_banks': [],
            'diagnostics': {
                'warnings': [],
                'errors': []
            }
        }

        # Find episode directories
        episode_dirs = sorted(self.run_dir.glob('*/episode_*'))
        if not episode_dirs:
            # Try alternative structure
            episode_dirs = sorted(self.run_dir.glob('episode_*'))

        if not episode_dirs:
            discovery['diagnostics']['warnings'].append(
                'No episode directories found. Expected pattern: */episode_* or episode_*'
            )

        for ep_dir in episode_dirs:
            ep_num = self._extract_episode_number(ep_dir.name)
            if ep_num is None:
                continue

            arm_name = self._extract_arm_name(ep_dir)

            discovery['episodes'][ep_num] = {
                'dir': str(ep_dir),
                'arm': arm_name,
                'csv_files': [str(f) for f in ep_dir.glob('*.csv*')],
                'json_files': [str(f) for f in ep_dir.glob('*.json')],
                'config_files': [str(f) for f in ep_dir.glob('*.toml')] +
                               [str(f) for f in ep_dir.glob('*.yaml')]
            }

        # Find global config files
        for pattern in ['*.yaml', '*.toml', '*.json']:
            discovery['config_files'].extend([str(f) for f in self.run_dir.glob(pattern)])

        # Find log files
        for pattern in ['*.log', '*.txt']:
            discovery['log_files'].extend([str(f) for f in self.run_dir.glob(pattern)])

        # Find seed bank files
        seed_patterns = ['*seed*.json', '*seed*.csv', 'seed_bank/*']
        for pattern in seed_patterns:
            discovery['seed_banks'].extend([str(f) for f in self.run_dir.rglob(pattern)])

        logger.info(f"Discovered {len(discovery['episodes'])} episodes")
        return discovery

    def _extract_episode_number(self, name: str) -> int:
        """Extract episode number from directory name."""
        import re
        match = re.search(r'episode[_-]?(\d+)', name, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return None

    def _extract_arm_name(self, ep_dir: Path) -> str:
        """Extract arm name from episode directory path."""
        parts = ep_dir.parts
        # Look for arm name in parent directory
        if len(parts) > 1:
            parent = parts[-2]
            if 'episode' not in parent.lower():
                return parent
        return 'unknown'

    def validate_structure(self, discovery: Dict[str, Any]) -> Dict[str, Any]:
        """Validate discovered structure and return diagnostics."""
        diag = discovery['diagnostics']

        if not discovery['episodes']:
            diag['errors'].append('No episodes found')

        # Check for common files
        has_molecules = False
        for ep_data in discovery['episodes'].values():
            if ep_data['csv_files']:
                has_molecules = True
                break

        if not has_molecules:
            diag['warnings'].append('No CSV files found in episodes. Molecule data may be missing.')

        if not discovery['config_files']:
            diag['warnings'].append('No configuration files found at run root.')

        return diag
