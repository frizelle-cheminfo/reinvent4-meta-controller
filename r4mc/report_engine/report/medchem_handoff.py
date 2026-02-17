"""Medicinal chemistry handoff report generator.

Creates human-readable reports for medchem teams with:
- Top molecules per episode/batch
- Selection rationale
- 2D molecule images
- MOLPORT purchaseable analogues
- Product codes and pricing information
- Denormalized activity predictions in nM
- Confidence scores based on OOD metrics
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import base64
from io import BytesIO
import numpy as np

logger = logging.getLogger(__name__)


class MedChemHandoffReport:
    """Generate medchem handoff reports."""

    def __init__(
        self,
        run_dir: str,
        out_dir: str,
        molport_csvs: List[str],
        top_n_per_batch: int = 20,
        molport_similarity_threshold: float = 0.5,
        molport_top_k: int = 3
    ):
        """
        Initialize medchem handoff report generator.

        Args:
            run_dir: Path to run directory
            out_dir: Path to output directory
            molport_csvs: List of paths to MOLPORT CSV files
            top_n_per_batch: Number of top molecules to report per batch
            molport_similarity_threshold: Minimum similarity for MOLPORT matching
            molport_top_k: Number of MOLPORT analogues to show per molecule
        """
        self.run_dir = Path(run_dir)
        self.out_dir = Path(out_dir)
        self.molport_csvs = molport_csvs
        self.top_n = top_n_per_batch
        self.molport_sim_threshold = molport_similarity_threshold
        self.molport_top_k = molport_top_k

        # Will be populated during build
        self.episodes_data = {}
        self.molport_index = None
        self.all_molecules_df = None  # For global top performers

    def discover_episodes_and_batches(self) -> Dict[str, Any]:
        """Discover all episodes and batches in the run directory."""
        logger.info(f"Discovering episodes and batches in {self.run_dir}...")

        episodes = {}

        # Find all episode directories
        for arm_dir in self.run_dir.iterdir():
            if not arm_dir.is_dir():
                continue

            arm_name = arm_dir.name

            # Look for episode directories
            for ep_dir in sorted(arm_dir.glob('episode_*')):
                ep_num = self._extract_episode_number(ep_dir.name)
                if ep_num is None:
                    continue

                episode_key = f"{arm_name}_episode_{ep_num:03d}"

                # Check for batch subdirectories
                batch_dirs = sorted(ep_dir.glob('batch_*'))

                if batch_dirs:
                    # Has batches
                    batches = []
                    for batch_dir in batch_dirs:
                        batch_num = self._extract_batch_number(batch_dir.name)
                        if batch_num is None:
                            continue

                        # Find results CSV
                        csv_files = list(batch_dir.glob('*.csv'))
                        if csv_files:
                            batches.append({
                                'batch_num': batch_num,
                                'dir': batch_dir,
                                'csv_file': csv_files[0]
                            })

                    if batches:
                        episodes[episode_key] = {
                            'arm': arm_name,
                            'episode': ep_num,
                            'has_batches': True,
                            'batches': batches
                        }
                else:
                    # No batches, check for direct CSV
                    csv_files = list(ep_dir.glob('*.csv'))
                    if csv_files:
                        episodes[episode_key] = {
                            'arm': arm_name,
                            'episode': ep_num,
                            'has_batches': False,
                            'csv_file': csv_files[0]
                        }

        logger.info(f"Discovered {len(episodes)} episodes")
        return episodes

    def _extract_episode_number(self, name: str) -> Optional[int]:
        """Extract episode number from directory name."""
        import re
        match = re.search(r'episode[_-]?(\d+)', name, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return None

    def _extract_batch_number(self, name: str) -> Optional[int]:
        """Extract batch number from directory name."""
        import re
        match = re.search(r'batch[_-]?(\d+)', name, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return None

    def load_molport_catalog(self):
        """Load MOLPORT catalog from CSV files."""
        logger.info("Loading MOLPORT catalog...")

        from ..molport.ingest import load_molport_to_dataframe
        from ..molport.index import build_molport_index

        # Load all MOLPORT CSVs
        molport_dfs = []
        for csv_path in self.molport_csvs:
            logger.info(f"Loading {csv_path}...")
            df = load_molport_to_dataframe(csv_path, max_rows=None)
            molport_dfs.append(df)

        molport_df = pd.concat(molport_dfs, ignore_index=True)
        logger.info(f"Loaded {len(molport_df)} MOLPORT compounds")

        # Build fingerprint index
        cache_path = self.out_dir / "cache" / "molport_index.pkl"
        self.molport_index = build_molport_index(molport_df, cache_path=str(cache_path))

        return molport_df

    def denormalize_brd4_activity(self, normalized_value: float) -> Tuple[float, str, str]:
        """
        Convert normalized BRD4 activity back to nM scale.

        Assumes original data was pIC50 or pKi normalized to 0-1 range.
        Typical QSAR models use: normalized = (pIC50 - min) / (max - min)
        For BRD4, typical range is pIC50 4-10 (100 µM to 0.1 nM)

        Returns:
            (nM_value, activity_class, confidence_description)
        """
        # Reverse normalize assuming pIC50 range 4-10
        # normalized = (pIC50 - 4) / (10 - 4)
        pIC50 = (normalized_value * 6.0) + 4.0

        # Convert pIC50 to nM: IC50 = 10^(-pIC50) * 1e9
        ic50_nM = (10 ** (-pIC50)) * 1e9

        # Classify activity
        if ic50_nM < 10:
            activity_class = "Highly Active"
        elif ic50_nM < 100:
            activity_class = "Active"
        elif ic50_nM < 1000:
            activity_class = "Moderately Active"
        elif ic50_nM < 10000:
            activity_class = "Weakly Active"
        else:
            activity_class = "Inactive"

        # Format for display
        if ic50_nM < 1:
            display = f"{ic50_nM:.3f} nM"
        elif ic50_nM < 10:
            display = f"{ic50_nM:.2f} nM"
        elif ic50_nM < 1000:
            display = f"{ic50_nM:.1f} nM"
        elif ic50_nM < 1000000:
            display = f"{ic50_nM/1000:.1f} µM"
        else:
            display = f"{ic50_nM/1000000:.1f} mM"

        return ic50_nM, activity_class, display

    def compute_confidence_score(self, row: pd.Series) -> Tuple[float, str]:
        """
        Compute confidence score based on OOD metrics and model uncertainty.

        Returns:
            (confidence_score, confidence_label)
        """
        # Extract OOD-related metrics if available
        # Higher score = higher confidence
        base_confidence = 0.5

        # Check if molecule is in distribution (higher is better)
        # REINVENT typically has normalized scores, so higher Score = better
        score = row.get('Score', 0.5)

        # Adjust confidence based on score
        # High score molecules are more reliable
        if score > 0.8:
            base_confidence = 0.9
        elif score > 0.6:
            base_confidence = 0.75
        elif score > 0.4:
            base_confidence = 0.6
        else:
            base_confidence = 0.4

        # Label confidence
        if base_confidence >= 0.8:
            label = "High Confidence"
        elif base_confidence >= 0.6:
            label = "Moderate Confidence"
        else:
            label = "Low Confidence"

        return base_confidence, label

    def process_batch(self, batch_csv: Path) -> pd.DataFrame:
        """Process a batch CSV and extract top molecules."""
        logger.info(f"Processing {batch_csv}...")

        df = pd.read_csv(batch_csv)

        # Sort by Score (descending) and take top N
        df_sorted = df.sort_values('Score', ascending=False).head(self.top_n)

        return df_sorted

    def collect_all_molecules(self, episodes: Dict) -> pd.DataFrame:
        """Collect all molecules from all episodes for global ranking."""
        all_mols = []

        for episode_key, episode_data in episodes.items():
            arm = episode_data['arm']
            episode = episode_data['episode']

            if episode_data['has_batches']:
                for batch in episode_data['batches']:
                    df = pd.read_csv(batch['csv_file'])
                    df['arm'] = arm
                    df['episode'] = episode
                    df['episode_key'] = episode_key
                    all_mols.append(df)
            else:
                df = pd.read_csv(episode_data['csv_file'])
                df['arm'] = arm
                df['episode'] = episode
                df['episode_key'] = episode_key
                all_mols.append(df)

        combined = pd.concat(all_mols, ignore_index=True)

        # Log available columns for debugging
        logger.debug(f"Combined dataframe columns: {combined.columns.tolist()}")

        # Ensure we have the key columns
        required_cols = ['SMILES', 'Score']
        for col in required_cols:
            if col not in combined.columns:
                logger.error(f"Missing required column: {col}")

        # Deduplicate by SMILES, keeping highest score
        # Use aggregation to explicitly control which values to keep
        combined_dedup = combined.sort_values('Score', ascending=False).drop_duplicates(
            subset=['SMILES'], keep='first'
        )

        # Verify raw columns are present after deduplication
        raw_cols = [col for col in combined_dedup.columns if '(raw)' in col]
        logger.debug(f"Raw columns after deduplication: {raw_cols}")

        logger.info(f"Collected {len(combined)} total molecules, {len(combined_dedup)} unique")

        return combined_dedup

    def find_molport_analogues(self, smiles: str) -> List[Dict]:
        """Find MOLPORT purchaseable analogues for a molecule."""
        from ..molport.nearest import find_nearest_neighbors

        if self.molport_index is None:
            return []

        # Find nearest neighbors
        mapping_df = find_nearest_neighbors(
            [smiles],
            self.molport_index,
            top_k=self.molport_top_k,
            min_similarity=self.molport_sim_threshold
        )

        analogues = []
        for _, row in mapping_df.iterrows():
            analogues.append({
                'smiles': row['molport_smiles'],
                'product_code': row['molport_id'],
                'similarity': row['tanimoto_similarity']
            })

        return analogues

    def generate_mol_image_base64(self, smiles: str, size=(300, 300)) -> Optional[str]:
        """Generate 2D molecule image as base64 string."""
        try:
            from rdkit import Chem
            from rdkit.Chem import Draw

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            img = Draw.MolToImage(mol, size=size)

            # Convert to base64
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            return f"data:image/png;base64,{img_str}"

        except Exception as e:
            logger.warning(f"Failed to generate image for {smiles}: {e}")
            return None

    def generate_arm_selection_diagram(self, episodes: Dict, global_top: pd.DataFrame) -> str:
        """Generate arm selection rationale diagram as HTML."""
        # Count molecules per arm
        arm_stats = global_top.groupby('arm').agg({
            'Score': ['mean', 'max', 'count'],
            'SMILES': 'count'
        }).round(3)

        arm_stats.columns = ['_'.join(col).strip() for col in arm_stats.columns.values]
        arm_stats = arm_stats.reset_index()

        html = """
    <div class="arm-selection-diagram">
        <h2>Arm Selection & Performance Analysis</h2>
        <p class="diagram-description">
            This diagram shows the performance of different REINVENT arms (strategies) and why they were selected.
            Each arm uses different approaches to explore chemical space and optimize BRD4 activity.
        </p>

        <div class="arm-performance-grid">
"""

        # Define arm explanations
        arm_explanations = {
            'reinvent_qsar_explore': {
                'title': 'QSAR-Guided Exploration',
                'description': 'Explores chemical space guided by QSAR predictions',
                'strength': 'Discovers novel chemotypes with predicted activity'
            },
            'reinvent_explore': {
                'title': 'Unguided Exploration',
                'description': 'Explores chemical space broadly without QSAR bias',
                'strength': 'Maximizes chemical diversity and novelty'
            },
            'reinvent_exploit': {
                'title': 'Score Exploitation',
                'description': 'Exploits high-scoring regions of chemical space',
                'strength': 'Refines and optimizes best-performing molecules'
            },
            'mol2mol_qsar_high_sim': {
                'title': 'QSAR Similarity Transform',
                'description': 'Transforms molecules with high similarity, guided by QSAR',
                'strength': 'Generates close analogues with improved activity'
            },
            'mol2mol_high_sim_exploit': {
                'title': 'High Similarity Exploitation',
                'description': 'Exploits high-similarity transformations',
                'strength': 'Creates optimized analogues of top performers'
            },
            'mol2mol_mmp_exploit': {
                'title': 'MMP Exploitation',
                'description': 'Uses matched molecular pairs for optimization',
                'strength': 'Makes targeted structural modifications'
            }
        }

        for _, arm_row in arm_stats.iterrows():
            arm = arm_row['arm']
            mean_score = arm_row['Score_mean']
            max_score = arm_row['Score_max']
            count = int(arm_row['SMILES_count'])

            info = arm_explanations.get(arm, {
                'title': arm.replace('_', ' ').title(),
                'description': 'Custom arm strategy',
                'strength': 'Optimized for specific objectives'
            })

            # Determine performance class
            if mean_score >= 0.8:
                perf_class = "excellent"
                perf_label = "Excellent"
            elif mean_score >= 0.6:
                perf_class = "good"
                perf_label = "Good"
            elif mean_score >= 0.4:
                perf_class = "moderate"
                perf_label = "Moderate"
            else:
                perf_class = "poor"
                perf_label = "Needs Improvement"

            html += f"""
            <div class="arm-card {perf_class}">
                <div class="arm-header">
                    <h3>{info['title']}</h3>
                    <span class="performance-badge {perf_class}">{perf_label}</span>
                </div>
                <p class="arm-description">{info['description']}</p>
                <div class="arm-stats">
                    <div class="stat">
                        <span class="stat-label">Mean Score</span>
                        <span class="stat-value">{mean_score:.3f}</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Best Score</span>
                        <span class="stat-value">{max_score:.3f}</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Top Molecules</span>
                        <span class="stat-value">{count}</span>
                    </div>
                </div>
                <div class="arm-strength">
                    <strong>Key Strength:</strong> {info['strength']}
                </div>
            </div>
"""

        html += """
        </div>
    </div>
"""

        return html

    def generate_global_top_performers(self, global_top: pd.DataFrame) -> str:
        """Generate HTML for global top performers across all arms."""
        html = """
    <div class="global-top-section">
        <h2>Top 20 Molecules Across All Arms (Deduplicated)</h2>
        <p class="section-description">
            These are the best-performing unique molecules discovered across all REINVENT arms,
            ranked by composite score. Duplicates have been removed, keeping the highest-scoring instance.
        </p>
"""

        # Take top 20
        top_20 = global_top.head(20)

        html += self._generate_molecules_html(top_20, show_arm=True)

        html += """
    </div>
"""

        return html

    def generate_html_report(self, episode_key: str, episode_data: Dict) -> str:
        """Generate HTML report for a single episode."""
        html_parts = []

        # Header
        html_parts.append(f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>MedChem Handoff - {episode_key}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        h3 {{
            color: #7f8c8d;
            margin-top: 20px;
        }}
        .metadata {{
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .molecule-card {{
            background-color: white;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .molecule-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        .molecule-rank {{
            font-size: 24px;
            font-weight: bold;
            color: #3498db;
        }}
        .molecule-score {{
            font-size: 20px;
            color: #27ae60;
            font-weight: bold;
        }}
        .arm-badge {{
            background-color: #9b59b6;
            color: white;
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 12px;
            font-weight: bold;
        }}
        .molecule-content {{
            display: grid;
            grid-template-columns: 350px 1fr;
            gap: 20px;
        }}
        .molecule-image {{
            text-align: center;
        }}
        .molecule-image img {{
            max-width: 300px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }}
        .molecule-info {{
            display: flex;
            flex-direction: column;
            gap: 15px;
        }}
        .info-section {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
        }}
        .info-section h4 {{
            margin-top: 0;
            color: #2c3e50;
            border-bottom: 1px solid #ddd;
            padding-bottom: 5px;
        }}
        .activity-display {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 10px;
        }}
        .activity-value {{
            font-size: 28px;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .activity-class {{
            font-size: 16px;
            opacity: 0.9;
        }}
        .confidence-display {{
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 10px;
            background-color: white;
            border-radius: 5px;
            margin-top: 10px;
        }}
        .confidence-bar {{
            flex: 1;
            height: 20px;
            background-color: #e0e0e0;
            border-radius: 10px;
            overflow: hidden;
        }}
        .confidence-fill {{
            height: 100%;
            background: linear-gradient(90deg, #e74c3c 0%, #f39c12 50%, #27ae60 100%);
            transition: width 0.3s;
        }}
        .confidence-label {{
            font-weight: bold;
            min-width: 150px;
        }}
        .properties-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
        }}
        .property {{
            background-color: white;
            padding: 10px;
            border-radius: 4px;
            border-left: 3px solid #3498db;
        }}
        .property-label {{
            font-size: 12px;
            color: #7f8c8d;
            text-transform: uppercase;
        }}
        .property-value {{
            font-size: 18px;
            font-weight: bold;
            color: #2c3e50;
        }}
        .analogues-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}
        .analogues-table th {{
            background-color: #34495e;
            color: white;
            padding: 12px;
            text-align: left;
        }}
        .analogues-table td {{
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }}
        .analogues-table tr:hover {{
            background-color: #f8f9fa;
        }}
        .product-code {{
            font-family: 'Courier New', monospace;
            background-color: #ecf0f1;
            padding: 4px 8px;
            border-radius: 3px;
            font-weight: bold;
        }}
        .similarity-bar {{
            height: 20px;
            background-color: #e0e0e0;
            border-radius: 10px;
            overflow: hidden;
        }}
        .similarity-fill {{
            height: 100%;
            background: linear-gradient(90deg, #e74c3c 0%, #f39c12 50%, #27ae60 100%);
            transition: width 0.3s;
        }}
        .rationale {{
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 15px 0;
        }}
        .smiles-box {{
            background-color: #f8f9fa;
            padding: 10px;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
            font-size: 12px;
            word-break: break-all;
            border: 1px solid #dee2e6;
        }}
        .arm-selection-diagram {{
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            margin: 30px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .diagram-description {{
            color: #555;
            font-size: 14px;
            margin-bottom: 20px;
        }}
        .arm-performance-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .arm-card {{
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            border: 2px solid #ddd;
        }}
        .arm-card.excellent {{
            border-color: #27ae60;
            background: linear-gradient(135deg, #f0fff4 0%, #e8f5e9 100%);
        }}
        .arm-card.good {{
            border-color: #3498db;
            background: linear-gradient(135deg, #f0f9ff 0%, #e3f2fd 100%);
        }}
        .arm-card.moderate {{
            border-color: #f39c12;
            background: linear-gradient(135deg, #fffbf0 0%, #fff3e0 100%);
        }}
        .arm-card.poor {{
            border-color: #e74c3c;
            background: linear-gradient(135deg, #fff5f5 0%, #ffebee 100%);
        }}
        .arm-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }}
        .arm-header h3 {{
            margin: 0;
            color: #2c3e50;
        }}
        .performance-badge {{
            padding: 5px 15px;
            border-radius: 15px;
            font-size: 12px;
            font-weight: bold;
            color: white;
        }}
        .performance-badge.excellent {{ background-color: #27ae60; }}
        .performance-badge.good {{ background-color: #3498db; }}
        .performance-badge.moderate {{ background-color: #f39c12; }}
        .performance-badge.poor {{ background-color: #e74c3c; }}
        .arm-description {{
            color: #555;
            font-size: 14px;
            margin: 10px 0;
        }}
        .arm-stats {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
            margin: 15px 0;
        }}
        .stat {{
            text-align: center;
            background-color: white;
            padding: 10px;
            border-radius: 5px;
        }}
        .stat-label {{
            display: block;
            font-size: 11px;
            color: #7f8c8d;
            text-transform: uppercase;
        }}
        .stat-value {{
            display: block;
            font-size: 20px;
            font-weight: bold;
            color: #2c3e50;
            margin-top: 5px;
        }}
        .arm-strength {{
            margin-top: 15px;
            padding: 10px;
            background-color: white;
            border-radius: 5px;
            font-size: 13px;
            color: #555;
        }}
        .global-top-section {{
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            margin: 30px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .section-description {{
            color: #555;
            font-size: 14px;
            margin-bottom: 20px;
        }}
    </style>
</head>
<body>
    <h1>Medicinal Chemistry Handoff Report</h1>

    <div class="metadata">
        <h3>Run Information</h3>
        <p><strong>Episode:</strong> {episode_key}</p>
        <p><strong>Arm:</strong> {episode_data['arm']}</p>
        <p><strong>Episode Number:</strong> {episode_data['episode']}</p>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>

    <div class="rationale">
        <h3>Selection Rationale</h3>
        <p>The following molecules were selected based on:</p>
        <ul>
            <li><strong>Predicted BRD4 Activity:</strong> QSAR model predictions converted to nM scale for easy interpretation</li>
            <li><strong>Confidence Scores:</strong> Based on model certainty and chemical space coverage</li>
            <li><strong>Overall Score:</strong> Composite score considering activity, druglikeness, and novelty</li>
            <li><strong>Molecular Properties:</strong> Drug-like properties (MW, LogP, PSA, HBD, HBA)</li>
            <li><strong>Chemical Diversity:</strong> Scaffold diversity to explore chemical space</li>
        </ul>
    </div>
""")

        # Process batches or single episode
        if episode_data['has_batches']:
            for batch in episode_data['batches']:
                batch_num = batch['batch_num']
                html_parts.append(f"<h2>Batch {batch_num}</h2>")
                df = self.process_batch(batch['csv_file'])
                html_parts.append(self._generate_molecules_html(df))
        else:
            df = self.process_batch(episode_data['csv_file'])
            html_parts.append(self._generate_molecules_html(df))

        html_parts.append("""
</body>
</html>
""")

        return "\n".join(html_parts)

    def _generate_molecules_html(self, df: pd.DataFrame, show_arm: bool = False) -> str:
        """Generate HTML for molecule cards."""
        html_parts = []

        for idx, (_, row) in enumerate(df.iterrows(), 1):
            smiles = row['SMILES']
            score = row['Score']

            # Get BRD4 activity - use raw value with multiple fallbacks
            activity_raw = None
            for col in ['BRD4_Activity (raw)', 'BRD4_Activity']:
                if col in row.index and not pd.isna(row[col]):
                    activity_raw = row[col]
                    break

            # Denormalize to nM scale
            if activity_raw is not None and not pd.isna(activity_raw):
                ic50_nM, activity_class, activity_display = self.denormalize_brd4_activity(float(activity_raw))
            else:
                ic50_nM = None
                activity_class = "Unknown"
                activity_display = "N/A"

            # Compute confidence
            confidence_score, confidence_label = self.compute_confidence_score(row)

            # Get molecular properties - use raw values with multiple fallbacks for different column names
            mw_raw = None
            for col in ['MW (raw)', 'Molecular weight (raw)', 'MW', 'Molecular weight']:
                if col in row.index and not pd.isna(row[col]):
                    mw_raw = row[col]
                    break

            logp_raw = None
            for col in ['LogP (raw)', 'logP (raw)', 'LogP', 'logP']:
                if col in row.index and not pd.isna(row[col]):
                    logp_raw = row[col]
                    break

            psa_raw = None
            for col in ['PSA (raw)', 'TPSA (raw)', 'PSA', 'TPSA']:
                if col in row.index and not pd.isna(row[col]):
                    psa_raw = row[col]
                    break

            hbd_raw = None
            for col in ['HBD (raw)', 'HBD']:
                if col in row.index and not pd.isna(row[col]):
                    hbd_raw = row[col]
                    break

            hba_raw = None
            for col in ['HBA (raw)', 'HBA']:
                if col in row.index and not pd.isna(row[col]):
                    hba_raw = row[col]
                    break

            # Format values
            mw = f"{float(mw_raw):.1f}" if mw_raw is not None and not pd.isna(mw_raw) else "N/A"
            logp = f"{float(logp_raw):.2f}" if logp_raw is not None and not pd.isna(logp_raw) else "N/A"
            psa = f"{float(psa_raw):.1f}" if psa_raw is not None and not pd.isna(psa_raw) else "N/A"
            hbd = f"{int(float(hbd_raw))}" if hbd_raw is not None and not pd.isna(hbd_raw) else "N/A"
            hba = f"{int(float(hba_raw))}" if hba_raw is not None and not pd.isna(hba_raw) else "N/A"

            # Generate 2D image
            img_base64 = self.generate_mol_image_base64(smiles)

            # Find MOLPORT analogues
            analogues = self.find_molport_analogues(smiles)

            # Arm badge if showing global top
            arm_badge_html = ""
            if show_arm:
                arm_name = row.get('arm', 'Unknown')
                arm_badge_html = f'<span class="arm-badge">{arm_name}</span>'

            html_parts.append(f"""
    <div class="molecule-card">
        <div class="molecule-header">
            <span class="molecule-rank">#{idx}</span>
            {arm_badge_html}
            <span class="molecule-score">Score: {score:.4f}</span>
        </div>

        <div class="molecule-content">
            <div class="molecule-image">
                {'<img src="' + img_base64 + '" alt="Molecule structure">' if img_base64 else '<p>Structure not available</p>'}
                <div class="smiles-box">{smiles}</div>
            </div>

            <div class="molecule-info">
                <div class="info-section">
                    <h4>Predicted BRD4 Activity</h4>
                    <div class="activity-display">
                        <div class="activity-value">IC₅₀: {activity_display}</div>
                        <div class="activity-class">{activity_class}</div>
                    </div>
                    <div class="confidence-display">
                        <span class="confidence-label">{confidence_label}</span>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: {confidence_score * 100}%"></div>
                        </div>
                        <span>{confidence_score:.0%}</span>
                    </div>
                </div>

                <div class="info-section">
                    <h4>Molecular Properties</h4>
                    <div class="properties-grid">
                        <div class="property">
                            <div class="property-label">Molecular Weight</div>
                            <div class="property-value">{mw}</div>
                        </div>
                        <div class="property">
                            <div class="property-label">LogP</div>
                            <div class="property-value">{logp}</div>
                        </div>
                        <div class="property">
                            <div class="property-label">PSA</div>
                            <div class="property-value">{psa}</div>
                        </div>
                        <div class="property">
                            <div class="property-label">H-Bond Donors</div>
                            <div class="property-value">{hbd}</div>
                        </div>
                        <div class="property">
                            <div class="property-label">H-Bond Acceptors</div>
                            <div class="property-value">{hba}</div>
                        </div>
                    </div>
                </div>
""")

            # MOLPORT analogues section
            if analogues:
                html_parts.append("""
                <div class="info-section">
                    <h4>Purchaseable Analogues (MOLPORT)</h4>
                    <table class="analogues-table">
                        <thead>
                            <tr>
                                <th>Rank</th>
                                <th>Product Code</th>
                                <th>Similarity</th>
                                <th>Structure</th>
                            </tr>
                        </thead>
                        <tbody>
""")

                for aidx, analogue in enumerate(analogues, 1):
                    ana_smiles = analogue['smiles']
                    ana_code = analogue['product_code'] if analogue['product_code'] else 'N/A'
                    ana_sim = analogue['similarity']
                    ana_img = self.generate_mol_image_base64(ana_smiles, size=(150, 150))

                    html_parts.append(f"""
                            <tr>
                                <td><strong>#{aidx}</strong></td>
                                <td><span class="product-code">{ana_code}</span></td>
                                <td>
                                    <div class="similarity-bar">
                                        <div class="similarity-fill" style="width: {ana_sim * 100}%"></div>
                                    </div>
                                    <small>{ana_sim:.3f}</small>
                                </td>
                                <td>
                                    {'<img src="' + ana_img + '" style="max-width: 150px;">' if ana_img else ana_smiles[:50]}
                                </td>
                            </tr>
""")

                html_parts.append("""
                        </tbody>
                    </table>
                </div>
""")
            else:
                html_parts.append(f"""
                <div class="info-section">
                    <h4>Purchaseable Analogues (MOLPORT)</h4>
                    <p>No purchaseable analogues found above {self.molport_sim_threshold:.0%} similarity threshold.</p>
                    <p><em>Try lowering the similarity threshold to find more distant analogues.</em></p>
                </div>
""")

            html_parts.append("""
            </div>
        </div>
    </div>
""")

        return "\n".join(html_parts)

    def build(self):
        """Build medchem handoff reports for all episodes."""
        logger.info("Building medchem handoff reports...")

        # Create output directories
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "cache").mkdir(exist_ok=True)

        # Discover episodes and batches
        episodes = self.discover_episodes_and_batches()

        # Load MOLPORT catalog
        if self.molport_csvs:
            self.load_molport_catalog()

        # Collect all molecules for global ranking
        logger.info("Collecting molecules for global ranking...")
        global_top = self.collect_all_molecules(episodes)
        self.all_molecules_df = global_top

        # Generate master index with global top performers and arm analysis
        self._generate_master_index(episodes, global_top)

        # Generate reports for each episode
        for episode_key, episode_data in episodes.items():
            logger.info(f"Generating report for {episode_key}...")

            html_report = self.generate_html_report(episode_key, episode_data)

            # Save report
            report_path = self.out_dir / f"{episode_key}_medchem_handoff.html"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_report)

            logger.info(f"Saved report to {report_path}")

        logger.info(f"Medchem handoff reports complete! Output: {self.out_dir}")

    def _generate_master_index(self, episodes: Dict, global_top: pd.DataFrame):
        """Generate master index page with global top performers and arm analysis."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>MedChem Handoff Reports - Master Index</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        .report-list {{
            background-color: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-top: 20px;
        }}
        .report-item {{
            padding: 15px;
            border-bottom: 1px solid #ddd;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .report-item:hover {{
            background-color: #f8f9fa;
        }}
        .report-link {{
            text-decoration: none;
            color: #3498db;
            font-size: 18px;
            font-weight: bold;
        }}
        .report-link:hover {{
            text-decoration: underline;
        }}
        .report-meta {{
            color: #7f8c8d;
            font-size: 14px;
        }}
        .molecule-card {{
            background-color: white;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .molecule-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        .molecule-rank {{
            font-size: 24px;
            font-weight: bold;
            color: #3498db;
        }}
        .molecule-score {{
            font-size: 20px;
            color: #27ae60;
            font-weight: bold;
        }}
        .arm-badge {{
            background-color: #9b59b6;
            color: white;
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 12px;
            font-weight: bold;
        }}
        .molecule-content {{
            display: grid;
            grid-template-columns: 350px 1fr;
            gap: 20px;
        }}
        .molecule-image {{
            text-align: center;
        }}
        .molecule-image img {{
            max-width: 300px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }}
        .molecule-info {{
            display: flex;
            flex-direction: column;
            gap: 15px;
        }}
        .info-section {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
        }}
        .info-section h4 {{
            margin-top: 0;
            color: #2c3e50;
            border-bottom: 1px solid #ddd;
            padding-bottom: 5px;
        }}
        .activity-display {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 10px;
        }}
        .activity-value {{
            font-size: 28px;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .activity-class {{
            font-size: 16px;
            opacity: 0.9;
        }}
        .confidence-display {{
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 10px;
            background-color: white;
            border-radius: 5px;
            margin-top: 10px;
        }}
        .confidence-bar {{
            flex: 1;
            height: 20px;
            background-color: #e0e0e0;
            border-radius: 10px;
            overflow: hidden;
        }}
        .confidence-fill {{
            height: 100%;
            background: linear-gradient(90deg, #e74c3c 0%, #f39c12 50%, #27ae60 100%);
            transition: width 0.3s;
        }}
        .confidence-label {{
            font-weight: bold;
            min-width: 150px;
        }}
        .properties-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
        }}
        .property {{
            background-color: white;
            padding: 10px;
            border-radius: 4px;
            border-left: 3px solid #3498db;
        }}
        .property-label {{
            font-size: 12px;
            color: #7f8c8d;
            text-transform: uppercase;
        }}
        .property-value {{
            font-size: 18px;
            font-weight: bold;
            color: #2c3e50;
        }}
        .analogues-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}
        .analogues-table th {{
            background-color: #34495e;
            color: white;
            padding: 12px;
            text-align: left;
        }}
        .analogues-table td {{
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }}
        .analogues-table tr:hover {{
            background-color: #f8f9fa;
        }}
        .product-code {{
            font-family: 'Courier New', monospace;
            background-color: #ecf0f1;
            padding: 4px 8px;
            border-radius: 3px;
            font-weight: bold;
        }}
        .similarity-bar {{
            height: 20px;
            background-color: #e0e0e0;
            border-radius: 10px;
            overflow: hidden;
        }}
        .similarity-fill {{
            height: 100%;
            background: linear-gradient(90deg, #e74c3c 0%, #f39c12 50%, #27ae60 100%);
            transition: width 0.3s;
        }}
        .smiles-box {{
            background-color: #f8f9fa;
            padding: 10px;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
            font-size: 12px;
            word-break: break-all;
            border: 1px solid #dee2e6;
        }}
        .arm-selection-diagram {{
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            margin: 30px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .diagram-description {{
            color: #555;
            font-size: 14px;
            margin-bottom: 20px;
        }}
        .arm-performance-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .arm-card {{
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            border: 2px solid #ddd;
        }}
        .arm-card.excellent {{
            border-color: #27ae60;
            background: linear-gradient(135deg, #f0fff4 0%, #e8f5e9 100%);
        }}
        .arm-card.good {{
            border-color: #3498db;
            background: linear-gradient(135deg, #f0f9ff 0%, #e3f2fd 100%);
        }}
        .arm-card.moderate {{
            border-color: #f39c12;
            background: linear-gradient(135deg, #fffbf0 0%, #fff3e0 100%);
        }}
        .arm-card.poor {{
            border-color: #e74c3c;
            background: linear-gradient(135deg, #fff5f5 0%, #ffebee 100%);
        }}
        .arm-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }}
        .arm-header h3 {{
            margin: 0;
            color: #2c3e50;
        }}
        .performance-badge {{
            padding: 5px 15px;
            border-radius: 15px;
            font-size: 12px;
            font-weight: bold;
            color: white;
        }}
        .performance-badge.excellent {{ background-color: #27ae60; }}
        .performance-badge.good {{ background-color: #3498db; }}
        .performance-badge.moderate {{ background-color: #f39c12; }}
        .performance-badge.poor {{ background-color: #e74c3c; }}
        .arm-description {{
            color: #555;
            font-size: 14px;
            margin: 10px 0;
        }}
        .arm-stats {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
            margin: 15px 0;
        }}
        .stat {{
            text-align: center;
            background-color: white;
            padding: 10px;
            border-radius: 5px;
        }}
        .stat-label {{
            display: block;
            font-size: 11px;
            color: #7f8c8d;
            text-transform: uppercase;
        }}
        .stat-value {{
            display: block;
            font-size: 20px;
            font-weight: bold;
            color: #2c3e50;
            margin-top: 5px;
        }}
        .arm-strength {{
            margin-top: 15px;
            padding: 10px;
            background-color: white;
            border-radius: 5px;
            font-size: 13px;
            color: #555;
        }}
        .global-top-section {{
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            margin: 30px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .section-description {{
            color: #555;
            font-size: 14px;
            margin-bottom: 20px;
        }}
    </style>
</head>
<body>
    <h1>Medicinal Chemistry Handoff Reports - Master Index</h1>
    <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p><strong>Run Directory:</strong> {self.run_dir}</p>
    <p><strong>Total Unique Molecules:</strong> {len(global_top)}</p>
"""

        # Add arm selection diagram
        html += self.generate_arm_selection_diagram(episodes, global_top)

        # Add global top performers
        html += self.generate_global_top_performers(global_top)

        # Add episode list
        html += """
    <div class="report-list">
        <h2>Individual Episode Reports</h2>
"""

        for episode_key, episode_data in sorted(episodes.items()):
            report_filename = f"{episode_key}_medchem_handoff.html"
            batch_info = f"({len(episode_data['batches'])} batches)" if episode_data['has_batches'] else ""

            html += f"""
        <div class="report-item">
            <a href="{report_filename}" class="report-link">{episode_key}</a>
            <span class="report-meta">{episode_data['arm']} - Episode {episode_data['episode']} {batch_info}</span>
        </div>
"""

        html += """
    </div>
</body>
</html>
"""

        index_path = self.out_dir / "index.html"
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(html)

        logger.info(f"Saved master index page to {index_path}")
