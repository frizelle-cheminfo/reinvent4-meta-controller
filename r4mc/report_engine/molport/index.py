"""MolPort fingerprint index for fast similarity search."""
import numpy as np
import pandas as pd
import pickle
import logging
from pathlib import Path
from typing import Optional, List
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

logger = logging.getLogger(__name__)


class MolPortIndex:
    """Fingerprint index for MolPort molecules."""

    def __init__(self, fp_radius: int = 2, fp_nbits: int = 2048):
        """
        Initialize MolPort index.

        Args:
            fp_radius: Morgan fingerprint radius
            fp_nbits: Number of bits in fingerprint
        """
        self.fp_radius = fp_radius
        self.fp_nbits = fp_nbits
        self.fingerprints = []
        self.smiles = []
        self.ids = []

    def add_molecules(self, df: pd.DataFrame):
        """
        Add molecules from DataFrame to index.

        Args:
            df: DataFrame with 'canonical_smiles' and 'id' columns
        """
        for _, row in df.iterrows():
            smiles = row['canonical_smiles']
            mol_id = row['id']

            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue

                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol, self.fp_radius, nBits=self.fp_nbits
                )

                self.fingerprints.append(fp)
                self.smiles.append(smiles)
                self.ids.append(mol_id)

            except Exception as e:
                logger.debug(f"Failed to add molecule {mol_id}: {e}")
                continue

    def save(self, path: str):
        """
        Save index to disk.

        Args:
            path: Path to save index
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'fp_radius': self.fp_radius,
            'fp_nbits': self.fp_nbits,
            'fingerprints': self.fingerprints,
            'smiles': self.smiles,
            'ids': self.ids
        }

        with open(path, 'wb') as f:
            pickle.dump(data, f)

        logger.info(f"Saved MolPort index to {path} ({len(self.fingerprints)} molecules)")

    @classmethod
    def load(cls, path: str) -> 'MolPortIndex':
        """
        Load index from disk.

        Args:
            path: Path to load index from

        Returns:
            Loaded MolPortIndex
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Index not found: {path}")

        with open(path, 'rb') as f:
            data = pickle.load(f)

        index = cls(fp_radius=data['fp_radius'], fp_nbits=data['fp_nbits'])
        index.fingerprints = data['fingerprints']
        index.smiles = data['smiles']
        index.ids = data['ids']

        logger.info(f"Loaded MolPort index from {path} ({len(index.fingerprints)} molecules)")

        return index

    def __len__(self):
        return len(self.fingerprints)


def build_molport_index(
    molport_df: pd.DataFrame,
    cache_path: Optional[str] = None,
    fp_radius: int = 2,
    fp_nbits: int = 2048
) -> MolPortIndex:
    """
    Build MolPort fingerprint index.

    Args:
        molport_df: DataFrame with MolPort molecules
        cache_path: Optional path to cache index
        fp_radius: Morgan fingerprint radius
        fp_nbits: Number of bits in fingerprint

    Returns:
        MolPortIndex
    """
    # Check if cached index exists
    if cache_path and Path(cache_path).exists():
        logger.info(f"Loading cached MolPort index from {cache_path}")
        return MolPortIndex.load(cache_path)

    logger.info(f"Building MolPort index for {len(molport_df)} molecules...")

    index = MolPortIndex(fp_radius=fp_radius, fp_nbits=fp_nbits)
    index.add_molecules(molport_df)

    # Save to cache if requested
    if cache_path:
        index.save(cache_path)

    logger.info(f"Built MolPort index with {len(index)} molecules")

    return index
