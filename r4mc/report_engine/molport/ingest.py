"""MolPort CSV ingestion with chunked processing."""
import pandas as pd
import logging
from pathlib import Path
from typing import Optional, Iterator
from ..utils.smiles import canonicalize_smiles

logger = logging.getLogger(__name__)


def ingest_molport_csv(
    csv_path: str,
    smiles_col: str = 'SMILES',
    id_col: str = 'ID',
    chunk_size: int = 10000
) -> Iterator[pd.DataFrame]:
    """
    Ingest MolPort CSV in chunks for memory safety.

    Args:
        csv_path: Path to MolPort CSV file
        smiles_col: Name of SMILES column
        id_col: Name of ID column
        chunk_size: Number of rows to read at a time

    Yields:
        DataFrames with SMILES and IDs
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"MolPort CSV not found: {csv_path}")

    logger.info(f"Ingesting MolPort CSV from {csv_path} (chunk size: {chunk_size})")

    try:
        # Read in chunks
        for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunk_size)):
            # Verify columns exist
            if smiles_col not in chunk.columns:
                # Try to find SMILES column
                smiles_candidates = ['SMILES', 'smiles', 'Smiles', 'SMILE', 'smile']
                smiles_col = next((c for c in smiles_candidates if c in chunk.columns), None)

                if smiles_col is None:
                    raise ValueError(f"SMILES column not found in CSV. Available columns: {chunk.columns.tolist()}")

            if id_col not in chunk.columns:
                # Try to find ID column
                id_candidates = ['ID', 'id', 'Id', 'compound_id', 'CompoundID']
                id_col = next((c for c in id_candidates if c in chunk.columns), None)

                if id_col is None:
                    # Use row index as ID
                    chunk['ID'] = chunk.index
                    id_col = 'ID'

            # Extract relevant columns
            df = chunk[[smiles_col, id_col]].copy()
            df.columns = ['smiles', 'id']

            # Drop NA
            df = df.dropna(subset=['smiles'])

            # Canonicalize SMILES
            df['canonical_smiles'] = df['smiles'].apply(canonicalize_smiles)
            df = df.dropna(subset=['canonical_smiles'])

            logger.debug(f"Processed chunk {i + 1}: {len(df)} valid molecules")

            yield df

    except Exception as e:
        logger.error(f"Error ingesting MolPort CSV: {e}")
        raise


def load_molport_to_dataframe(
    csv_path: str,
    max_rows: Optional[int] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Load MolPort CSV to a single DataFrame.

    Args:
        csv_path: Path to MolPort CSV file
        max_rows: Maximum number of rows to load
        **kwargs: Additional arguments for ingest_molport_csv

    Returns:
        DataFrame with all MolPort molecules
    """
    chunks = []

    for i, chunk in enumerate(ingest_molport_csv(csv_path, **kwargs)):
        chunks.append(chunk)

        if max_rows is not None:
            total_rows = sum(len(c) for c in chunks)
            if total_rows >= max_rows:
                break

    if not chunks:
        return pd.DataFrame(columns=['smiles', 'id', 'canonical_smiles'])

    df = pd.concat(chunks, ignore_index=True)

    if max_rows is not None:
        df = df.head(max_rows)

    logger.info(f"Loaded {len(df)} molecules from MolPort CSV")

    return df
