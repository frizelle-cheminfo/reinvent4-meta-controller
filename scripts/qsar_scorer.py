#!/usr/bin/env python3
"""
QSAR Scoring Wrapper for REINVENT4 ExternalProcess Component

This script loads a trained QSAR model and scores molecules via SMILES input.
Designed to be called by REINVENT4's ExternalProcess scoring component.

Usage:
    python qsar_scorer.py <model_path> <transformation>

Input: SMILES strings (one per line) via stdin
Output: JSON object with predictions via stdout in format:
    {"version": 1, "payload": {"predictions": [score1, score2, ...]}}

Author: Generated for BRD4 QSAR-guided discovery campaign
"""

import sys
import json
import pickle
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors


def calculate_features(mol):
    """
    Calculate molecular features (ECFP4 + descriptors)

    Args:
        mol: RDKit molecule object

    Returns:
        Feature vector as numpy array
    """
    # ECFP4 fingerprint (radius 2, 2048 bits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    fp_array = np.array(fp)

    # Molecular descriptors (must match training script exactly)
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    rotatable = Descriptors.NumRotatableBonds(mol)
    aromatic_rings = Descriptors.NumAromaticRings(mol)
    fraction_csp3 = Descriptors.FractionCSP3(mol)
    num_rings = Descriptors.RingCount(mol)
    mol_mr = Descriptors.MolMR(mol)

    # Combine fingerprint and descriptors
    descriptors = np.array([mw, logp, tpsa, hbd, hba, rotatable, aromatic_rings,
                           fraction_csp3, num_rings, mol_mr])
    features = np.concatenate([fp_array, descriptors])

    return features


def transform_score(raw_score, transformation="sigmoid"):
    """
    Transform raw QSAR prediction to 0-1 score

    Args:
        raw_score: Raw pActivity prediction
        transformation: Transformation type (sigmoid, linear, or none)

    Returns:
        Transformed score in [0, 1]
    """
    if transformation == "sigmoid":
        # Sigmoid transformation centered at pActivity = 6.5
        center = 6.5
        steepness = 0.5
        return 1.0 / (1.0 + np.exp(-steepness * (raw_score - center)))

    elif transformation == "linear":
        # Linear scaling: pActivity 4.0 → 0.0, 9.0 → 1.0
        min_val = 4.0
        max_val = 9.0
        return np.clip((raw_score - min_val) / (max_val - min_val), 0.0, 1.0)

    else:  # none
        return raw_score


def main():
    """Main scoring function"""
    if len(sys.argv) < 3:
        print("ERROR: Missing arguments", file=sys.stderr)
        print(f"Usage: {sys.argv[0]} <model_path> <transformation>", file=sys.stderr)
        sys.exit(1)

    model_path = Path(sys.argv[1])
    transformation = sys.argv[2]

    # Load model
    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            # Handle dict-wrapped models
            if isinstance(data, dict) and 'model' in data:
                model = data['model']
            else:
                model = data
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}", file=sys.stderr)
        sys.exit(1)

    # Collect all scores for JSON output
    scores = []

    # Read SMILES from stdin
    for line in sys.stdin:
        smiles = line.strip()

        if not smiles:
            continue

        try:
            # Parse SMILES
            mol = Chem.MolFromSmiles(smiles)

            if mol is None:
                # Invalid SMILES → score 0
                scores.append(0.0)
                continue

            # Calculate features
            features = calculate_features(mol)

            # Predict pActivity
            features_2d = features.reshape(1, -1)
            raw_prediction = model.predict(features_2d)[0]

            # Transform to [0, 1] score
            score = transform_score(raw_prediction, transformation)
            scores.append(float(score))

        except Exception as e:
            # Any error → score 0
            scores.append(0.0)
            print(f"WARNING: Error scoring {smiles}: {e}", file=sys.stderr)

    # Output JSON in REINVENT4 ExternalProcess format
    output = {
        "version": 1,
        "payload": {
            "predictions": scores
        }
    }
    print(json.dumps(output))


if __name__ == "__main__":
    main()
