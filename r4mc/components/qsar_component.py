"""
Custom QSAR Scoring Component for REINVENT4
Loads a trained ML model and predicts activity from SMILES
"""

import pickle
import numpy as np
from pathlib import Path
from typing import List
from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

__all__ = ["QSARComponent"]


@dataclass
class Parameters:
    """Parameters for QSAR scoring component"""
    model_path: str
    transformation: str = "sigmoid"  # sigmoid, linear, or raw
    min_score: float = 0.0
    max_score: float = 1.0


class QSARComponent:
    """
    QSAR-based scoring component for REINVENT4

    Loads a pre-trained model and scores molecules based on predicted activity.
    Compatible with REINVENT4 scoring function interface.
    """

    def __init__(self, parameters: Parameters):
        """
        Initialize QSAR component

        Args:
            parameters: Configuration with model_path and transformation settings
        """
        self.parameters = parameters

        # Load model
        model_path = Path(parameters.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.model_name = model_data['model_name']
        self.feature_names = model_data['feature_names']
        self.metrics = model_data.get('metrics', {})

        print(f"Loaded {self.model_name} model from {model_path}")
        print(f"Model metrics: Test R² = {self.metrics.get('test_r2', 'N/A')}")

    def __call__(self, smilies: List[str]) -> np.ndarray:
        """
        Score molecules using QSAR model

        Args:
            smilies: List of SMILES strings

        Returns:
            Array of scores (0-1 range)
        """
        scores = []

        for smiles in smilies:
            try:
                # Calculate features
                features = self._calculate_features(smiles)
                if features is None:
                    scores.append(0.0)
                    continue

                # Predict activity
                prediction = self.model.predict(features.reshape(1, -1))[0]

                # Transform to 0-1 score
                score = self._transform_prediction(prediction)
                scores.append(score)

            except Exception as e:
                print(f"Error scoring {smiles}: {e}")
                scores.append(0.0)

        return np.array(scores, dtype=np.float32)

    def _calculate_features(self, smiles: str) -> np.ndarray:
        """Calculate ECFP4 + descriptor features for a SMILES"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # ECFP4 fingerprint (radius=2, 2048 bits)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        fp_array = np.array(fp)

        # Physicochemical descriptors (same as training)
        try:
            descriptors = np.array([
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.TPSA(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.NumRotatableBonds(mol),
                Descriptors.NumAromaticRings(mol),
                Descriptors.FractionCSP3(mol),
                Descriptors.RingCount(mol),
                Descriptors.MolMR(mol),
            ])
        except Exception as e:
            print(f"Error calculating descriptors for {smiles}: {e}")
            return None

        # Combine features
        features = np.concatenate([fp_array, descriptors])
        return features

    def _transform_prediction(self, prediction: float) -> float:
        """
        Transform raw prediction to 0-1 score

        Args:
            prediction: Raw model prediction (pActivity)

        Returns:
            Score in 0-1 range
        """
        if self.parameters.transformation == "raw":
            # Clip to min/max range
            return np.clip(prediction, self.parameters.min_score, self.parameters.max_score)

        elif self.parameters.transformation == "linear":
            # Linear scaling from typical pActivity range (2-11) to 0-1
            pActivity_min = 2.0
            pActivity_max = 11.0
            score = (prediction - pActivity_min) / (pActivity_max - pActivity_min)
            return np.clip(score, 0.0, 1.0)

        elif self.parameters.transformation == "sigmoid":
            # Sigmoid transformation centered at pActivity 6.5 (1 µM)
            # This gives smooth 0-1 scores with target around mid-range
            center = 6.5
            steepness = 1.0
            score = 1.0 / (1.0 + np.exp(-steepness * (prediction - center)))
            return score

        else:
            raise ValueError(f"Unknown transformation: {self.parameters.transformation}")


# REINVENT4 component registration
def component_wrapper(smilies: List[str], parameters: Parameters) -> np.ndarray:
    """
    Wrapper function for REINVENT4 component interface

    This is the function that REINVENT4 will call when using this component
    in a scoring configuration.
    """
    component = QSARComponent(parameters)
    return component(smilies)
