# QSAR Model Training Guide

Complete guide to training your own QSAR models for use with the meta-controller.

## Overview

The meta-controller uses QSAR models to score molecules during generation. You can train models on your own data and plug them directly into the workflow.

## Quick Start

```python
# Train a QSAR model in 5 steps
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from scripts.train_qsar import train_qsar_model  # Helper script

# 1. Load your data (SMILES + activity)
data = pd.read_csv('my_training_data.csv')
# Columns: smiles, pActivity (or IC50, Kd, etc.)

# 2. Train model
model, metrics = train_qsar_model(
    data=data,
    smiles_col='smiles',
    target_col='pActivity',
    output_path='models/my_target/model.pkl'
)

# 3. Check performance
print(f"Test R²: {metrics['test_r2']:.3f}")
print(f"Test RMSE: {metrics['test_rmse']:.3f}")

# 4. Use in meta-controller
# See configuration examples below
```

## Data Requirements

### Minimum Dataset Size
- **Hit discovery**: 300+ compounds (sparse SAR acceptable)
- **Lead optimization**: 500+ compounds (dense SAR recommended)
- **Publication-quality**: 1000+ compounds

### Data Format

CSV file with at minimum:
```csv
smiles,pActivity
CCO,4.5
CC(C)O,5.2
CCCC,6.8
```

**Required columns:**
- `smiles`: Valid SMILES strings
- Activity column: pIC50, pKi, pKd, %inhibition, etc.

**Optional columns:**
- `compound_id`: Identifiers
- `set`: 'train' or 'test' for predefined splits
- `assay_type`: For multi-assay data

### Activity Units

Convert to pActivity (higher = better):
```python
# IC50 (nM) to pIC50
pIC50 = -np.log10(IC50_nM * 1e-9)

# Ki (μM) to pKi
pKi = -np.log10(Ki_uM * 1e-6)

# % inhibition (keep as-is, 0-100 scale)
activity = percent_inhibition
```

## Feature Calculation

The QSAR scorer uses:
- **ECFP4 fingerprints** (radius 2, 2048 bits)
- **10 RDKit descriptors**:
  1. Molecular weight
  2. LogP
  3. TPSA
  4. H-bond donors
  5. H-bond acceptors
  6. Rotatable bonds
  7. Aromatic rings
  8. Fraction Csp3
  9. Ring count
  10. Molar refractivity

**Total: 2058 features**

### Feature Calculation Code

```python
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

def calculate_features(smiles):
    """Calculate ECFP4 + descriptors for a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # ECFP4 (2048 bits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    fp_array = np.array(fp)

    # Descriptors (10 values)
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
        Descriptors.MolMR(mol)
    ])

    return np.concatenate([fp_array, descriptors])
```

## Training Script

Complete training script with cross-validation:

```python
#!/usr/bin/env python3
"""Train QSAR model with proper validation."""

import pickle
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from pathlib import Path

def calculate_features(smiles):
    """Calculate molecular features."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    fp_array = np.array(fp)

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
        Descriptors.MolMR(mol)
    ])

    return np.concatenate([fp_array, descriptors])

def train_qsar_model(
    data_path,
    smiles_col='smiles',
    target_col='pActivity',
    output_dir='models/my_target',
    test_size=0.2,
    random_state=42
):
    """
    Train a QSAR model with cross-validation.

    Parameters:
        data_path: Path to CSV file with SMILES and activity
        smiles_col: Name of SMILES column
        target_col: Name of activity column
        output_dir: Where to save the model
        test_size: Fraction for test set
        random_state: Random seed
    """

    # Load data
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} compounds")

    # Calculate features
    print("Calculating features...")
    features = []
    valid_indices = []

    for idx, smiles in enumerate(df[smiles_col]):
        feat = calculate_features(smiles)
        if feat is not None:
            features.append(feat)
            valid_indices.append(idx)

    X = np.array(features)
    y = df.loc[valid_indices, target_col].values

    print(f"Valid compounds: {len(X)}")
    print(f"Feature shape: {X.shape}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Train model
    print("Training Random Forest...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        random_state=random_state,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Cross-validation
    cv_scores = cross_val_score(
        model, X_train, y_train, cv=5, scoring='r2', n_jobs=-1
    )

    # Test set predictions
    y_pred = model.predict(X_test)

    # Metrics
    metrics = {
        'train_r2': model.score(X_train, y_train),
        'test_r2': r2_score(y_test, y_pred),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'test_mae': mean_absolute_error(y_test, y_pred),
        'cv_r2_mean': cv_scores.mean(),
        'cv_r2_std': cv_scores.std(),
        'n_train': len(X_train),
        'n_test': len(X_test)
    }

    # Print results
    print("\n=== Training Results ===")
    print(f"Train R²: {metrics['train_r2']:.3f}")
    print(f"Test R²:  {metrics['test_r2']:.3f}")
    print(f"Test RMSE: {metrics['test_rmse']:.3f}")
    print(f"Test MAE:  {metrics['test_mae']:.3f}")
    print(f"CV R² (5-fold): {metrics['cv_r2_mean']:.3f} ± {metrics['cv_r2_std']:.3f}")

    # Save model
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_file = output_path / 'model.pkl'
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nSaved model to: {model_file}")

    # Save metadata
    import json
    metadata = {
        'metrics': metrics,
        'training_data': str(data_path),
        'n_features': X.shape[1],
        'algorithm': 'RandomForestRegressor',
        'hyperparameters': {
            'n_estimators': 100,
            'max_depth': 20,
            'min_samples_split': 5
        }
    }

    metadata_file = output_path / 'metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to: {metadata_file}")

    return model, metrics

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python train_qsar.py <data.csv>")
        sys.exit(1)

    train_qsar_model(sys.argv[1])
```

## Model Validation

### Metrics to Check

**Minimum acceptable performance:**
- R² > 0.5 (test set)
- RMSE < 1.0 pIC50 units
- CV R² > 0.4

**Good performance:**
- R² > 0.7
- RMSE < 0.7
- CV R² > 0.6

### Applicability Domain

Monitor out-of-distribution (OOD) predictions:

```python
from sklearn.ensemble import IsolationForest

# Train OOD detector
ood_detector = IsolationForest(contamination=0.1)
ood_detector.fit(X_train)

# Check if new molecule is in domain
is_ood = ood_detector.predict(new_features) == -1
```

## Using Your Model

### 1. Test the Model

```bash
# Test scoring a single molecule
echo "CCO" | python scripts/qsar_scorer.py models/my_target/model.pkl sigmoid

# Expected output:
# {"version": 1, "payload": {"predictions": [0.42]}}
```

### 2. Configure Meta-Controller

In `configs/my_campaign.yaml`:

```yaml
run:
  mode: "hit_discovery"  # or "lead_optimisation"
  n_episodes: 50

qsar:
  model_path: "models/my_target/model.pkl"
  transformation: "sigmoid"  # or "linear", "none"
  uncertainty_threshold: 0.3
  ood_threshold: 0.7

arms:
  - name: "reinvent_qsar_explore"
    scoring:
      - qsar:
          weight: 1.0
          model_path: "models/my_target/model.pkl"
```

### 3. Run Campaign

```bash
# Full pipeline
make run CONFIG=configs/my_campaign.yaml

# Or step-by-step
r4mc init --campaign my_target
r4mc run --config configs/my_campaign.yaml
r4mc report --run-dir out/my_target
```

## Advanced Topics

### Ensemble Models

Combine multiple models for better predictions:

```python
# Train 5 models with different seeds
models = []
for seed in range(5):
    model = train_qsar_model(data, random_state=seed)
    models.append(model)

# Average predictions
predictions = np.mean([m.predict(X) for m in models], axis=0)
```

### Uncertainty Quantification

Use Random Forest variance:

```python
# Get predictions from all trees
tree_predictions = np.array([tree.predict(X) for tree in model.estimators_])

# Calculate uncertainty
mean_pred = tree_predictions.mean(axis=0)
std_pred = tree_predictions.std(axis=0)

# Flag high uncertainty
high_uncertainty = std_pred > threshold
```

### Temporal Validation

For time-series data:

```python
# Split by date instead of random
train_data = df[df['date'] < '2023-01-01']
test_data = df[df['date'] >= '2023-01-01']
```

## Troubleshooting

### Low R² Score
- **More data**: Collect more training compounds
- **Better features**: Try different fingerprints (MACCS, FCFP)
- **Outlier removal**: Remove inconsistent measurements
- **Different algorithm**: Try GradientBoosting, SVM

### Overfitting
- **Reduce complexity**: Lower `max_depth`, increase `min_samples_split`
- **More regularization**: Try Ridge/Lasso
- **Feature selection**: Remove correlated features

### Slow Predictions
- **Smaller ensemble**: Reduce `n_estimators`
- **Batch processing**: Score molecules in batches
- **Simpler model**: Try linear models for speed

## See Also

- [QSAR Scoring Integration](qsar_scoring.md)
- [QSAR Scorer Script](../scripts/README.md)
- [Model Storage Guide](../models/README.md)
