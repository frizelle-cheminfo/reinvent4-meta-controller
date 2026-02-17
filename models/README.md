# QSAR Models

This directory should contain your trained QSAR models for scoring molecules.

## Structure

Each model should be in its own subdirectory:

```
models/
├── README.md
├── my_target/
│   ├── model.pkl              # Trained scikit-learn model
│   ├── training_report.txt    # Optional: training metrics
│   └── metadata.json          # Optional: model metadata
└── another_target/
    └── model.pkl
```

## Model Format

Models should be pickled scikit-learn regressors trained on:
- **ECFP4 fingerprints** (radius 2, 2048 bits)
- **10 molecular descriptors** (MW, LogP, TPSA, HBD, HBA, rotatable bonds, aromatic rings, fraction Csp3, ring count, molar refractivity)

Total: 2058 features

## Creating a Model

### Training Script Example

```python
import pandas as pd
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Load your data
df = pd.read_csv('training_data.csv')  # Columns: smiles, pActivity

# Calculate features
def get_features(smiles):
    mol = Chem.MolFromSmiles(smiles)

    # ECFP4
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    fp_array = np.array(fp)

    # Descriptors
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

X = np.array([get_features(s) for s in df['smiles']])
y = df['pActivity'].values

# Train model
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)
model.fit(X, y)

# Save model
import os
os.makedirs('models/my_target', exist_ok=True)
with open('models/my_target/model.pkl', 'wb') as f:
    pickle.dump(model, f)
```

### Model Metadata (Optional)

Save metadata alongside your model:

```json
{
  "target": "BRD4",
  "training_size": 1500,
  "test_r2": 0.72,
  "features": "ECFP4_2048 + 10_descriptors",
  "algorithm": "RandomForestRegressor",
  "date_trained": "2026-02-15",
  "notes": "Trained on ChEMBL BRD4 data"
}
```

## Using Models

Reference your model in REINVENT4 configs:

```toml
[scoring.component.qsar]
endpoint = "ExternalProcess"

[scoring.component.qsar.params]
executable = "python"
args = ["scripts/qsar_scorer.py", "models/my_target/model.pkl", "sigmoid"]
```

Or in meta-controller configs:

```yaml
arms:
  - name: "reinvent_qsar_explore"
    scoring:
      - qsar:
          model_path: "models/my_target/model.pkl"
          transformation: "sigmoid"
          weight: 1.0
```

## Supported Algorithms

The `scripts/qsar_scorer.py` works with any scikit-learn regressor:
- RandomForestRegressor (recommended)
- GradientBoostingRegressor
- SVR
- Ridge/Lasso
- Any model with `.predict()` method

## Validation

Test your model before using:

```bash
# Test scoring
echo "CCO" | python scripts/qsar_scorer.py models/my_target/model.pkl sigmoid

# Expected output:
# {"version": 1, "payload": {"predictions": [0.42]}}
```

## Model Performance

Good QSAR model characteristics:
- **R² > 0.6** on test set
- **RMSE < 0.8** pIC50 units
- **Diverse training set** (500+ compounds recommended)
- **Applicability domain** monitoring (OOD detection)

## See Also

- `scripts/qsar_scorer.py` - Scoring script
- `scripts/README.md` - QSAR scorer documentation
- `docs/qsar_scoring.md` - Integration guide
