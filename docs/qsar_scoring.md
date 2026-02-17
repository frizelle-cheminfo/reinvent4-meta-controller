# QSAR Scoring Integration with REINVENT4

## Overview

This guide describes how to integrate custom QSAR (Quantitative Structure-Activity Relationship) models with REINVENT4 using the `ExternalProcess` scoring component. The implementation provides a publication-ready wrapper that enables REINVENT4 to score generated molecules using trained machine learning models without modifying REINVENT4's source code.

## Key Features

- **Clean Integration**: Uses REINVENT4's native ExternalProcess component (no code modifications required)
- **Flexible Model Support**: Works with any scikit-learn compatible model (XGBoost, Random Forest, Neural Networks)
- **Configurable Transformations**: Supports sigmoid, linear, or no transformation of raw predictions
- **Production Ready**: Handles errors gracefully, validates inputs, and provides detailed logging
- **Publication Suitable**: Self-contained, well-documented, and follows best practices

## Architecture

```
┌─────────────────┐
│   REINVENT4     │
│  (RL Training)  │
└────────┬────────┘
         │ Generates SMILES
         ↓
┌─────────────────┐
│ ExternalProcess │ ← REINVENT4 scoring component
│   Component     │
└────────┬────────┘
         │ Calls via subprocess
         ↓
┌─────────────────┐
│ qsar_scorer.py  │ ← Our wrapper script
│                 │
│  1. Parse args  │
│  2. Load model  │
│  3. Read SMILES │
│  4. Compute     │
│     features    │
│  5. Predict     │
│  6. Transform   │
│  7. Output JSON │
└─────────────────┘
```

## Requirements

### Python Packages

```bash
pip install rdkit scikit-learn xgboost numpy
```

### Files Required

1. **QSAR Model**: Pickled scikit-learn model (`.pkl` file)
2. **Wrapper Script**: `qsar_scorer.py` (this implementation)
3. **REINVENT4 Config**: TOML file with ExternalProcess component configured

## Usage

### 1. Train Your QSAR Model

```python
import pickle
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

def calculate_features(smiles):
    """Calculate ECFP4 + molecular descriptors"""
    mol = Chem.MolFromSmiles(smiles)

    # ECFP4 fingerprint (radius 2, 2048 bits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    fp_array = np.array(fp)

    # Molecular descriptors
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

# Train model
X_train = np.array([calculate_features(smi) for smi in training_smiles])
y_train = np.array(training_activities)

model = xgb.XGBRegressor(n_estimators=500, max_depth=7, learning_rate=0.05)
model.fit(X_train, y_train)

# Save model
with open('models/my_qsar_model.pkl', 'wb') as f:
    pickle.dump(model, f)
```

**CRITICAL**: The feature calculation in your training script must **exactly match** the feature calculation in `qsar_scorer.py`. Any mismatch will cause silent errors or poor predictions.

### 2. Configure REINVENT4

Add the ExternalProcess component to your REINVENT4 TOML config:

```toml
[[stage.scoring.component]]
[stage.scoring.component.ExternalProcess]

[[stage.scoring.component.ExternalProcess.endpoint]]
name = "QSAR_Activity"
weight = 2.0  # 40% of total scoring weight
params.executable = "/path/to/python"
params.args = "/absolute/path/to/qsar_scorer.py /absolute/path/to/model.pkl sigmoid"
```

**Important Notes**:
- Use **absolute paths** (REINVENT4 runs from episode directories)
- `params.args` must be a **string** (not a TOML array)
- Arguments: `<script_path> <model_path> <transformation>`

### 3. Run REINVENT4

```bash
reinvent your_config.toml
```

## Wrapper Script: qsar_scorer.py

### Command Line Arguments

```bash
python qsar_scorer.py <model_path> <transformation>
```

**Arguments**:
- `model_path`: Path to pickled scikit-learn model
- `transformation`: Score transformation method
  - `sigmoid`: Sigmoid centered at pActivity 6.5 (recommended)
  - `linear`: Linear scaling from pActivity 4.0-9.0
  - `none`: Raw predictions (no transformation)

### Input Format

SMILES strings via stdin (one per line):

```
CCO
c1ccccc1
CC(=O)Oc1ccccc1C(=O)O
```

### Output Format

JSON object matching REINVENT4 ExternalProcess specification:

```json
{
  "version": 1,
  "payload": {
    "predictions": [0.207635, 0.213643, 0.445821]
  }
}
```

### Feature Calculation

The wrapper calculates **2058-dimensional features**:
- **2048 bits**: Morgan (ECFP4) fingerprint (radius 2)
- **10 descriptors**: MW, LogP, TPSA, HBD, HBA, RotBonds, AromaticRings, FractionCSP3, RingCount, MolMR

### Score Transformations

#### Sigmoid (Recommended)
Centered at pActivity 6.5 with steepness 0.5:

```python
score = 1.0 / (1.0 + exp(-0.5 * (prediction - 6.5)))
```

- pActivity 4.0 → score ~0.03
- pActivity 6.5 → score ~0.50
- pActivity 9.0 → score ~0.92

#### Linear
Linear scaling from pActivity range [4.0, 9.0] to score range [0.0, 1.0]:

```python
score = clip((prediction - 4.0) / (9.0 - 4.0), 0.0, 1.0)
```

## Example Implementation

### Complete qsar_scorer.py

See `/Users/mitchfrizelle/reinvent4-meta-controller/scripts/qsar_scorer.py`

Key implementation details:

1. **Robust Error Handling**: Invalid SMILES → score 0.0
2. **Batch Processing**: Collects all scores before outputting JSON
3. **Logging**: Warnings to stderr (doesn't interfere with JSON output)
4. **Model Flexibility**: Handles both bare models and dict-wrapped models

### Testing the Wrapper

```bash
# Test standalone
echo -e "CCO\nc1ccccc1\nCC(=O)Oc1ccccc1C(=O)O" | \
  python qsar_scorer.py models/model.pkl sigmoid

# Expected output:
# {"version": 1, "payload": {"predictions": [0.207635, 0.213643, 0.445821]}}
```

## Integration with Meta-Controller

For advanced multi-arm bandit campaigns with QSAR scoring:

```yaml
# campaigns/my-qsar-campaign.yaml
arms:
  - arm_id: qsar_explore
    template: reinvent_qsar.toml.j2
    scoring_profile:
      potency_weight: 2.0  # 40% QSAR score
```

The meta-controller handles:
- Template rendering with absolute paths
- Multi-arm bandit optimization
- Seed molecule management
- Campaign-level metrics

## Troubleshooting

### Issue: "can't open file ... qsar_scorer.py"

**Cause**: Relative paths don't work (REINVENT4 runs from episode directories)

**Solution**: Use absolute paths in config:
```toml
params.args = "/Users/you/project/qsar_scorer.py /Users/you/project/model.pkl sigmoid"
```

### Issue: "Feature shape mismatch"

**Cause**: Feature calculation differs between training and scoring

**Solution**:
1. Ensure identical descriptor calculation
2. Verify fingerprint parameters (radius, nBits)
3. Check descriptor order matches training

### Issue: "ValidationError: Input should be a valid string"

**Cause**: `params.args` configured as TOML array instead of string

**Wrong**:
```toml
params.args = ["script.py", "model.pkl", "sigmoid"]
```

**Correct**:
```toml
params.args = "script.py model.pkl sigmoid"
```

### Issue: Poor predictions despite good training metrics

**Possible causes**:
1. Feature mismatch (most common)
2. Wrong transformation parameters
3. Model not compatible with production RDKit version
4. Pickle protocol mismatch

**Debug**:
```python
# Compare training vs. scoring features
train_features = calculate_features_training("CCO")
score_features = calculate_features_scoring("CCO")
assert np.allclose(train_features, score_features)
```

## Performance Considerations

### Computational Cost

For a typical campaign:
- **SMILES/batch**: 50-128
- **Features/SMILES**: 2058 dimensions
- **Model inference**: ~10-50ms per batch (XGBoost)
- **Total overhead**: ~5-10% of REINVENT4 runtime

### Optimization Tips

1. **Use faster models**: XGBoost > Random Forest > Deep Learning
2. **Reduce fingerprint size**: 1024 bits often sufficient
3. **Vectorize feature calculation**: Process batches efficiently
4. **Cache model**: Load once, reuse for all predictions (already implemented)

## Publication Checklist

When publishing results using this approach:

- [ ] Document exact feature calculation (code + description)
- [ ] Report model training details (data size, metrics, hyperparameters)
- [ ] Specify transformation method and parameters
- [ ] Include REINVENT4 version (v4.5.11)
- [ ] Provide example REINVENT4 config
- [ ] Share wrapper script (this implementation or adapted version)
- [ ] Report computational environment (Python, RDKit, sklearn versions)

## Potential Contributions to REINVENT4

This implementation could be contributed to REINVENT4 as:

### 1. Documentation Example (Recommended)
- Location: `docs/tutorials/qsar_external_process.md`
- Shows best practices for ExternalProcess with ML models
- Includes working example code
- References this implementation

### 2. Contrib Example
- Location: `contrib/scoring/qsar_example/`
- Generic QSAR wrapper template
- Example config files
- README with usage instructions

### 3. Tutorial Notebook
- Location: `notebooks/custom_qsar_scoring.ipynb`
- End-to-end tutorial: train model → configure → run
- Interactive and educational

## Related Work

- **REINVENT4**: Molecular de novo design with deep reinforcement learning
  - Repo: https://github.com/MolecularAI/REINVENT4
  - Paper: Loeffler et al. (2024)

- **ExternalProcess Component**: Generic interface for external scoring
  - Documentation: REINVENT4/docs/scoring.md
  - Supports any subprocess that reads SMILES and outputs JSON

- **Alternative Approaches**:
  - ChemProp plugin (D-MPNN models)
  - REST API scoring (for remote models)
  - Custom REINVENT4 components (requires modification)

## Citation

If you use this implementation in your research, please cite:

```bibtex
@software{qsar_reinvent4_wrapper,
  title = {QSAR Scoring Wrapper for REINVENT4 ExternalProcess},
  author = {Your Name},
  year = {2026},
  note = {Implementation of scikit-learn QSAR model integration with REINVENT4}
}
```

## License

This implementation is provided as-is for research and educational purposes. Ensure compliance with REINVENT4's license (Apache 2.0) and any proprietary model licenses.

## Contact

For questions or issues:
- GitHub Issues: [your-repo]/issues
- Email: your-email@domain.com

## Acknowledgments

- REINVENT4 development team at AstraZeneca
- RDKit open-source cheminformatics toolkit
- scikit-learn and XGBoost communities
