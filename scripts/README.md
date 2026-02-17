# Scripts

Standalone scripts for REINVENT4 integration.

## qsar_scorer.py

**QSAR scoring wrapper for REINVENT4 ExternalProcess component.**

This is a crucial external component that allows you to plug your own QSAR models into REINVENT4 for molecular scoring. It's designed to be called by REINVENT4's `ExternalProcess` scoring component.

### What It Does

- Loads a trained scikit-learn QSAR model (pickle format)
- Receives SMILES strings via stdin
- Calculates molecular features (ECFP4 fingerprints + RDKit descriptors)
- Predicts activity scores
- Transforms predictions to 0-1 range
- Outputs JSON in REINVENT4 ExternalProcess format

### Usage

```bash
python qsar_scorer.py <model_path> <transformation>
```

**Arguments:**
- `model_path`: Path to pickled QSAR model (`.pkl` file)
- `transformation`: Score transformation method
  - `sigmoid`: Sigmoid centered at pActivity 6.5 (recommended)
  - `linear`: Linear scaling from pActivity 4.0-9.0
  - `none`: Raw predictions

**Input:** SMILES strings (one per line) via stdin

**Output:** JSON via stdout:
```json
{
  "version": 1,
  "payload": {
    "predictions": [0.85, 0.42, 0.91, ...]
  }
}
```

### Integration with REINVENT4

Reference this script in your REINVENT4 configuration:

```toml
[scoring.component.qsar]
endpoint = "ExternalProcess"

[scoring.component.qsar.params]
executable = "python"
args = ["scripts/qsar_scorer.py", "models/my_qsar/model.pkl", "sigmoid"]
```

### Feature Calculation

The script calculates:
- **ECFP4 fingerprint** (radius 2, 2048 bits)
- **Molecular descriptors:**
  - Molecular weight
  - LogP
  - TPSA
  - H-bond donors/acceptors
  - Rotatable bonds
  - Aromatic rings
  - Fraction Csp3
  - Ring count
  - Molar refractivity

**Important:** Your QSAR model must be trained on these exact features in this order.

### Example QSAR Model Training

```python
import pickle
from sklearn.ensemble import RandomForestRegressor

# Train your model
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)  # X_train: 2058 features (2048 FP + 10 descriptors)

# Save as pickle
with open('my_qsar_model.pkl', 'wb') as f:
    pickle.dump(model, f)
```

Or save as dict with metadata:
```python
data = {
    'model': model,
    'metadata': {
        'trained_on': 'BRD4 dataset',
        'date': '2026-01-15'
    }
}
pickle.dump(data, f)
```

### Requirements

- Python ≥ 3.10
- RDKit ≥ 2022.9
- NumPy ≥ 1.23
- scikit-learn ≥ 1.0 (if using RandomForest/other sklearn models)

### Error Handling

- Invalid SMILES → score 0.0
- Feature calculation errors → score 0.0
- Warnings printed to stderr (not stdout, so won't interfere with JSON)

### Use Cases

1. **QSAR-guided molecular generation** - Score generated molecules during REINVENT4 runs
2. **Virtual screening** - Score large compound libraries
3. **Lead optimization** - Guide SAR exploration
4. **Multi-objective optimization** - Combine with other scoring functions

### Customization

To use different features or models, modify:
- `calculate_features()`: Change feature extraction
- `transform_score()`: Adjust score transformation
- Model loading: Support different serialization formats

### Example Workflow

```bash
# 1. Train QSAR model
python train_qsar.py --data brd4_training.csv --output models/brd4_qsar/model.pkl

# 2. Test standalone
echo "CCO" | python scripts/qsar_scorer.py models/brd4_qsar/model.pkl sigmoid

# 3. Use in REINVENT4 meta-controller
make run CONFIG=configs/mode_hit_discovery.yaml
```

### See Also

- `docs/qsar_scoring.md` - Integration guide
- `r4mc/components/qsar_component.py` - Meta-controller QSAR component
- REINVENT4 ExternalProcess docs: https://github.com/MolecularAI/REINVENT4
