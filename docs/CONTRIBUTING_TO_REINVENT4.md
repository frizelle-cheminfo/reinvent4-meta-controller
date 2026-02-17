# Contributing QSAR Scoring Example to REINVENT4

## Summary

**YES, this implementation is publication-worthy and could be contributed to REINVENT4!**

However, it's best suited as a **tutorial/example** rather than core functionality, since it demonstrates how to use REINVENT4's existing `ExternalProcess` component rather than adding new features.

## What Makes This Contribution Valuable

1. **Fills Documentation Gap**: REINVENT4 has `ExternalProcess` component but lacks complete ML model examples
2. **Common Use Case**: Many researchers want to integrate custom QSAR models
3. **Best Practices**: Shows proper JSON format, error handling, and path management
4. **Tested & Working**: Proven implementation with real XGBoost model
5. **Well-Documented**: Comprehensive guide with troubleshooting

## Recommended Contribution Options

### Option 1: Tutorial/Documentation (RECOMMENDED)

**What to contribute:**
- Tutorial document: `docs/tutorials/custom_qsar_scoring.md`
- Example wrapper script: `examples/scoring/qsar_scorer.py`
- Example config: `examples/configs/qsar_rl_config.toml`
- Training example: `examples/training/train_simple_qsar.py`

**Advantages:**
- Most helpful to community
- Doesn't require code review for core functionality
- Easy to maintain and update
- Shows real-world usage pattern

**Location in REINVENT4 repo:**
```
REINVENT4/
├── docs/
│   └── tutorials/
│       └── custom_qsar_external_process.md
└── examples/
    ├── scoring/
    │   ├── qsar_scorer.py
    │   └── README.md
    ├── configs/
    │   └── qsar_example.toml
    └── training/
        └── train_example_qsar.py
```

### Option 2: Contrib Package

**What to contribute:**
- Self-contained package in `contrib/scoring/qsar_example/`
- Includes: wrapper, docs, tests, example data

**Location:**
```
REINVENT4/
└── contrib/
    └── scoring/
        └── qsar_sklearn/
            ├── README.md
            ├── qsar_scorer.py
            ├── train_example.py
            ├── example_config.toml
            ├── tests/
            │   └── test_scorer.py
            └── requirements.txt
```

### Option 3: Jupyter Notebook Tutorial

**What to contribute:**
- Interactive notebook: `notebooks/custom_qsar_scoring.ipynb`
- Shows: train model → test wrapper → run REINVENT4 → analyze results

**Advantages:**
- Most accessible for beginners
- Can include visualizations
- End-to-end demonstration

## What NOT to Contribute

### Your Meta-Controller System
The multi-arm bandit controller is **your novel research contribution** and should remain in your own repository. This is publication-worthy research that extends beyond REINVENT4's scope.

Consider:
- Publishing meta-controller as separate package
- Writing a paper about the multi-arm approach
- Linking from REINVENT4 docs as "Advanced Usage"

### BRD4-Specific Code
The BRD4 model and data are domain-specific. Instead:
- Use a generic toy example (e.g., LogP prediction)
- Provide template that users adapt for their targets
- Reference your work as a case study

## Preparing for Contribution

### 1. Create Generic Version

Make the example target-agnostic:

```python
# Generic version for contribution
def calculate_features(mol):
    """
    Calculate molecular features for QSAR prediction.

    Adapt this function to match YOUR training pipeline.
    This example uses ECFP4 + basic descriptors.
    """
    # ECFP4 fingerprint
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)

    # Common descriptors
    descriptors = np.array([
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        # ... etc
    ])

    return np.concatenate([fp, descriptors])
```

### 2. Add Tests

```python
# tests/test_qsar_scorer.py
def test_feature_calculation():
    """Test feature vector dimensions"""
    mol = Chem.MolFromSmiles("CCO")
    features = calculate_features(mol)
    assert len(features) == 2058  # 2048 FP + 10 descriptors

def test_json_output_format():
    """Test output matches ExternalProcess spec"""
    output = score_molecules(["CCO", "c1ccccc1"])
    assert "version" in output
    assert output["version"] == 1
    assert "payload" in output
    assert "predictions" in output["payload"]
```

### 3. Create Minimal Example Data

```python
# examples/training/generate_toy_data.py
"""Generate toy dataset for testing QSAR wrapper"""

# Simple LogP prediction example
training_data = [
    ("CCO", 0.5),           # ethanol
    ("c1ccccc1", 2.1),      # benzene
    ("CC(=O)O", 0.2),       # acetic acid
    # ... 100-200 molecules
]

# Train simple model
X = np.array([calculate_features(smi) for smi, _ in training_data])
y = np.array([activity for _, activity in training_data])

model = RandomForestRegressor(n_estimators=100)
model.fit(X, y)

# Save
with open('example_logp_model.pkl', 'wb') as f:
    pickle.dump(model, f)
```

### 4. Write Clear Documentation

```markdown
# QSAR External Process Example

This example demonstrates how to integrate custom QSAR models
with REINVENT4 using the ExternalProcess scoring component.

## Quick Start

1. Train a model (or use provided example)
2. Configure REINVENT4 to call qsar_scorer.py
3. Run de novo generation with QSAR-guided scoring

## Adapting for Your Model

**CRITICAL**: Feature calculation must match training!
...
```

## Contribution Process

### Step 1: Open Discussion Issue

Create GitHub issue on REINVENT4:

```markdown
Title: [Enhancement] Tutorial for ExternalProcess with ML models

Hi REINVENT4 team,

I've implemented a working example of using ExternalProcess with
scikit-learn QSAR models and would like to contribute it as a tutorial.

**Value to community:**
- Common use case (many researchers need custom ML scoring)
- Working code with real example
- Addresses documentation gap

**Proposed contribution:**
- Tutorial document: docs/tutorials/custom_qsar_scoring.md
- Example wrapper: examples/scoring/qsar_scorer.py
- Minimal training example with toy data

Would this be a valuable addition? Happy to adjust scope/format
based on your preferences.
```

### Step 2: Wait for Feedback

REINVENT4 team will likely:
- ✅ Welcome tutorial/example contributions
- Suggest preferred location and format
- Request specific changes or scope adjustments

### Step 3: Prepare Pull Request

```bash
# Fork REINVENT4
git clone https://github.com/YOUR-USERNAME/REINVENT4.git
cd REINVENT4

# Create branch
git checkout -b feature/qsar-external-process-tutorial

# Add files
cp ~/reinvent4-meta-controller/scripts/qsar_scorer.py \
   examples/scoring/qsar_scorer.py

cp ~/reinvent4-meta-controller/docs/QSAR_SCORING_GUIDE.md \
   docs/tutorials/custom_qsar_scoring.md

# Edit to make generic, add tests, etc.

# Commit
git add .
git commit -m "Add ExternalProcess QSAR scoring tutorial

- Tutorial document with step-by-step guide
- Working wrapper script with error handling
- Example training code with toy data
- Tests for feature calculation and JSON output

Addresses common use case of integrating custom ML models
with REINVENT4's ExternalProcess component."

# Push and create PR
git push origin feature/qsar-external-process-tutorial
```

### Step 4: PR Description

```markdown
## Description

This PR adds a comprehensive tutorial for using ExternalProcess
with custom QSAR models, including:

- 📚 Step-by-step tutorial document
- 💻 Working wrapper script (qsar_scorer.py)
- 🧪 Example training code with toy dataset
- ✅ Tests for validation
- 🐛 Troubleshooting guide

## Motivation

Many researchers want to integrate custom ML models (XGBoost,
Random Forest, Neural Networks) with REINVENT4 for QSAR-guided
generation. While ExternalProcess supports this, there's no
complete working example with ML models.

## Testing

- [x] Wrapper produces correct JSON format
- [x] Feature calculation is consistent
- [x] Works with REINVENT4 v4.5.11
- [x] Tested with XGBoost and RandomForest models
- [x] Error handling works for invalid SMILES

## Documentation

- Tutorial covers installation, configuration, and troubleshooting
- Code is well-commented with docstrings
- Includes example REINVENT4 config files

## Related Issues

Closes #XXX (if there's a related issue)
```

## Alternative: Your Own Repository

If REINVENT4 maintainers prefer minimal examples, consider:

### 1. Standalone Repository

```
qsar-reinvent4-integration/
├── README.md
├── LICENSE
├── docs/
│   └── guide.md
├── qsar_scorer.py
├── train_example.py
├── configs/
│   └── example.toml
└── tests/
    └── test_scorer.py
```

**Advantages:**
- Full control over content and updates
- Can include your specific use cases (BRD4, etc.)
- Citable independently
- More visible as standalone project

### 2. Blog Post / Preprint

Write detailed tutorial on:
- Medium / personal blog
- ChemRxiv preprint
- Journal methods paper

Link from REINVENT4 discussions/issues.

## What Makes This Publishable

Your implementation is publication-worthy because it:

1. **Solves Real Problem**: Many researchers struggle with ML integration
2. **Novel Approach**: Meta-controller with multi-arm bandit (your contribution)
3. **Well-Engineered**: Robust error handling, proper JSON format, absolute paths
4. **Documented**: Clear documentation with troubleshooting
5. **Tested**: Working example with real QSAR model
6. **Reproducible**: All code and instructions provided

## Citation Considerations

### If Contributing to REINVENT4

Your contribution becomes part of REINVENT4:
- Credit in commit history
- Mentioned in release notes
- Referenced in REINVENT4 citations

### If Standalone Repository

You can cite independently:
```bibtex
@software{your_qsar_wrapper,
  title = {QSAR-REINVENT4 Integration: Multi-Arm Bandit Approach},
  author = {Your Name},
  year = {2026},
  url = {https://github.com/yourusername/qsar-reinvent4},
  note = {Wrapper and meta-controller for QSAR-guided de novo design}
}
```

## Recommended Next Steps

1. **Create discussion issue** on REINVENT4 GitHub
2. **Gauge interest** from maintainers
3. **Prepare generic version** based on feedback
4. **Submit PR** with tutorial/example
5. **Publish your meta-controller** separately
6. **Write paper** about multi-arm bandit approach

## Summary

- ✅ Wrapper script: Publishable as REINVENT4 example
- ✅ Tutorial documentation: Valuable community contribution
- ❌ Meta-controller: Your research - publish separately
- ❌ BRD4 specifics: Use generic example for REINVENT4

The QSAR wrapper is production-ready and would be a valuable contribution to REINVENT4's documentation ecosystem. The multi-arm bandit meta-controller is your novel research contribution and deserves independent publication.
