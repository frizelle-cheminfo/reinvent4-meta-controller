# REINVENT 4 Meta-Controller

Adaptive exploration/exploitation for de novo molecular design under real-world constraints.

## What This Does

Generative molecular design typically uses a single fixed strategy (explore broadly OR exploit known actives). This works poorly because drug discovery needs both at different times.

The meta-controller adapts between multiple strategies ("arms") based on performance:
- **Explore** when the search stagnates
- **Exploit** when high-quality seeds are available
- **Transform** existing chemistry when QSAR confidence is high

It also handles real-world constraints:
- QSAR reliability varies (out-of-distribution predictions are less trustworthy)
- Operational mode matters (hit discovery vs lead optimisation need different settings)
- Purchasable analogues inform practical next steps

### Multi-Armed Bandit Strategy

The meta-controller uses a **multi-armed bandit approach** (Upper Confidence Bound / UCB) to adaptively select which generation strategy ("arm") to use for each episode:

1. **Arms Available**: Different generation strategies (e.g., QSAR-guided exploration, scaffold hopping, close analogue generation)
2. **Reward Signal**: Episode quality metrics (score improvement, diversity, novelty)
3. **UCB Policy**: Balances exploitation (arms with proven performance) vs exploration (untried or uncertain arms)
4. **Hard Rules**: Override UCB when certain conditions trigger (e.g., stagnation → force exploration)

**Cold Start**: Initially, all arms are tried equally to gather performance data. After 5-10 episodes, the bandit learns which arms work best for your target.

**Benefits**:
- No manual strategy switching required
- Adapts to campaign-specific performance patterns
- Avoids premature convergence to suboptimal strategies

### Exploration Strategy Through Staged Learning

The meta-controller naturally progresses through chemical space via **staged learning**:

**Phase 1: Broad Exploration (Episodes 1-10)**
- High novelty emphasis
- Diverse arm selection
- Building initial seed bank
- Mapping chemical space boundaries

**Phase 2: Focused Exploitation (Episodes 10-30)**
- Arms that refine promising scaffolds get priority
- QSAR confidence guides selection
- Seed bank drives scaffold hopping
- SAR patterns emerge

**Phase 3: SAR Refinement (Episodes 30+)**
- Conservative transformations (MMP-based)
- High-similarity analogues
- Purchasable constraint emphasis
- Synthesis-ready optimization

This progression happens **automatically** via the bandit policy - no manual intervention needed. The system learns which arms perform best at each stage based on actual results.

## Key Features

- **Adaptive arm selection** via bandit policy (UCB) + hard-rule triggers
- **Operational modes** for hit discovery vs lead optimisation
- **Medchem handoff reports** - Bridges computational and wet lab work
  - Synthesis priorities with confidence scores
  - Commercial analogue matching (MolPort, ZINC, Enamine)
  - Product codes for immediate ordering
  - Risk assessment (synthesis vs purchase)
  - Actionable recommendations for chemists
- **Purchasable constraint layer** maps generated molecules to commercial catalogues
- **QSAR-guided generation** with real-time scoring via ExternalProcess
- **CPU-runnable demo** with toy data (no proprietary dependencies)
- **Memory-efficient batching** for low-memory systems (configurable batch sizes)
- **Bring-your-own models** - Train QSAR models on your data and plug them in

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run demo (5 minutes on CPU)
make demo

# View outputs
cat out/demo_report/report.md
open out/demo_report/medchem_handoff/index.html
```

## CLI Usage

The meta-controller provides a complete command-line interface for running campaigns:

### Initialize a Campaign

```bash
# Create new campaign
r4mc init --campaign my_target

# Initialize with custom config
r4mc init --campaign my_target --config configs/my_config.yaml
```

### Run Episodes

**Single Episode** (useful for testing or manual control):
```bash
# Run one episode
r4mc run-episode --campaign my_target

# Run with lite mode (faster, smaller batches)
r4mc run-episode --campaign my_target --lite

# Dry-run mode (simulate without REINVENT4)
r4mc run-episode --campaign my_target --dry-run
```

**Full Campaign** (run multiple episodes):
```bash
# Run 50 episodes
r4mc run --campaign my_target --episodes 50

# Run with custom REINVENT4 binary
r4mc run --campaign my_target --episodes 50 --reinvent-bin /path/to/reinvent

# Run from custom config
r4mc run --campaign my_target --episodes 50 --config configs/my_config.yaml
```

### Generate Reports

```bash
# Generate behavioral report
r4mc report --campaign my_target

# Generate medchem handoff report
python -m r4mc.report_engine.cli medchem --run-dir runs/my_target
```

### Utility Commands

```bash
# List available arms
r4mc list-arms

# Check which priors are available
r4mc check-priors --priors-dir ./priors

# Clean temporary files
r4mc clean --campaign my_target

# Show version
r4mc version
```

### Memory Management and Batching

For systems with limited memory, configure batch sizes in your config:

```yaml
episode:
  steps: 1000           # Total molecules per episode
  batch_size: 64        # Molecules per batch (reduce for low memory)

# For very low memory systems (< 8GB RAM):
episode:
  steps: 500
  batch_size: 32

# For high-memory systems (> 32GB RAM):
episode:
  steps: 2000
  batch_size: 128
```

**How batching works:**
1. REINVENT4 generates molecules in batches (e.g., 64 at a time)
2. Each batch is scored via QSAR scorer
3. Gradients computed and model updated
4. Next batch generated
5. Memory freed between batches

This allows running large campaigns even on modest hardware. The `--lite` flag automatically sets `batch_size=50` and `steps=100` for quick testing.

## QSAR Scorer Integration

The `scripts/qsar_scorer.py` script is a crucial standalone component that scores molecules **in real-time during episode runs**. REINVENT4 calls it via ExternalProcess to score each generated molecule, and those scores guide the next generation step.

```bash
# Test QSAR scorer
echo "CCO" | python scripts/qsar_scorer.py models/my_model.pkl sigmoid

# Output: {"version": 1, "payload": {"predictions": [0.42]}}
```

**How it works:**
1. REINVENT4 generates SMILES during episode
2. Calls `qsar_scorer.py` via stdin (batch of SMILES)
3. Script calculates features (ECFP4 + descriptors) and predicts scores
4. Returns JSON with predictions to REINVENT4
5. REINVENT4 uses scores to guide next generation step

**Features:**
- Loads any scikit-learn QSAR model (pickle format)
- Calculates ECFP4 fingerprints + molecular descriptors
- Real-time scoring during generation (not post-processing)
- Outputs JSON in REINVENT4 ExternalProcess format
- Pluggable into any REINVENT4 workflow

See [scripts/README.md](scripts/README.md) and [docs/qsar_scoring.md](docs/qsar_scoring.md) for integration details.

## Training Your Own QSAR Models

The meta-controller is designed to work with **your own QSAR models** trained on your target-specific data. This is critical for real-world applications where public models are insufficient.

### Quick QSAR Training Example

```python
# Train a QSAR model in 5 steps
import pandas as pd
from scripts.train_qsar import train_qsar_model

# 1. Prepare your data (SMILES + activity)
data = pd.read_csv('my_training_data.csv')
# Required columns: 'smiles', 'pActivity' (or IC50, Kd, etc.)

# 2. Train model
model, metrics = train_qsar_model(
    data=data,
    smiles_col='smiles',
    target_col='pActivity',
    output_path='models/my_target/model.pkl'
)

# 3. Check performance
print(f"Test R²: {metrics['test_r2']:.3f}")  # Should be > 0.6
print(f"Test RMSE: {metrics['test_rmse']:.3f}")

# 4. Test the scorer
# echo "CCO" | python scripts/qsar_scorer.py models/my_target/model.pkl sigmoid
```

### Data Requirements

- **Minimum**: 300+ compounds (for hit discovery)
- **Recommended**: 500+ compounds (for lead optimization)
- **Format**: CSV with `smiles` and activity column (pIC50, pKi, etc.)
- **Features**: Automatically calculated (ECFP4 + 10 molecular descriptors)

### Model Performance Guidelines

**Acceptable:**
- Test R² > 0.5
- RMSE < 1.0 pIC50 units

**Good:**
- Test R² > 0.7
- RMSE < 0.7 pIC50 units

### Complete Training Guide

See **[docs/qsar_model_training.md](docs/qsar_model_training.md)** for a complete guide including:
- Data preparation and formatting
- Feature calculation (2058 features: ECFP4 + descriptors)
- Full training script with cross-validation
- Model validation and metrics
- Integration with meta-controller
- Troubleshooting common issues
- Advanced topics (ensembles, uncertainty quantification)

## Purchasable Constraints Configuration

The meta-controller can map generated molecules to **commercially available analogues** from MolPort, ZINC, Enamine, and other vendors. This enables:

1. **Immediate testing**: Order analogues instead of synthesizing
2. **Risk mitigation**: Validate computationally-designed molecules experimentally
3. **Prioritization**: Re-rank candidates based on sourceability

### Obtaining a Purchasable Library

**Option 1: MolPort Export** (Recommended)

1. Visit [MolPort](https://www.molport.com)
2. Navigate to "Download Database"
3. Select filter criteria:
   - Drug-like compounds (Lipinski's Rule of Five)
   - In-stock compounds only
   - Molecular weight: 200-500 Da
4. Export as CSV with columns:
   - `smiles`: SMILES string
   - `molport_id`: Product code (e.g., "MolPort-001-234-567")
   - `supplier`: Vendor name
   - `price_mg`: Price per mg (optional)
   - `stock_g`: Stock in grams (optional)
5. Save to `data/purchasable_library.csv`

**Option 2: ZINC15** (Free, academic use)

```bash
# Download ZINC15 in-stock subset
wget http://files.docking.org/catalogs/13/zinc15_now_3D.csv.gz
gunzip zinc15_now_3D.csv.gz

# Convert to required format
python scripts/convert_zinc_to_molport.py zinc15_now_3D.csv data/purchasable_library.csv
```

**Option 3: Enamine REAL** (Ultra-large library)

- Visit [Enamine REAL](https://enamine.net/compound-collections/real-compounds)
- Download subset relevant to your target
- Convert to standard format

### Configure in Meta-Controller

```yaml
# In your campaign config YAML
purchasable:
  enabled: true
  csv_path: "data/purchasable_library.csv"
  similarity_threshold: 0.7  # Tanimoto similarity cutoff
  max_analogues: 5           # Top-K analogues to report per molecule

# Required CSV columns:
# - smiles (required)
# - product_code (required, e.g., MolPort-001-234-567)
# - supplier (optional)
# - price (optional)
# - stock (optional)
```

### Example MolPort CSV Format

```csv
smiles,product_code,supplier,price_mg,stock_g
CCO,MolPort-001-234-567,Sigma-Aldrich,0.50,10.0
c1ccccc1,MolPort-002-345-678,Enamine,0.30,25.0
CC(C)O,MolPort-003-456-789,ChemDiv,0.75,5.0
```

### Using Purchasable Constraints

The medchem handoff report will automatically include:
- Top-3 purchasable analogues per generated molecule
- Tanimoto similarity scores
- Product codes for immediate ordering
- Supplier and pricing information (if available)

```bash
# Generate medchem report with purchasable mapping
python -m r4mc.report_engine.cli medchem \
  --run-dir runs/my_target \
  --purchasable-csv data/purchasable_library.csv \
  --similarity-threshold 0.7
```

**Output:** Prioritized synthesis list with "Order Now" vs "Synthesize" recommendations based on analogue availability.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Meta-Controller                                            │
│                                                             │
│  ┌───────────────┐      ┌──────────────┐                  │
│  │ Bandit Policy │─────▶│ Arm Selector │                  │
│  │ (UCB)         │      │              │                  │
│  └───────────────┘      └──────┬───────┘                  │
│                                │                            │
│                                ▼                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Arms (Strategies)                                   │  │
│  │                                                      │  │
│  │  • reinvent_qsar_explore   (novel + predicted)     │  │
│  │  • reinvent_explore        (broad diversity)        │  │
│  │  • reinvent_exploit        (refine seeds)           │  │
│  │  • mol2mol_qsar_high_sim   (scaffold hop)           │  │
│  │  • mol2mol_high_sim_exploit (close analogues)       │  │
│  │  • mol2mol_mmp_exploit     (conservative SAR)       │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌──────────────┐      ┌─────────────────────────────────┐  │
│  │  Seed Bank   │◀─────│ Episode Run (1000 steps)       │  │
│  │  (Top Mols)  │      │                                 │  │
│  └──────────────┘      │  ┌───────────────────────────┐ │  │
│                        │  │ REINVENT4 generates SMILES│ │  │
│                        │  └──────────┬────────────────┘ │  │
│                        │             ▼                   │  │
│                        │  ┌───────────────────────────┐ │  │
│                        │  │ QSAR Scorer (real-time)   │ │  │
│                        │  │ scripts/qsar_scorer.py    │ │  │
│                        │  │ (ExternalProcess)         │ │  │
│                        │  └──────────┬────────────────┘ │  │
│                        │             ▼                   │  │
│                        │  Scores guide next generation   │  │
│                        └─────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│  Post-Processing (after episode completes)                  │
│                                                             │
│  • OOD detection analysis                                  │
│  • Uncertainty quantification                              │
│  • Purchasable analogue mapping                            │
│  • Novelty vs score frontier                               │
│  • Diversity metrics                                       │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│  Outputs (r4mc.report_engine)                               │
│                                                             │
│  • Behavioral report (stability, diversity, OOD)           │
│  • Medchem handoff (priorities + product codes)            │
│  • Run manifest (reproducibility)                          │
│                                                             │
│  CLI: python -m r4mc.report_engine.cli build/medchem       │
└─────────────────────────────────────────────────────────────┘

External Components:
  • scripts/qsar_scorer.py - QSAR scoring for REINVENT4 ExternalProcess
  • priors/ - REINVENT4 prior models (download from Zenodo)
  • models/ - Your trained QSAR models (scikit-learn pickles)
```

## Operational Modes

### Hit Discovery
- Tolerates high OOD (exploring new chemical space)
- QSAR is directional, not quantitative
- Emphasizes novelty and diversity
- Accepts uncertain predictions

**Use when:**
- Target is novel with sparse training data
- Goal is to find new chemotypes
- Willing to accept false positives

### Lead Optimisation
- Stays within QSAR applicability domain
- Penalizes high uncertainty
- Emphasizes ranking stability
- Conservative exploit arms

**Use when:**
- Validated hit series exists
- Dense SAR data available
- Need reliable ranking for synthesis prioritisation

See [`docs/operational_modes.md`](docs/operational_modes.md) for details.

## Purchasable Constraints

Generated molecules are mapped to commercially available analogues (MolPort, ZINC, Enamine). This provides:

1. **Reality check**: What can actually be tested?
2. **Validation strategy**: Test analogues before committing to synthesis
3. **Re-ranked priorities**: Adjust for sourceability

See [`docs/purchasable_constraints.md`](docs/purchasable_constraints.md) for how this works.

## Configuration

```yaml
run:
  mode: "hit_discovery"  # or "lead_optimisation"
  n_episodes: 50

qsar:
  uncertainty_threshold: 0.3  # Higher for hit discovery
  ood_threshold: 0.7          # Higher for hit discovery

arms:
  - name: "reinvent_qsar_explore"
    scoring:
      - qsar: 1.0
      - novelty: 0.5  # Higher for hit discovery

purchasable:
  enabled: true
  csv_path: "data/purchasable_library.csv"
  similarity_threshold: 0.7
```

See [`configs/`](configs/) for examples.

## Example: Running a Campaign

```bash
# Hit discovery mode
make run CONFIG=configs/mode_hit_discovery.yaml

# Generate reports
make report RUN_DIR=out/hit_discovery
make medchem RUN_DIR=out/hit_discovery

# Review outputs
open out/hit_discovery_report/report.html
cat out/hit_discovery_medchem/handoff.md
```

## Medchem Handoff Report

The handoff report translates computational output into synthesis priorities:

```markdown
# Top Candidates

1. Molecule A (Score: 0.89, High Confidence)
   - Predicted pIC50: 7.8 nM
   - Purchasable analogue: MOLPORT-12345 (Sim: 0.82)
   - Recommendation: Order analogue for validation

2. Molecule B (Score: 0.86, Medium Confidence)
   - Predicted pIC50: 7.5 nM
   - Purchasable analogue: None (Sim: 0.58)
   - Recommendation: Requires synthesis, high risk
```

See [`docs/medchem_handoff.md`](docs/medchem_handoff.md) for format details.

## Outputs

```
out/my_run_report/
├── report.md              # Behavioral summary
├── manifest.json          # Reproducibility metadata
├── tables/
│   ├── stability.csv      # Top-N Jaccard over time
│   ├── diversity.csv      # Chemical diversity metrics
│   ├── ood_uncertainty.csv
│   └── reward.csv         # Score decomposition
└── medchem_handoff/
    ├── handoff.md         # Human-readable priorities
    ├── candidates.csv     # Full data table
    └── summary.json
```

## Requirements

- Python ≥ 3.10
- RDKit ≥ 2022.9
- pandas ≥ 1.5
- REINVENT 4 (for actual generation)
- REINVENT4 Priors - https://zenodo.org/records/15641297

Demo runs without REINVENT (uses synthetic data).

## Installation

```bash
# Clone repository
git clone https://github.com/frizelle-cheminfo/reinvent4-meta-controller.git
cd reinvent4-meta-controller

# Install Python package
pip install -e ".[dev]"

# Download REINVENT4 priors (required for full functionality)
# Visit: https://zenodo.org/records/15641297
# Place .prior files in priors/ directory
# See priors/README.md for details
```

**Note**: The demo works without downloading priors (uses synthetic data), but full functionality requires REINVENT4 priors from Zenodo.

## Testing

```bash
make test          # Run all tests
make test-fast     # Skip slow tests
make lint          # Check code style
make check         # Lint + test
```

## Folder Structure

```
reinvent4-meta-controller/
├── r4mc/                       # Main package
│   ├── __init__.py
│   ├── controller.py           # Meta-controller orchestration
│   ├── arms.py                 # Arm definitions
│   ├── episode.py              # Episode execution logic
│   ├── seeds.py                # Seed bank management
│   ├── config.py               # Configuration handling
│   ├── cli.py                  # Command-line interface
│   ├── components/             # QSAR and other components
│   │   └── qsar_component.py   # QSAR integration
│   ├── templates/              # Config templates for REINVENT4
│   └── report_engine/          # Report generation
│       ├── cli.py              # Report CLI (build, medchem, doctor)
│       ├── report/             # Report building
│       ├── metrics/            # Metrics calculation
│       ├── molport/            # Purchasable analogue matching
│       └── io/                 # Data loading
├── scripts/
│   ├── qsar_scorer.py          # QSAR scoring wrapper for REINVENT4
│   └── README.md               # QSAR scorer documentation
├── priors/
│   └── README.md               # Zenodo download link
├── models/
│   └── README.md               # QSAR model training guide
├── configs/                    # Configuration examples
│   ├── demo.yaml
│   ├── mode_hit_discovery.yaml
│   └── mode_lead_opt.yaml
├── data/                       # Toy datasets
│   ├── toy_training_set.csv
│   └── toy_purchasable_stub.csv
├── docs/                       # Documentation
│   ├── arms.md
│   ├── operational_modes.md
│   ├── purchasable_constraints.md
│   ├── medchem_handoff.md
│   └── qsar_scoring.md         # QSAR integration guide
├── tests/                      # Test suite
├── README.md
├── pyproject.toml
├── Makefile
└── LICENSE
```

## Documentation

- [Arms (Strategies)](docs/arms.md) - Available generation strategies and when to use them
- [Operational Modes](docs/operational_modes.md) - Hit discovery vs lead optimization settings
- [QSAR Model Training Guide](docs/qsar_model_training.md) - **Complete guide to training your own QSAR models**
- [QSAR Scoring Integration](docs/qsar_scoring.md) - How QSAR scoring works with REINVENT4
- [QSAR Scorer Script](scripts/README.md) - Technical details on the scoring wrapper
- [Purchasable Constraints](docs/purchasable_constraints.md) - Mapping molecules to commercial catalogues
- [Medchem Handoff](docs/medchem_handoff.md) - Report format for wet lab teams
- [Model Storage Guide](models/README.md) - How to organize and use QSAR models
- [Priors Download](priors/README.md) - REINVENT4 priors from Zenodo

## Comparison to Alternatives

| Approach | Pros | Cons |
|----------|------|------|
| Single-arm (explore) | Broad diversity | Can't refine promising chemistry |
| Single-arm (exploit) | Local optimization | Stagnates, no novelty |
| Fixed schedule | Simple | Doesn't adapt to context |
| **Meta-controller** | Adapts to performance | More complex to configure |

## Limitations

1. **QSAR dependency**: Performance limited by QSAR quality
2. **Bandit cold start**: Needs initial episodes to learn arm performance
3. **Computational cost**: Multiple arms mean more REINVENT runs
4. **Configuration tuning**: Requires domain knowledge to set thresholds

## When to Use This

**Good fit:**
- Multi-objective optimization with trade-offs
- Campaigns lasting 20+ episodes
- Situations where exploration vs exploitation matters
- Projects with synthesis constraints

**Not needed:**
- Single-objective, well-defined target
- Very short campaigns (< 10 episodes)
- Pure diversity generation
- Proof-of-concept demos

## Citation

If you use this in your work:

```bibtex
@software{reinvent4_meta_controller,
  title={REINVENT4-Meta-Controller},
  author={Mitchell Frizelle},
  year={2026},
  url={https://github.com/frizelle-cheminfo/reinvent4-meta-controller}
}
```

Also cite:
- REINVENT 4: [citation]
- RDKit: [citation]

## License

MIT License - see [LICENSE](LICENSE)

## Contributing

PRs welcome for:
- New arm strategies
- Improved bandit policies
- Additional operational modes
- Better default configurations

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Acknowledgments

Built on:
- REINVENT 4 (generative models)
- RDKit (cheminformatics)
- MolPort/ZINC/Enamine (purchasable catalogues)

## Support

- Issues: [GitHub Issues](https://github.com/frizelle-cheminfo/reinvent4-meta-controller/issues)
- Discussions: [GitHub Discussions](https://github.com/frizelle-cheminfo/reinvent4-meta-controller/discussions)

---

