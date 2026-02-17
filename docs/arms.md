# Arms (Strategies)

An "arm" is a specific molecular generation strategy. The meta-controller adapts between arms based on performance.

## Why Multiple Arms?

Different stages of drug discovery need different approaches:

- **Early exploration** benefits from broad chemical space coverage
- **Focused SAR** needs local perturbations around known actives
- **Lead optimisation** requires conservative, reliable predictions

A single fixed strategy underperforms because it can't adapt to changing context.

## Available Arms

### REINVENT Arms (De Novo)

**`reinvent_qsar_explore`**
- Generates novel molecules guided by QSAR predictions
- Balanced between activity and novelty
- Good for: finding new chemotypes with predicted activity

**`reinvent_explore`**
- Maximizes chemical diversity and novelty
- Minimal scoring bias
- Good for: broad exploration when QSAR is unreliable

**`reinvent_exploit`**
- Refines and optimizes around high-scoring seeds
- Emphasizes similarity to known actives
- Good for: optimising validated chemistry

### Mol2Mol Arms (Scaffold Hopping / Transformation)

**`mol2mol_qsar_high_sim`**
- Transforms seed molecules to improve activity
- Maintains moderate structural similarity
- Good for: scaffold hopping with activity improvement

**`mol2mol_high_sim_exploit`**
- Creates close analogues of top performers
- High structural similarity constraint
- Good for: lead optimisation, SAR expansion

**`mol2mol_mmp_exploit`**
- Applies matched molecular pair (MMP) transforms
- Very conservative structural changes
- Good for: systematic SAR with high confidence

## Arm Selection

The controller uses a bandit policy (typically Upper Confidence Bound) to balance:
- **Exploitation**: Choose the best-performing arm
- **Exploration**: Try underexplored arms

Hard-rule triggers can override the bandit:
- If seed bank has high-quality seeds → switch to exploit arm
- If stagnation detected → switch to explore arm
- If OOD rate too high → switch to conservative arm

## Configuration

Example arm definition in config:

```yaml
arms:
  - name: "reinvent_qsar_explore"
    type: "reinvent"
    prior: "priors/libinvent.prior"
    scoring:
      - qsar: 1.0
      - novelty: 0.3
    batch_size: 128
    n_steps: 100
```

**Key parameters:**
- `scoring`: Component weights (how much to value each property)
- `batch_size`: Number of molecules generated per step
- `n_steps`: Number of RL steps per episode
- `prior`: Pre-trained generative model checkpoint

## Performance Metrics

Each arm is tracked with:
- Mean score (average quality)
- Top score (best molecule found)
- Pull count (number of times selected)
- Compute time

The report shows which arms produced the best molecules and why.

## Choosing Arms for Your Problem

**Hit discovery:**
- Include `reinvent_explore` and `reinvent_qsar_explore`
- Tolerate lower QSAR confidence
- Emphasize novelty and diversity

**Lead optimisation:**
- Include `mol2mol_high_sim_exploit` and `mol2mol_mmp_exploit`
- Require high QSAR confidence
- Penalize OOD predictions

**Full campaign:**
- Include all arms
- Let the controller adapt automatically
- Monitor which arms perform best for your target
