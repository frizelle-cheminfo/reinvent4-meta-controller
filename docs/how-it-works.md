# How the REINVENT4 Meta-Controller Works

## Overview

The meta-controller is a higher-level orchestration layer that sits above REINVENT4. It does not modify REINVENT4 itself, but rather:

1. **Invokes** REINVENT4 as a subprocess CLI tool
2. **Manages** multiple "arms" (prior + configuration combinations)
3. **Decides** which arm to run next based on metrics from previous episodes
4. **Maintains** a seed bank for seed-based generators
5. **Reports** on the campaign's progress and molecule evolution

## Key Concepts

### Arms

An **arm** is a complete configuration for running one REINVENT4 episode:
- **Prior**: The generative model (e.g., `reinvent.prior`, `mol2mol_mmp.prior`)
- **Generator type**: reinvent, mol2mol, libinvent, or linkinvent
- **Regime**: explore (discover new space) or exploit (optimize known space)
- **Scoring profile**: Weights for potency, novelty, similarity, uncertainty, diversity
- **Diversity settings**: Scaffold bucket similarity, penalty strength
- **Seeds**: For seed-based generators (mol2mol, libinvent, linkinvent)

### Episodes

An **episode** is one short REINVENT4 run (e.g., 500 RL steps). The meta-controller runs many episodes sequentially, switching arms based on performance.

### Regimes

- **Explore**: Prioritize novelty and diversity. Venture into new chemical space.
- **Exploit**: Prioritize potency and local optimization. Stay near known good molecules.

The controller balances these two regimes automatically.

### Seed Bank

For seed-based generators (mol2mol, libinvent, linkinvent), the controller maintains a **seed bank**: a curated set of high-scoring molecules from previous episodes.

- **Exploit arms** select seeds by **top score**
- **Explore arms** select seeds by **diversity** (scaffold clustering)

## Decision Logic

### Step 1: Hard Safety Rules (Override)

The controller first checks if any hard rules are triggered. These are safety overrides based on current metrics:

| Trigger | Threshold | Action |
|---------|-----------|--------|
| **OOD spike** | OOD rate > 0.3 | Switch to exploit/similarity/MMP arms |
| **Uncertainty spike** | Uncertainty > 0.4 | Switch to exploit/seed-based arms |
| **Diversity collapse** | Internal diversity < 0.1 | Switch to explore/scaffold/global arms |
| **Rediscovery spike** | Rediscovery rate > 0.4 | Switch to explore/scaffold arms |
| **Property filter collapse** | Property pass rate < 0.3 | Switch to similarity/MMP/libinvent arms |
| **Stagnation** | No improvement for 3 episodes | Switch to opposite regime |

If any rule triggers, the controller **forces** that arm type, overriding the bandit.

### Step 2: Bandit Selection

If no hard rules trigger, the controller uses **UCB (Upper Confidence Bound)** to select among all available arms.

**UCB formula**:
```
UCB(arm) = mean_quality(arm) + C * sqrt(log(total_episodes) / episodes_run(arm))
```

- **Exploitation term** (`mean_quality`): Favor arms that have worked well
- **Exploration term** (the sqrt part): Favor arms we haven't tried much
- **C parameter** (default 1.5): Controls exploration vs exploitation tradeoff

The arm with highest UCB is selected.

### Step 3: Quality Scoring

After each episode, the controller computes a **quality score** for that arm:

```
quality = + 2.0 * topk_gain        # Reward improvements to best score
          + hit_yield               # Reward high-scoring molecules
          + (internal_diversity + scaffold_diversity) / 2.0  # Reward diversity
          - ood_rate                # Penalize out-of-distribution
          - uncertainty_mean        # Penalize high uncertainty
          - rediscovery_rate        # Penalize rediscovery
```

This quality score updates the arm's statistics for the bandit.

## Metrics Computed

For each episode, the controller computes:

| Metric | Description |
|--------|-------------|
| **Validity %** | Fraction of valid SMILES |
| **Unique %** | Fraction of unique molecules |
| **Internal diversity** | Avg pairwise Tanimoto distance (1 - similarity) |
| **Scaffold diversity** | Fraction of unique Murcko scaffolds |
| **Rediscovery rate** | Fraction matching prior episodes or reference actives |
| **Novelty** | Avg distance to nearest support set molecule |
| **OOD rate** | Fraction with low similarity to support OR high uncertainty |
| **Uncertainty mean** | Average uncertainty (proxy: inverse similarity to support) |
| **Top-k gain** | Improvement over campaign best score |
| **Hit yield** | Fraction of molecules above score threshold |
| **Property pass rate** | Fraction passing MW/logP/TPSA/HBD/HBA constraints |

## Seed Selection Policies

### Exploit Policy

For exploit arms (e.g., `mol2mol_mmp_exploit`, `libinvent_exploit`):

1. Sort seed bank by **score** (descending)
2. Select **top N** seeds
3. Write to `seeds.smi` file

**Goal**: Focus on known high-scoring molecules for local optimization.

### Explore Policy

For explore arms (e.g., `mol2mol_scaffold_explore`, `linkinvent_explore`):

1. Cluster seeds by **Murcko scaffold**
2. Within each cluster, pick the **best-scoring** representative
3. **Round-robin** select from clusters until N seeds collected

**Goal**: Sample diverse regions of chemical space.

## Report Generation

The controller produces three report files:

### 1. manifest.json

Machine-readable JSON with:
- All episode data (arm, metrics, reasons)
- Arm statistics (episodes run, mean quality, successes/failures)
- Campaign summary (best score, total molecules)

### 2. report.md

Human-readable Markdown narrative with:
- Episode timeline table
- Strategy switches (what triggered each switch)
- Molecule evolution (best molecules per episode)
- Arm performance summary

### 3. report.html

Standalone HTML page with:
- Summary metrics (total episodes, best score, molecules generated)
- Interactive plots:
  - Best score progression
  - Diversity over time
  - OOD rate and uncertainty
  - Arm selection frequency
  - Arm performance comparison
- Episode timeline table
- Embedded plots as base64 PNG images (no external dependencies)

## Workflow Diagram

```
                   START
                     |
                     v
            ┌────────────────┐
            │ Initialize     │
            │ - Load config  │
            │ - Load state   │
            │ - Get arms     │
            └────────┬───────┘
                     |
                     v
            ┌────────────────┐
            │ Episode Loop   │<───────────┐
            └────────┬───────┘            │
                     |                     │
                     v                     │
            ┌────────────────┐            │
            │ Check Hard     │            │
            │ Rules          │            │
            └────────┬───────┘            │
                     |                     │
              ┌──────┴──────┐            │
              |             |             │
        Triggered        None            │
              |             |             │
              v             v             │
         Force Arm    Bandit Select      │
              |             |             │
              └──────┬──────┘            │
                     |                     │
                     v                     │
            ┌────────────────┐            │
            │ Select Seeds   │            │
            │ (if needed)    │            │
            └────────┬───────┘            │
                     |                     │
                     v                     │
            ┌────────────────┐            │
            │ Render Config  │            │
            │ (Jinja2 TOML)  │            │
            └────────┬───────┘            │
                     |                     │
                     v                     │
            ┌────────────────┐            │
            │ Invoke         │            │
            │ REINVENT4      │            │
            │ (subprocess)   │            │
            └────────┬───────┘            │
                     |                     │
                     v                     │
            ┌────────────────┐            │
            │ Parse Output   │            │
            │ (CSV)          │            │
            └────────┬───────┘            │
                     |                     │
                     v                     │
            ┌────────────────┐            │
            │ Compute        │            │
            │ Metrics        │            │
            └────────┬───────┘            │
                     |                     │
                     v                     │
            ┌────────────────┐            │
            │ Update State   │            │
            │ - Seed bank    │            │
            │ - Arm stats    │            │
            │ - Best mols    │            │
            └────────┬───────┘            │
                     |                     │
                     v                     │
            ┌────────────────┐            │
            │ Save State     │            │
            └────────┬───────┘            │
                     |                     │
              More episodes? ─────────────┘
                     | No
                     v
            ┌────────────────┐
            │ Generate       │
            │ Report         │
            └────────┬───────┘
                     |
                     v
                   DONE
```

## Why This Approach?

### Advantages

1. **Uncertainty awareness**: Detects when the generator is OOD and switches to safer strategies
2. **Escape local optima**: Diversity collapse triggers global exploration
3. **Reproducible**: All decisions logged and deterministic (with seed)
4. **No REINVENT4 modification**: Clean separation of concerns
5. **Flexible**: Easy to add new arms or customize rules
6. **Transparent**: Every switch explained in report

### Limitations

1. **Overhead**: Short episodes mean more checkpointing/loading
2. **Discrete switches**: Can't blend strategies within an episode
3. **Simple metrics**: Uncertainty is a proxy (no calibrated model)
4. **Local controller**: Not distributed (single machine)

## Extending the Controller

### Add a New Arm

1. Add prior file to `priors/`
2. Edit `src/reinvent_meta/arms.py` and add `ArmConfig(...)`
3. (Optional) Create new template in `src/reinvent_meta/templates/`

### Add a New Hard Rule

Edit `src/reinvent_meta/controller.py`, method `_apply_hard_rules()`:

```python
# Rule: Custom trigger
if latest_metrics.my_metric > self.thresholds.my_threshold:
    reasons.append("my-trigger")
    custom_arms = [arm for arm in self.available_arms if "my-tag" in arm.tags]
    if custom_arms:
        return self.rng.choice(custom_arms), reasons
```

### Add a New Metric

Edit `src/reinvent_meta/metrics.py`, method `compute_episode_metrics()`:

```python
# Compute new metric
my_metric = self._compute_my_metric(smiles_list, scores)

return EpisodeMetrics(
    ...,
    my_metric=my_metric,
)
```

Update `src/reinvent_meta/state.py`, class `EpisodeMetrics` to include the new field.

### Switch to Thompson Sampling

Edit `src/reinvent_meta/controller.py`, method `_bandit_selection()` to use Thompson sampling instead of UCB.

## Further Reading

- [REINVENT4 documentation](https://github.com/MolecularAI/REINVENT4)
- [Multi-armed bandit algorithms](https://en.wikipedia.org/wiki/Multi-armed_bandit)
- [Molecular generation review paper](https://doi.org/10.1002/wcms.1608)
