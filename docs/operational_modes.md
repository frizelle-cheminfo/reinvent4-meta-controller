# Operational Modes

## The Core Insight

**Hit discovery** and **lead optimisation** are fundamentally different problems that require different strategies:

| Aspect | Hit Discovery | Lead Optimisation |
|--------|---------------|-------------------|
| QSAR reliability | Directional guidance | Quantitative ranking |
| OOD tolerance | High (exploring new space) | Low (stay in validated domain) |
| Novelty priority | High (find new chemotypes) | Low (refine known chemistry) |
| Uncertainty acceptable | Yes (accept risk) | No (need confidence) |
| Preferred arms | Explore, diversify | Exploit, transform |

This is not a trivial distinction. Using lead-opt settings for hit discovery will produce boring, incremental molecules. Using hit-discovery settings for lead opt will chase unreliable predictions.

## Mode: Hit Discovery

**When to use:**
- Early in a project with limited SAR data
- Target is novel with sparse training data
- Goal is to find new chemotypes
- Willing to accept false positives

**Philosophy:**
QSAR predictions are directional rather than quantitative. A molecule predicted at pIC50 7.5 might actually be 6.8 or 8.2, but it's more likely active than one predicted at 5.0. The goal is to enrich the search toward interesting chemistry, not to rank precisely.

**Config characteristics:**
```yaml
qsar:
  uncertainty_threshold: 0.4  # High tolerance
  ood_threshold: 0.7          # High tolerance
  confidence_weight: 0.3      # Low penalty

arms:
  # More explore arms, higher novelty weights
  - reinvent_qsar_explore: {novelty: 0.5}
  - reinvent_explore: {diversity: 0.8}

termination:
  patience: 5  # More patience before stopping
```

**What happens:**
- Controller explores broadly
- QSAR is used as a soft preference, not a hard gate
- High-scoring but uncertain molecules are accepted
- Emphasis on chemical diversity
- Longer runs before switching arms

## Mode: Lead Optimisation

**When to use:**
- Validated hit series exists
- Training data is dense around the series
- Goal is to optimize properties (potency, selectivity, ADME)
- Need reliable ranking for synthesis prioritisation

**Philosophy:**
QSAR predictions should be reliable within the validated chemical space. Molecules that fall outside the applicability domain are de-prioritized. Uncertainty is a red flag, not acceptable. Ranking stability matters because medchem teams will synthesize based on predictions.

**Config characteristics:**
```yaml
qsar:
  uncertainty_threshold: 0.2  # Low tolerance
  ood_threshold: 0.5          # Low tolerance
  confidence_weight: 0.7      # High penalty

arms:
  # More exploit arms, lower novelty
  - reinvent_exploit: {similarity: 0.6}
  - mol2mol_high_sim_exploit: {similarity: 0.85}
  - mol2mol_mmp_exploit: {mmp: 0.9}

termination:
  patience: 2  # Less patience
  confidence_degradation: true  # Stop if uncertainty rises
```

**What happens:**
- Controller stays close to validated chemistry
- High uncertainty predictions are penalized
- Emphasis on ranking stability
- Conservative exploit arms dominate
- Stops quickly if confidence degrades

## Operational Timeline Example

A realistic campaign might transition:

```
Episodes 1-10:   Hit Discovery (find chemotypes)
  ├─ Explore arms dominate
  ├─ QSAR used directionally
  └─ Build diverse seed bank

Episodes 11-15:  Validation (test top hits)
  └─ Real activity data validates QSAR

Episodes 16-30:  Lead Optimisation (refine hits)
  ├─ Exploit arms dominate
  ├─ QSAR used quantitatively
  └─ Generate synthesis-ready analogues
```

You can trigger this manually by switching config files, or implement it as a meta-policy that detects when to transition.

## How to Choose

**Start with hit discovery if:**
- Training set < 100 molecules
- Target is novel
- No validated series exists

**Use lead optimisation if:**
- Validated hits exist
- Training set > 500 molecules
- QSAR cross-validation R² > 0.7
- Goal is to make rank-order decisions for synthesis

**Run both separately:**
- Hit discovery to find chemotypes
- Lead opt to optimize each chemotype
- Compare diversity vs refinement outcomes

## Common Mistakes

**Mistake 1: Using lead-opt mode too early**
Result: Controller refuses to explore, iterates around weak analogues of training set

**Mistake 2: Using hit-discovery mode for synthesis prioritisation**
Result: Synthesize molecules that looked promising but were OOD predictions

**Mistake 3: Not checking applicability domain**
Result: Report shows great scores but medchem team finds predictions don't match assay

**Mistake 4: Treating QSAR as ground truth**
Result: Missing true actives that QSAR mis-predicts

## Recommended Practice

1. Run hit discovery mode to generate candidates
2. Review purchasable analogues report
3. Test a diverse sample (20-50 compounds)
4. Retrain QSAR with new data
5. Switch to lead-opt mode for the validated series
6. Iterate
