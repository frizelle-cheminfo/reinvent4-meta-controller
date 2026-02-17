# Purchasable Constraints

## The Reality Check

Generative models can design molecules that don't exist and can't be easily sourced. This creates a disconnect between computational output and practical next steps.

**Purchasable constraint layer** bridges that gap by:
1. Mapping generated molecules to commercially available analogues
2. Re-interpreting novelty in the context of what's actually accessible
3. Showing how rankings change when sourceability is considered

## When Constraints Are Applied

The purchasable filter is applied **late in the funnel** (during reporting, not generation). This preserves the creative freedom of the generative model while providing a reality-grounded view for decision-making.

**Two modes:**

**Soft filter (default):**
- All molecules are reported
- Nearest purchasable analogue is shown
- Rankings can be re-weighted by similarity to purchasable
- Useful for: understanding sourceability without discarding novel chemistry

**Hard filter (optional):**
- Only molecules with close purchasable analogues (e.g., Tanimoto > 0.7) are kept
- Useful for: immediate testability, rapid validation

## How It Works

### Step 1: Build Purchasable Index

Load commercially available compound library (e.g., MolPort, ZINC, Enamine):
- 600K+ compounds
- Morgan fingerprints pre-computed
- Cached for fast lookup

### Step 2: Nearest Neighbour Search

For each generated molecule:
- Compute Morgan fingerprint
- Find top-K nearest neighbours by Tanimoto similarity
- Return product codes and structures

### Step 3: Re-rank (Optional)

Apply sourceability pressure:
```
adjusted_score = base_score * (1 - α) + similarity_to_purchasable * α
```

Where `α` controls how much to penalize hard-to-source molecules.

### Step 4: Report

Output shows:
- Original generated molecule
- Predicted score
- Nearest purchasable analogue
- Tanimoto similarity
- Product code for ordering

## Impact on Rankings

**Example: Top 10 molecules before vs after purchasable constraint**

| Rank | SMILES (generated) | Score | Nearest Purchasable | Sim | New Rank |
|------|-------------------|-------|---------------------|-----|----------|
| 1 | `novel_chemotype_A` | 0.92 | `analogue_A` | 0.65 | 3 ↓ |
| 2 | `known_scaffold_B` | 0.89 | `exact_match_B` | 1.00 | 1 ↑ |
| 3 | `scaffold_hop_C` | 0.87 | `analogue_C` | 0.78 | 2 ↑ |

**Insight:**
- Molecule 1 was top-ranked but hard to source
- Molecule 2 dropped to rank 2 but is immediately purchasable
- After re-ranking, Molecule 2 becomes the practical priority

## Configuration

```yaml
purchasable:
  enabled: true
  csv_path: "data/purchasable_library.csv"
  similarity_threshold: 0.7  # Min similarity to show
  apply_hard_filter: false   # true = discard if no match
  rerank_alpha: 0.3          # Weight for sourceability
  top_k: 5                   # Show top-5 analogues
```

## Interpreting Novelty

**Before purchasable constraints:**
Novelty = Tanimoto distance to training set

**After purchasable constraints:**
Novelty = Tanimoto distance to both training set AND purchasable catalogue

**What this reveals:**
- A "novel" molecule might have close purchasable analogues
- True novelty = distant from both training AND purchasable space
- This helps distinguish:
  - Incrementally novel (easy to test via analogues)
  - Radically novel (requires synthesis)

## Use Cases

### For Computational Chemists

1. **Evaluate model creativity:**
   - How often does it generate truly novel vs. near-purchasable?
   - Is novelty driven by useful diversity or synthetic complexity?

2. **Understand feasibility:**
   - What fraction of top-N have close analogues?
   - Are we designing in accessible chemical space?

### For Medicinal Chemists

1. **Immediate testing:**
   - Order purchasable analogues to validate QSAR predictions
   - Test without waiting for synthesis

2. **Synthesis prioritisation:**
   - If analogue is inactive → don't synthesize exact molecule
   - If analogue is active → worth the synthesis effort

3. **Lead hopping:**
   - Use purchasable analogues as new seeds
   - Transform them with mol2mol arms

## Medchem Handoff Report

The purchasable constraint analysis feeds into the medchem handoff report:

```
Top Candidates for Testing
--------------------------
1. Molecule X (Score: 0.89)
   - Predicted pIC50: 7.8 nM
   - Purchasable analogue: MOLPORT-12345 (Sim: 0.82)
   - Recommendation: Order analogue for validation

2. Molecule Y (Score: 0.86)
   - Predicted pIC50: 7.5 nM
   - Purchasable analogue: None (Sim: 0.58)
   - Recommendation: Requires synthesis, high risk

3. Molecule Z (Score: 0.85)
   - Predicted pIC50: 7.4 nM
   - Purchasable analogue: MOLPORT-67890 (Sim: 0.95)
   - Recommendation: Analogue is near-exact, test first
```

## Data Sources

**Included in repo:**
- `data/toy_purchasable_stub.csv` (25 molecules, demo only)

**For real use:**
- Download MolPort, ZINC15, Enamine catalogues
- Point to local CSV via `--purchasable_csv`
- Index will be cached for subsequent runs

**CSV format:**
```csv
SMILES,Product_Code,Supplier,Price_USD,Stock_Status
CCO,MOLPORT-001,Supplier,15.00,in_stock
c1ccccc1,MOLPORT-002,Supplier,12.50,in_stock
```

## Caveats

1. **Similarity != Activity**
   - Close analogue doesn't guarantee activity transfer
   - Use as validation, not replacement

2. **Catalogue Staleness**
   - Commercial availability changes
   - Recheck before ordering

3. **Similarity Threshold**
   - 0.7 is a guideline, not a rule
   - Adjust based on chemotype

4. **Synthesis Feasibility**
   - Some "novel" molecules are trivial to synthesize
   - Some "purchasable" analogues might be out of stock
   - Human judgment still required
