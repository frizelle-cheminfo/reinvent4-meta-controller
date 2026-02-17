# Medicinal Chemistry Handoff

## Purpose

The medchem handoff report translates computational output into actionable synthesis priorities. It's designed to be forwarded directly to medicinal chemists with all context needed for decision-making.

## What's Included

### 1. Executive Summary
- Run mode (hit discovery vs lead opt)
- Total molecules screened
- Top-N candidates selected
- Selection criteria used
- Key caveats about QSAR reliability

### 2. Candidate Table

For each prioritised molecule:

| Field | Description |
|-------|-------------|
| Rank | Overall ranking by composite score |
| SMILES | Molecular structure |
| Predicted Activity | pIC50 in nM (denormalised) |
| Confidence | QSAR prediction confidence (Low/Med/High) |
| OOD Flag | Whether molecule is out-of-domain |
| Novelty | Tanimoto distance to training set |
| MW, LogP, PSA | Calculated properties |
| Purchasable Analogue | Nearest commercially available compound |
| Product Code | For ordering |
| Rationale | Why this molecule was selected |

### 3. Purchasable Analogues

For validation and rapid testing:
- Top-5 analogues per candidate
- Structural similarity scores
- Product codes and suppliers
- Estimated pricing (if available)

### 4. Recommendations

Categorised by risk/effort:

**Immediate Test (Low Risk)**
- Purchasable analogues with >0.8 similarity
- Action: Order and test

**Validation Required (Medium Risk)**
- Novel chemotypes with 0.6-0.8 analogue similarity
- Action: Test analogue first, synthesize if active

**Synthesis Required (High Risk)**
- <0.6 similarity to purchasable
- Action: Requires custom synthesis, consider carefully

### 5. Caveats

**For Hit Discovery Mode:**
> "QSAR predictions are directional guidance, not quantitative. Uncertainty is higher for novel chemotypes. Recommend testing diverse subset (10-20 compounds) to validate predictions before committing to full series."

**For Lead Optimisation Mode:**
> "Predictions are within applicability domain. Confidence is high for ranking. However, absolute values may still have ±0.5 log unit error. Consider SAR trends rather than individual point estimates."

## File Outputs

```
out/demo_report/medchem_handoff/
├── handoff.md          # Human-readable summary
├── candidates.csv      # Full data table
├── summary.json        # Metadata and stats
└── structures/         # 2D structure images
    ├── candidate_001.png
    ├── candidate_002.png
    └── ...
```

## Usage

### Generate Handoff Report

```bash
python -m r4mc.report \
  --run_dir out/demo_run \
  --out_dir out/demo_report \
  --purchasable_csv data/toy_purchasable_stub.csv \
  --medchem_handoff
```

### Review Report

1. Open `handoff.md` in Markdown viewer
2. Review executive summary and caveats
3. Check candidate table for top priorities
4. Export `candidates.csv` for sorting/filtering
5. Share with medicinal chemistry team

### Making Decisions

**Step 1: Validate with Analogues**
- Order top 5-10 purchasable analogues
- Test in primary assay
- Confirm QSAR predictions

**Step 2: Synthesis Prioritisation**
- If analogues confirm predictions → synthesize exact molecules
- If analogues fail → revisit QSAR or try different chemotypes

**Step 3: Iterate**
- Feed assay results back into training set
- Retrain QSAR
- Run another generation cycle

## Example Handoff

```markdown
# Medicinal Chemistry Handoff Report

**Run**: demo_run
**Mode**: Hit Discovery
**Generated**: 2026-02-17
**Total Molecules**: 2,847 unique
**Candidates Presented**: 20

## Executive Summary

This run used hit discovery mode to explore novel chemotypes with predicted
BRD4 activity. QSAR guidance was directional (not quantitative). High
uncertainty and OOD rates are expected and acceptable at this stage.

**Key Findings:**
- 8 candidates with predicted pIC50 > 7.0 nM
- 12 candidates have purchasable analogues (similarity > 0.7)
- 3 novel scaffolds not present in training set

**Recommended Actions:**
1. Order purchasable analogues for immediate testing (Products: TOY-012, TOY-020, TOY-021)
2. Test diverse sample to validate QSAR before committing to synthesis
3. Focus on molecules with medium confidence scores (more reliable)

## Top Candidates

### 1. Candidate A (Score: 0.89, High Confidence)

**Structure:**
![](structures/candidate_001.png)

**Predicted Activity:** 15.2 nM (pIC50: 7.82)

**Properties:**
- MW: 425.5
- LogP: 3.2
- PSA: 68.4
- HBD: 2, HBA: 5

**Purchasable Analogue:**
- Product: TOY-020
- Similarity: 0.82
- Price: $350 USD
- **Recommendation:** Order analogue for validation

**Selection Rationale:**
Novel quinazoline scaffold with good predicted activity. Within applicability
domain. Analogue available for rapid testing. If analogue confirms activity,
worth synthesizing exact structure for optimization.

---

### 2. Candidate B (Score: 0.86, Medium Confidence)

... (continues for top 20)
```

## Integration with Workflow

The handoff report is the **output artefact** that drives experimental work:

```
Computational Campaign → Medchem Handoff → Experimental Validation → Data Feedback
                              ↓
                    Decision making happens here
```

It serves as:
1. **Communication layer** between comp chem and med chem
2. **Decision record** for why molecules were prioritised
3. **Risk assessment** for synthesis investment
4. **Validation plan** for QSAR reliability

## Best Practices

**Do:**
- Include caveats about QSAR limitations
- Provide multiple tiers of recommendations (immediate/medium/synthesis)
- Show purchasable analogues even for novel molecules
- Explain selection rationale clearly

**Don't:**
- Oversell predictions ("these will definitely work")
- Hide uncertainty or OOD flags
- Omit molecules just because they're hard to source
- Present rankings without context

**Remember:**
The goal is to enable informed decisions, not to make decisions for the medchem team. Provide data, context, and recommendations, then let domain experts decide.
