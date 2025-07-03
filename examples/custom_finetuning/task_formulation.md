
# Token-level vs Sequence-level Models: A Guide for Task Formulation

## Token-level Models: When Each Position Matters

### When to Use Token-level Models:
1. When you need predictions for EACH POSITION in your sequence
2. When your labels/annotations correspond to individual nucleotides/amino acids
3. When you want to analyze position-specific properties

### Examples:
1. **RNA Secondary Structure Prediction**
   ```
   Sequence: GGAAAGUUGGACUGU...
   Labels:   .....((((((...)))))) 
   ```
   - Each nucleotide has its own structural state (dot or bracket)
   - Need position-by-position prediction

2. **mRNA Degradation Rate Prediction**
   ```
   Sequence: GGAAAGUUGGACUGU...
   Labels:   [0.1, 0.2, 0.15, ...] 
   ```
   - Each position has its own degradation rate value
   - Position-specific measurements

## Sequence-level Models: When the Whole Sequence Matters

### When to Use Sequence-level Models:
1. When you have ONE label for the ENTIRE sequence
2. When you're interested in global properties
3. When the property isn't position-specific

### Examples:
1. **Protein Function Classification**
   ```
   Sequence: MAEGEITTFTALTEKFNLPPGNYKKPKLLY...
   Label: "Kinase"
   ```
   - One classification for the whole protein
   - Not position-specific

2. **RNA Stability Prediction**
   ```
   Sequence: GGAAAGUUGGACUGU...
   Label: 45.2 (stability score)
   ```
   - Single stability value for entire sequence
   - Global property

## Quick Decision Guide:

Ask yourself:
- Do I need a prediction for each nucleotide/amino acid? → Use Token-level
- Do I need one prediction for the whole sequence? → Use Sequence-level
- Are my experimental measurements for individual positions? → Use Token-level
- Is my measurement a single value for the entire sequence? → Use Sequence-level

This distinction is crucial for choosing the right model architecture and preparing your data correctly for your biological research question.