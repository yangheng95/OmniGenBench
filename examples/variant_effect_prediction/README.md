# Variant Effect Prediction Examples

This directory contains comprehensive tutorials for predicting the functional impact of genetic variants using OmniGenBench.

## Overview

**Variant Effect Prediction (VEP)** is a **zero-shot** approach that uses embedding comparison to assess how mutations affect biological function. This technique:

- **Requires no training**: Uses pre-trained models directly
- **Generalizes broadly**: Works across different variant types
- **Interprets naturally**: Embedding distance correlates with functional impact
- **Scales efficiently**: Can process thousands of variants quickly

## What is Variant Effect Prediction?

Given a reference sequence and an alternative (mutated) sequence, VEP:
1. Extracts embeddings from both sequences
2. Computes distance/similarity between embeddings
3. Interprets distance as functional impact magnitude

**Key Insight**: Functionally important mutations cause larger embedding changes.

## Files in this Directory

### Quick Start
- **`quickstart_vep.ipynb`**: End-to-end VEP workflow (**START HERE**)

### Detailed Step-by-Step Tutorials
- **`01_vep_data_preparation.ipynb`**: Variant data formatting and preprocessing
- **`02_vep_model_setup.ipynb`**: Model selection and embedding extraction
- **`03_embedding_and_scoring.ipynb`**: Distance metrics and scoring strategies
- **`04_visualization_and_export.ipynb`**: Results visualization and export (if available)

## Quick Start

### Option 1: Jupyter Notebook

```bash
cd examples/variant_effect_prediction
jupyter notebook quickstart_vep.ipynb
```

### Option 2: Python API

```python
from omnigenbench import OmniModelForSequenceClassification, OmniTokenizer
import torch

# Load tokenizer
tokenizer = OmniTokenizer.from_pretrained(
    "yangheng/PlantRNA-FM",
    trust_remote_code=True
)

# Load model (any task-specific model works!)
model = OmniModelForSequenceClassification(
    "yangheng/PlantRNA-FM",
    tokenizer=tokenizer,
    num_labels=2,  # Not used for VEP, but required for initialization
    trust_remote_code=True
)
model = model.eval()

# Define variant
ref_seq = "ATCGATCGATCGATCGATCG"
alt_seq = "ATCGAACGATCGATCGATCG"  # A>AA insertion at position 6

# Extract embeddings
ref_emb = model.encode(ref_seq, agg="mean")
alt_emb = model.encode(alt_seq, agg="mean")

# Compute functional impact score
similarity = model.compute_similarity(ref_emb, alt_emb)
impact_score = 1 - similarity  # Higher score = larger impact

print(f"Variant impact score: {impact_score:.4f}")
```

### Option 3: Batch Processing

For efficient processing of multiple variants:

```python
from omnigenbench import OmniModelForSequenceClassification, OmniTokenizer
import torch
import json

# Load model and tokenizer
tokenizer = OmniTokenizer.from_pretrained(
    "yangheng/PlantRNA-FM",
    trust_remote_code=True
)

model = OmniModelForSequenceClassification(
    "yangheng/PlantRNA-FM",
    tokenizer=tokenizer,
    num_labels=2,
    trust_remote_code=True
)
model = model.eval()

# Load variants from file
with open("variants.json", "r") as f:
    variants = json.load(f)

# Batch process
results = []
for variant in variants:
    ref_emb = model.encode(variant["ref_seq"], agg="mean")
    alt_emb = model.encode(variant["alt_seq"], agg="mean")
    
    similarity = model.compute_similarity(ref_emb, alt_emb)
    impact_score = 1 - similarity
    
    results.append({
        "variant_id": variant["variant_id"],
        "impact_score": float(impact_score),
        "prediction": "pathogenic" if impact_score > 0.5 else "benign"
    })

# Save results
with open("vep_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"Processed {len(results)} variants")
```

## Data Format

### Input Format

Variant data should include reference and alternative sequences:

```json
[
  {
    "variant_id": "rs123456",
    "ref_seq": "ATCGATCGATCGATCGATCG",
    "alt_seq": "ATCGAACGATCGATCGATCG",
    "position": 6,
    "mutation_type": "insertion",
    "label": 1,
    "description": "Known pathogenic variant"
  },
  {
    "variant_id": "rs789012",
    "ref_seq": "GCGCGCGCGCGCGCGCGCGC",
    "alt_seq": "GCGCGAGCGCGCGCGCGCGC",
    "position": 6,
    "mutation_type": "substitution",
    "label": 0,
    "description": "Benign variant"
  }
]
```

**Required fields**:
- `ref_seq`: Reference sequence
- `alt_seq`: Alternative (mutated) sequence

**Optional fields**:
- `position`: Mutation position (for analysis)
- `mutation_type`: Type of mutation
- `label`: Ground truth (1 = pathogenic, 0 = benign)
- `variant_id`, `description`: Metadata

### Creating Sequences from VCF

If you have VCF files, extract sequences with flanking regions:

```python
def get_variant_sequences(chrom, pos, ref, alt, genome, flank=100):
    """Extract reference and alternative sequences from genome."""
    start = max(0, pos - flank)
    end = pos + len(ref) + flank
    
    # Get reference sequence
    ref_seq = genome[chrom][start:end]
    
    # Create alternative sequence
    alt_seq = (
        genome[chrom][start:pos] +
        alt +
        genome[chrom][pos+len(ref):end]
    )
    
    return ref_seq, alt_seq

# Example usage
ref_seq, alt_seq = get_variant_sequences(
    chrom="chr1",
    pos=12345,
    ref="A",
    alt="G",
    genome=genome_dict,
    flank=200  # 200bp on each side
)
```

## Distance Metrics

Different metrics capture different aspects of functional impact:

| Metric | Description | Use Case |
|--------|-------------|----------|
| **Cosine Distance** | 1 - cosine_similarity | General-purpose, scale-invariant |
| **Euclidean Distance** | L2 norm | Magnitude-sensitive |
| **Manhattan Distance** | L1 norm | Robust to outliers |
| **Correlation Distance** | 1 - Pearson correlation | Pattern similarity |

**Recommendation**: Start with **cosine distance** for most applications.

```python
from sklearn.metrics.pairwise import (
    cosine_similarity,
    euclidean_distances,
    manhattan_distances
)
import numpy as np

# Compute different distances
cosine_dist = 1 - cosine_similarity(ref_emb.reshape(1, -1), alt_emb.reshape(1, -1))[0, 0]
euclidean_dist = euclidean_distances(ref_emb.reshape(1, -1), alt_emb.reshape(1, -1))[0, 0]
manhattan_dist = manhattan_distances(ref_emb.reshape(1, -1), alt_emb.reshape(1, -1))[0, 0]

print(f"Cosine distance: {cosine_dist:.4f}")
print(f"Euclidean distance: {euclidean_dist:.4f}")
print(f"Manhattan distance: {manhattan_dist:.4f}")
```

## Aggregation Strategies

### Position-Specific vs. Global Embeddings

**Global Aggregation** (entire sequence):
```python
ref_emb = model.encode(ref_seq, agg="mean")
alt_emb = model.encode(alt_seq, agg="mean")
```

**Position-Specific** (focus on mutation site):
```python
# Extract token-level embeddings
ref_tokens = model.encode_tokens(ref_seq)
alt_tokens = model.encode_tokens(alt_seq)

# Focus on mutation position
mutation_pos = 6
ref_mut_emb = ref_tokens[mutation_pos]
alt_mut_emb = alt_tokens[mutation_pos]

# Compute position-specific distance
pos_impact = model.compute_similarity(ref_mut_emb, alt_mut_emb)
```

### Multi-Layer Analysis

Different layers capture different information:

```python
# Extract embeddings from multiple layers
outputs = model.model(
    **tokenizer(ref_seq, return_tensors="pt"),
    output_hidden_states=True
)

# Compare across layers
for i, hidden_state in enumerate(outputs.hidden_states[::3]):  # Every 3rd layer
    layer_emb = hidden_state.mean(dim=1)
    print(f"Layer {i*3}: embedding shape {layer_emb.shape}")
```

## Evaluation Metrics

If ground truth labels are available:

| Metric | Description | When to Use |
|--------|-------------|-------------|
| **AUC-ROC** | Area under ROC curve | Binary classification (pathogenic vs. benign) |
| **Spearman Correlation** | Rank correlation | When you have continuous impact scores |
| **Precision@k** | Precision in top-k predictions | Prioritizing high-confidence predictions |

```python
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Compute impact scores for all variants
impact_scores = []
for ref, alt in zip(ref_sequences, alt_sequences):
    ref_emb = model.encode(ref, agg="mean")
    alt_emb = model.encode(alt, agg="mean")
    score = 1 - model.compute_similarity(ref_emb, alt_emb)
    impact_scores.append(score)

# Evaluate if labels available
if labels is not None:
    auc = roc_auc_score(labels, impact_scores)
    print(f"AUC-ROC: {auc:.4f}")
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(labels, impact_scores)
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()
```

## Advanced Techniques

### 1. Ensemble Scoring

Combine scores from multiple models:

```python
from omnigenbench import OmniModelForSequenceClassification

models = [
    OmniModelForSequenceClassification(model="yangheng/OmniGenome-52M", num_labels=2),
    OmniModelForSequenceClassification(model="yangheng/OmniGenome-186M", num_labels=2),
    OmniModelForSequenceClassification(model="yangheng/PlantRNA-FM", num_labels=2)
]

# Compute ensemble score
scores = []
for model in models:
    ref_emb = model.encode(ref_seq)
    alt_emb = model.encode(alt_seq)
    score = 1 - model.compute_similarity(ref_emb, alt_emb)
    scores.append(score)

ensemble_score = np.mean(scores)
print(f"Ensemble impact score: {ensemble_score:.4f}")
```

### 2. Context-Aware Scoring

Include flanking sequence context:

```python
# Short context
ref_short = ref_seq  # 50bp window
alt_short = alt_seq

# Long context
ref_long = extended_ref_seq  # 500bp window
alt_long = extended_alt_seq

# Compare impact at different scales
short_score = compute_impact(model, ref_short, alt_short)
long_score = compute_impact(model, ref_long, alt_long)

print(f"Short context impact: {short_score:.4f}")
print(f"Long context impact: {long_score:.4f}")
```

### 3. Multi-Objective Scoring

Combine embedding distance with attention patterns:

```python
# Embedding-based score
ref_emb = model.encode(ref_seq)
alt_emb = model.encode(alt_seq)
emb_score = 1 - model.compute_similarity(ref_emb, alt_emb)

# Attention-based score
ref_attn = model.extract_attention_scores(ref_seq)
alt_attn = model.extract_attention_scores(alt_seq)
attn_score = compute_attention_change(ref_attn, alt_attn)

# Combined score
combined_score = 0.7 * emb_score + 0.3 * attn_score
```

## Performance Tips

### 1. Batch Processing

Process multiple variants efficiently:

```python
# Extract all embeddings in batches
ref_embeddings = model.batch_encode(ref_sequences, batch_size=32)
alt_embeddings = model.batch_encode(alt_sequences, batch_size=32)

# Compute all distances
from sklearn.metrics.pairwise import cosine_similarity
similarities = cosine_similarity(ref_embeddings, alt_embeddings)
impact_scores = 1 - np.diag(similarities)  # Diagonal = pairwise scores
```

### 2. GPU Acceleration

```python
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Embeddings computed on GPU automatically
embeddings = model.batch_encode(sequences)
```

### 3. Caching Embeddings

Save embeddings for repeated analysis:

```python
import pickle

# Compute once
ref_embeddings = model.batch_encode(ref_sequences)
with open("ref_embeddings.pkl", "wb") as f:
    pickle.dump(ref_embeddings, f)

# Reuse
with open("ref_embeddings.pkl", "rb") as f:
    ref_embeddings = pickle.load(f)
```

## Common Issues

### Issue 1: All Impact Scores Too Similar

**Symptom**: All scores between 0.95-1.00

**Solutions**:
- Use longer flanking sequences (200-500bp)
- Try position-specific scoring instead of global
- Use different aggregation: `agg="max"` instead of `"mean"`
- Try a different distance metric

### Issue 2: Poor Correlation with Known Labels

**Symptom**: AUC < 0.6

**Solutions**:
- Check sequence orientation (forward vs. reverse complement)
- Verify mutation positions are correct
- Use model trained on similar organism/tissue
- Consider ensemble of multiple models
- Add biological features (conservation scores, etc.)

### Issue 3: Memory Issues with Large Datasets

**Symptom**: OOM errors

**Solutions**:
```python
# Process in chunks
chunk_size = 100
for i in range(0, len(variants), chunk_size):
    chunk_refs = ref_sequences[i:i+chunk_size]
    chunk_alts = alt_sequences[i:i+chunk_size]
    
    ref_embs = model.batch_encode(chunk_refs, return_on_cpu=True)
    alt_embs = model.batch_encode(chunk_alts, return_on_cpu=True)
    
    # Process chunk...
```

## Expected Results

### ClinVar Variants (Human)
- AUC-ROC: 0.70-0.85 (pathogenic vs. benign)
- Best with larger models (186M+ parameters)

### Plant SNPs
- AUC-ROC: 0.65-0.80
- Better with PlantRNA-FM

### Splice Site Variants
- AUC-ROC: 0.75-0.90
- Position-specific scoring works best

## Biological Interpretation

### What Does Embedding Distance Mean?

**Large Distance** (>0.5):
- Significant functional change
- Likely affects protein function, RNA structure, or regulation
- Prioritize for experimental validation

**Medium Distance** (0.2-0.5):
- Moderate functional impact
- May be context-dependent
- Consider additional evidence

**Small Distance** (<0.2):
- Likely neutral or synonymous
- Conserved function
- Lower priority

### Visualizing Impact

```python
import seaborn as sns

# Plot impact score distribution
sns.histplot(impact_scores, bins=50)
plt.xlabel('Impact Score')
plt.ylabel('Frequency')
plt.title('Distribution of Variant Impact Scores')
plt.show()

# Plot by mutation type
sns.boxplot(x=mutation_types, y=impact_scores)
plt.xlabel('Mutation Type')
plt.ylabel('Impact Score')
plt.show()
```

## Next Steps

After completing this tutorial:
1. **Attention Analysis**: See which sequence regions drive the model's assessment
2. **RNA Structure Prediction**: Understand how mutations affect secondary structure
3. **Transfer to Other Tasks**: Use VEP scores as features for predictive models

## Citation

```bibtex
@article{yang2025omnigenome,
  title={OmniGenome: Unified Genomic Foundation Models for Multi-Task Learning},
  author={Yang, Heng and others},
  journal={bioRxiv},
  year={2025}
}
```

## Support

- **GitHub Issues**: https://github.com/yangheng95/OmniGenBench/issues
- **Documentation**: [Getting Started Guide](../../docs/GETTING_STARTED.md)
