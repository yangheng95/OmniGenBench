# Attention Score Extraction Examples

Extract and visualize attention patterns from genomic foundation models to understand what the model "sees" in sequences.

## Key Feature

**ALL OmniModel types support attention extraction** through the unified `EmbeddingMixin`:
- OmniModelForEmbedding - Dedicated embedding & attention extraction
- OmniModelForSequenceClassification - Classification + attention
- OmniModelForSequenceRegression - Regression + attention
- OmniModelForTokenClassification - Token-level + attention
- All other OmniModel variants

You don't need a separate "embedding model" to extract attention - any task-specific model works!

## Quick Start

### Python Script

```bash
cd examples/attention_score_extraction
python attention_extraction_example.py
```

This runs 5 demonstrations:
1. Single sequence attention extraction
2. Batch attention processing
3. Attention visualization (heatmaps)
4. Attention pattern comparison
5. Layer/head-specific extraction

### Jupyter Notebook

```bash
jupyter notebook Attention_Analysis_Tutorial.ipynb
```

Interactive tutorial with visualizations covering:
- Attention extraction from any model type
- Statistical analysis of attention patterns
- Heatmap visualization
- Batch processing techniques
- Combined attention + embedding workflows

## API Examples

### Extract Attention from Any Model Type

```python
from omnigenbench import OmniModelForSequenceClassification, OmniTokenizer
import torch

# Load tokenizer
tokenizer = OmniTokenizer.from_pretrained(
    "yangheng/OmniGenome-186M",
    trust_remote_code=True
)

# Use a classification model (or any other OmniModel type)
model = OmniModelForSequenceClassification(
    "yangheng/OmniGenome-186M",
    tokenizer=tokenizer,
    num_labels=2,
    trust_remote_code=True
)

model = model.to(torch.device("cuda")).eval()

# Extract attention scores
attention_result = model.extract_attention_scores(
    sequence="ATCGATCGATCGTAGCTAGCTAGCT",
    max_length=128,
    layer_indices=None,  # All layers (or specify [0, 5, 11])
    head_indices=None,   # All heads (or specify [0, 1, 2])
    return_on_cpu=True
)

print(f"Attention shape: {attention_result['attentions'].shape}")
# Format: (layers, heads, seq_len, seq_len)
print(f"Tokens: {attention_result['tokens'][:10]}")
```

### Compute Attention Statistics

```python
# Analyze attention patterns
stats = model.get_attention_statistics(
    attention_result['attentions'],
    attention_result['attention_mask'],
    layer_aggregation="mean",  # Options: mean, max, sum, first, last
    head_aggregation="mean"    # Options: mean, max, sum
)

print(f"Attention entropy: {stats['attention_entropy'].mean():.4f}")
print(f"Self-attention: {stats['self_attention_scores'].mean():.4f}")
print(f"Concentration: {stats['attention_concentration'].max():.4f}")
```

### Visualize Attention Heatmap

```python
import matplotlib.pyplot as plt

# Create heatmap for specific layer and head
fig = model.visualize_attention_pattern(
    attention_result=attention_result,
    layer_idx=0,   # First layer
    head_idx=0,    # First attention head
    save_path="attention_heatmap.png",
    figsize=(12, 10)
)

plt.show()
```

### Batch Processing

```python
sequences = [
    "ATCGATCGATCGTAGCTAGCTAGCT",
    "GGCCTTAACCGGTTAACCGGTTAA",
    "TTTTAAAACCCCGGGGTTTTAAAA"
]

# Extract attention from multiple sequences efficiently
batch_results = model.batch_extract_attention_scores(
    sequences=sequences,
    batch_size=8,
    max_length=128,
    layer_indices=[0, -1],  # First and last layer only
    head_indices=[0, 1, 2], # First 3 heads only
    return_on_cpu=True
)

for i, result in enumerate(batch_results):
    print(f"Sequence {i+1} attention shape: {result['attentions'].shape}")
```

## Available Methods

All OmniModel types inherit these methods from `EmbeddingMixin`:

### Attention Methods
- `extract_attention_scores()` - Extract attention from single sequence
- `batch_extract_attention_scores()` - Batch attention extraction
- `get_attention_statistics()` - Compute attention metrics
- `visualize_attention_pattern()` - Create attention heatmaps

### Embedding Methods
- `encode()` - Generate sequence embeddings
- `batch_encode()` - Batch embedding generation
- `encode_tokens()` - Token-level embeddings
- `compute_similarity()` - Embedding similarity computation

## Use Cases

### 1. Model Interpretability
Understand which sequence positions the model focuses on for predictions:
```python
# Extract attention for a classified sequence
attention = classifier_model.extract_attention_scores(sequence)
stats = classifier_model.get_attention_statistics(attention['attentions'], attention['attention_mask'])

# High entropy = distributed attention across sequence
# Low entropy = focused attention on specific positions
print(f"Attention entropy: {stats['attention_entropy'].mean()}")
```

### 2. Motif Discovery
Identify important sequence motifs by analyzing attention patterns:
```python
# Find positions with high attention concentration
max_positions = stats['max_attention_per_position']
print(f"Most attended positions: {max_positions[:10]}")
```

### 3. Model Comparison
Compare attention patterns across different model architectures:
```python
# Extract attention from two models
attention_model_a = model_a.extract_attention_scores(sequence)
attention_model_b = model_b.extract_attention_scores(sequence)

# Compare attention distributions
stats_a = model_a.get_attention_statistics(attention_model_a['attentions'], attention_model_a['attention_mask'])
stats_b = model_b.get_attention_statistics(attention_model_b['attentions'], attention_model_b['attention_mask'])

print(f"Model A entropy: {stats_a['attention_entropy'].mean():.4f}")
print(f"Model B entropy: {stats_b['attention_entropy'].mean():.4f}")
```

### 4. Layer-wise Analysis
Examine how attention patterns evolve through model layers:
```python
# Extract all layers
attention_all = model.extract_attention_scores(
    sequence=sequence,
    layer_indices=None  # All layers
)

# Analyze each layer
num_layers = attention_all['attentions'].shape[0]
for layer_idx in range(num_layers):
    layer_attention = attention_all['attentions'][layer_idx:layer_idx+1]
    stats = model.get_attention_statistics(layer_attention, attention_all['attention_mask'])
    print(f"Layer {layer_idx}: entropy={stats['attention_entropy'].mean():.4f}")
```

## Files in This Directory

- **`Attention_Analysis_Tutorial.ipynb`** - Interactive tutorial with visualizations (⭐ **START HERE**)
- **`attention_extraction_example.py`** - Comprehensive Python script with 5 examples

## Key Benefits

### Unified Interface
- Same API works across all model types
- No need for separate embedding models
- Extract attention from fine-tuned task-specific models

### Rich Analysis Tools
- Statistical measures (entropy, concentration, self-attention)
- Visualization (heatmaps with token labels)
- Batch processing for efficiency
- Layer/head filtering for focused analysis

### Production-Ready
- GPU/CPU memory management
- Efficient batch processing
- Configurable output formats
- Compatible with all OmniGenBench models

## Requirements

```bash
pip install omnigenbench torch transformers matplotlib seaborn numpy
```

## Tips

1. **Memory Management**: Use `return_on_cpu=True` for large models to avoid GPU OOM
2. **Selective Extraction**: Specify `layer_indices` and `head_indices` to reduce computation
3. **Batch Size**: Adjust `batch_size` based on sequence length and available GPU memory
4. **Visualization**: Heatmaps work best with sequences < 100 tokens for readability

## Examples Output

After running the examples, you'll get:
- Console output with attention statistics
- `attention_heatmap.png` - Visualization of attention patterns
- Understanding of attention patterns across your sequences

## Related Examples

- [Genomic Embeddings](../genomic_embeddings/) - Learn about embedding extraction
- [Variant Effect Prediction](../variant_effect_prediction/) - Use attention for variant analysis
- [TFB Prediction](../tfb_prediction/) - Attention in multi-label classification

---

**Next Steps**: Run the tutorial notebook to interactively explore attention patterns in your genomic sequences!
