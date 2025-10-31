# Genomic Embeddings Examples

This directory contains tutorials for extracting and analyzing genomic sequence embeddings using OmniGenBench.

## Overview

**Genomic Embeddings** are dense vector representations of DNA/RNA sequences that capture biological semantics learned by pre-trained foundation models. These embeddings enable:

- **Sequence similarity search**: Find functionally similar sequences
- **Clustering and classification**: Group sequences by function
- **Zero-shot prediction**: Assess properties without training
- **Transfer learning**: Use as features for downstream tasks
- **Visualization**: Explore sequence space structure

## Key Feature: Universal Embedding Support

**ALL OmniModel types support embedding extraction** through the `EmbeddingMixin`:
- `OmniModelForEmbedding` - Dedicated embedding extraction
- `OmniModelForSequenceClassification` - Classification + embeddings
- `OmniModelForSequenceRegression` - Regression + embeddings
- `OmniModelForTokenClassification` - Token classification + embeddings
- And all other OmniModel variants!

This means you can:
- Use your **fine-tuned classification models** for embedding extraction
- Extract embeddings from **task-specific models** without loading separate models
- Get **task-adapted representations** that are optimized for your domain

## Files in this Directory

- **`RNA_Embedding_Tutorial.ipynb`**: Comprehensive tutorial covering all embedding operations

## Quick Start

### 1. Single Sequence Embedding

```python
from omnigenbench import OmniModelForEmbedding, OmniTokenizer
import torch

# Load tokenizer
tokenizer = OmniTokenizer.from_pretrained(
    "yangheng/OmniGenome-186M",
    trust_remote_code=True
)

# Load model
model = OmniModelForEmbedding(
    "yangheng/OmniGenome-186M",
    tokenizer=tokenizer,
    trust_remote_code=True
)

# Extract embedding
sequence = "ATCGATCGATCGATCGATCG"
embedding = model.encode(sequence, agg="mean")  # Shape: (hidden_dim,)

print(f"Embedding shape: {embedding.shape}")
print(f"Embedding (first 10 dims): {embedding[:10]}")
```

### 2. Batch Embedding Extraction

```python
sequences = [
    "ATCGATCGATCG",
    "GCGCGCGCGCGC",
    "TATATATATATAT"
]

# Efficient batch processing
embeddings = model.batch_encode(
    sequences,
    batch_size=32,
    agg="mean"  # Options: mean, max, cls, last
)

print(f"Batch embeddings shape: {embeddings.shape}")  # (num_sequences, hidden_dim)
```

### 3. Sequence Similarity

```python
# Compare two sequences
seq1 = "ATCGATCGATCG"
seq2 = "ATCGAACGATCG"  # One mismatch

emb1 = model.encode(seq1)
emb2 = model.encode(seq2)

# Cosine similarity
similarity = model.compute_similarity(emb1, emb2)
print(f"Similarity: {similarity:.4f}")  # Close to 1.0 for similar sequences
```

## Embedding Aggregation Strategies

Different aggregation methods capture different aspects of sequences:

| Strategy | Description | Use Case |
|----------|-------------|----------|
| **`head`** | Use the first token embedding (CLS-like) | Global sequence summary |
| **`mean`** | Average all token embeddings | General-purpose, balanced representation |
| **`tail`** | Use the last valid token embedding | Sequential/directional information |

**Recommendation**: Start with `mean` for most tasks.

## Advanced Embedding Configuration

After mastering basic embedding extraction, optimize your workflow with these advanced configurations:

### Aggregation Strategy Selection

Choose the aggregation method that best suits your task:

```python
from omnigenbench import OmniModelForEmbedding, OmniTokenizer

tokenizer = OmniTokenizer.from_pretrained("yangheng/OmniGenome-186M", trust_remote_code=True)
model = OmniModelForEmbedding("yangheng/OmniGenome-186M", tokenizer=tokenizer)

sequence = "ATCGATCGATCG"

# Mean pooling (RECOMMENDED - most robust)
emb_mean = model.encode(sequence, agg="mean")

# Head token (CLS-like)
emb_head = model.encode(sequence, agg="head")

# Tail token (last valid token)
emb_tail = model.encode(sequence, agg="tail")
```

**When to Use Each Strategy**:

| Task | Best Strategy | Reason |
|------|---------------|--------|
| Sequence similarity search | `mean` | Balanced, stable representation |
| Motif detection | `head` or `mean` | Global summary or averaged features |
| Classification (BERT-style) | `head` | CLS-like first token |
| Sequence generation | `tail` | Captures sequential context |
| Clustering | `mean` | Stable and generalizable |
| Zero-shot prediction | `mean` | Most generalizable |

**Comparison Example**:
```python
import numpy as np

strategies = ["mean", "head", "tail"]
sequence = "ATCGATCGATCG"

embeddings = {}
for strategy in strategies:
    embeddings[strategy] = model.encode(sequence, agg=strategy)
    
# Compare embedding magnitudes
for strategy, emb in embeddings.items():
    print(f"{strategy:6s}: norm={np.linalg.norm(emb):.4f}, shape={emb.shape}")

# Example output:
# mean  : norm=12.3, shape=(768,)
# head  : norm=11.2, shape=(768,)
# tail  : norm=13.4, shape=(768,)
```

### Batch Processing Configuration

Optimize batch size for your hardware:

```python
# Small GPU (4-8GB) - conservative batch size
embeddings = model.batch_encode(
    sequences,
    batch_size=16,
    agg="mean"
)

# Medium GPU (8-16GB) - standard batch size
embeddings = model.batch_encode(
    sequences,
    batch_size=32,
    agg="mean"
)

# Large GPU (16GB+) - aggressive batch size
embeddings = model.batch_encode(
    sequences,
    batch_size=64,
    agg="mean"
)

# CPU-only - smaller batches recommended
embeddings = model.batch_encode(
    sequences,
    batch_size=8,
    agg="mean"
)
```

**Performance Tuning**:
```python
import time

sequences = ["ATCGATCG" * 50] * 1000  # 1000 sequences

# Benchmark different batch sizes
for batch_size in [8, 16, 32, 64]:
    start = time.time()
    embeddings = model.batch_encode(sequences, batch_size=batch_size)
    elapsed = time.time() - start
    print(f"Batch size {batch_size:2d}: {elapsed:.2f}s ({len(sequences)/elapsed:.1f} seq/s)")

# Output (example on RTX 3090):
# Batch size  8: 12.34s (81.0 seq/s)
# Batch size 16: 7.89s (126.7 seq/s)
# Batch size 32: 5.67s (176.4 seq/s)
# Batch size 64: 4.56s (219.3 seq/s)
```

### Device Management

Control which hardware is used for embedding extraction:

```python
import torch

# Automatic device selection (default)
model = OmniModelForEmbedding("yangheng/OmniGenome-186M", tokenizer=tokenizer)
# Uses CUDA if available, otherwise CPU

# Force specific GPU or CPU using .to()
model.to("cuda:0")  # or model.to(torch.device("cpu"))

# Multi-GPU: manually distribute work
import torch

if torch.cuda.device_count() > 1:
    # Process half on GPU 0, half on GPU 1
    mid = len(sequences) // 2
    
    model_gpu0 = OmniModelForEmbedding("MODEL", tokenizer=tokenizer)
    model_gpu0.to("cuda:0")
    model_gpu1 = OmniModelForEmbedding("MODEL", tokenizer=tokenizer)
    model_gpu1.to("cuda:1")
    
    emb_0 = model_gpu0.batch_encode(sequences[:mid])
    emb_1 = model_gpu1.batch_encode(sequences[mid:])
    
    embeddings = np.vstack([emb_0, emb_1])
```

### Extracting Layer-Specific Embeddings

Get embeddings from specific transformer layers:

```python
# This requires using the model directly (advanced)
from omnigenbench import ModelHub

model = ModelHub.load("yangheng/OmniGenome-186M")
tokenizer = OmniTokenizer.from_pretrained("yangheng/OmniGenome-186M", trust_remote_code=True)

# Tokenize
inputs = tokenizer(
    "ATCGATCGATCG",
    return_tensors="pt",
    padding=True,
    truncation=True
)

# Get all hidden states
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)

# Extract embeddings from different layers
hidden_states = outputs.hidden_states  # Tuple of (num_layers, batch, seq_len, hidden_dim)

# Last layer (default)
last_layer = hidden_states[-1]  # Shape: (1, seq_len, hidden_dim)
emb_last_layer = last_layer.mean(dim=1)  # Mean pooling

# Middle layer (more general features)
mid_layer = hidden_states[len(hidden_states)//2]
emb_mid_layer = mid_layer.mean(dim=1)

# First layer (low-level features)
first_layer = hidden_states[1]  # [0] is embedding layer
emb_first_layer = first_layer.mean(dim=1)

print(f"First layer: {emb_first_layer.shape}")
print(f"Middle layer: {emb_mid_layer.shape}")
print(f"Last layer: {emb_last_layer.shape}")
```

**Layer Selection Guidelines**:
- **Last layers (10-12)**: Task-specific, fine-grained features
- **Middle layers (5-7)**: Balanced, transferable features (often best for transfer learning)
- **Early layers (1-3)**: Low-level sequence patterns (k-mers, motifs)

### Using Fine-Tuned Models for Embeddings

Extract embeddings from your own fine-tuned models:

```python
# Load your fine-tuned classification model
from omnigenbench import OmniModelForSequenceClassification

# Fine-tuned model for TFB prediction
tfb_model = OmniModelForSequenceClassification(
    "yangheng/ogb_tfb_finetuned",
    num_labels=919
)

# Extract embeddings using the classification model!
# (All OmniModel types support embedding extraction via EmbeddingMixin)
sequence = "ATCGATCGATCG"
embedding = tfb_model.encode(sequence, agg="mean")

print(f"Embedding shape: {embedding.shape}")
print(f"Embedding is task-adapted for TFB prediction!")

# These embeddings are specialized for your domain
# Better for: similarity search in your specific context
#             downstream tasks related to your fine-tuning task
```

**Benefits of Task-Adapted Embeddings**:
- Better separation for task-relevant features
- Higher similarity scores for functionally similar sequences
- Improved performance on related downstream tasks

### Memory-Efficient Embedding for Large Datasets

Process millions of sequences without running out of memory:

```python
import numpy as np
from tqdm import tqdm

def embed_large_dataset(model, sequences, batch_size=32, save_every=10000):
    """
    Embed large dataset with checkpointing.
    """
    all_embeddings = []
    
    for i in tqdm(range(0, len(sequences), batch_size)):
        batch = sequences[i:i+batch_size]
        batch_emb = model.batch_encode(batch, batch_size=batch_size)
        all_embeddings.append(batch_emb)
        
        # Save checkpoint every N sequences
        if (i + batch_size) % save_every == 0:
            partial = np.vstack(all_embeddings)
            np.save(f"embeddings_checkpoint_{i}.npy", partial)
            all_embeddings = []  # Free memory
    
    # Final save
    if all_embeddings:
        partial = np.vstack(all_embeddings)
        np.save(f"embeddings_final.npy", partial)
    
    return "Embeddings saved to disk"

# Use for very large datasets
sequences = ["ATCG" * 100] * 1_000_000  # 1M sequences
embed_large_dataset(model, sequences, batch_size=64, save_every=50000)
```

## Use Cases

### 1. Sequence Similarity Search

Find sequences similar to a query:

```python
# Database of sequences
database = [
    "ATCGATCGATCG",
    "GCGCGCGCGCGC",
    "TATATATATATAT",
    # ... thousands more
]

# Get embeddings for database
db_embeddings = model.batch_encode(database, batch_size=64)

# Query sequence
query = "ATCGAACGATCG"
query_emb = model.encode(query)

# Find most similar
from sklearn.metrics.pairwise import cosine_similarity
similarities = cosine_similarity(query_emb.reshape(1, -1), db_embeddings)[0]
top_k = similarities.argsort()[-5:][::-1]

print("Most similar sequences:")
for idx in top_k:
    print(f"  {database[idx]} (similarity: {similarities[idx]:.4f})")
```

### 2. Unsupervised Clustering

Group sequences by function without labels:

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Extract embeddings
embeddings = model.batch_encode(sequences)

# Cluster
kmeans = KMeans(n_clusters=5)
labels = kmeans.fit_predict(embeddings)

# Visualize with PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
coords = pca.fit_transform(embeddings)

plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap='viridis')
plt.title("Sequence Clustering")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()
```

### 3. Zero-Shot Classification

Classify sequences using prototype embeddings:

```python
# Define prototypes for each class
prototypes = {
    "promoter": model.encode("TATAAA..."),  # Known promoter sequence
    "coding": model.encode("ATGGCC..."),    # Known CDS
    "utr": model.encode("GCGCGC...")       # Known UTR
}

# Classify new sequence
new_seq = "TATAAA..."  # Unknown sequence
new_emb = model.encode(new_seq)

# Find closest prototype
similarities = {
    name: model.compute_similarity(new_emb, proto)
    for name, proto in prototypes.items()
}

predicted_class = max(similarities, key=similarities.get)
print(f"Predicted class: {predicted_class}")
print(f"Confidence: {similarities[predicted_class]:.4f}")
```

### 4. Embedding Visualization

Visualize embedding space:

```python
from sklearn.manifold import TSNE
import seaborn as sns

# Extract embeddings
embeddings = model.batch_encode(sequences)

# Dimensionality reduction
tsne = TSNE(n_components=2, perplexity=30)
coords = tsne.fit_transform(embeddings)

# Plot
plt.figure(figsize=(10, 8))
sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=labels)
plt.title("t-SNE Visualization of Genomic Embeddings")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.legend(title="Class")
plt.show()
```

### 5. Feature Extraction for ML Models

Use embeddings as features for traditional ML:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Extract embeddings
train_embs = model.batch_encode(train_sequences)
test_embs = model.batch_encode(test_sequences)

# Train classifier
clf = LogisticRegression()
clf.fit(train_embs, train_labels)

# Predict
predictions = clf.predict(test_embs)
```

## Advanced Features

### 1. Token-Level Embeddings

Get embeddings for each nucleotide:

```python
# Extract token embeddings (no aggregation)
token_embeddings = model.encode_tokens(
    sequence="ATCGATCG",
    max_length=512
)

print(f"Token embeddings shape: {token_embeddings.shape}")
# Shape: (seq_len, hidden_dim)

# Analyze position-specific representations
for i, emb in enumerate(token_embeddings):
    print(f"Position {i}: {emb[:5]}...")  # First 5 dimensions
```

### 2. Layer-Specific Embeddings

Extract embeddings from specific layers:

```python
# Get embeddings from different layers
outputs = model.model(
    **tokenizer(sequence, return_tensors="pt"),
    output_hidden_states=True
)

# Compare early vs. late layers
early_layer_emb = outputs.hidden_states[3].mean(dim=1)  # Layer 3
late_layer_emb = outputs.hidden_states[-1].mean(dim=1)   # Last layer

# Early layers: syntax/structure
# Late layers: semantics/function
```

### 3. Multi-Model Ensemble

Combine embeddings from multiple models:

```python
from omnigenbench import OmniModelForEmbedding

# Load multiple models
models = [
    OmniModelForEmbedding("yangheng/OmniGenome-52M"),
    OmniModelForEmbedding("yangheng/OmniGenome-186M"),
    OmniModelForEmbedding("yangheng/PlantRNA-FM")
]

# Extract and concatenate embeddings
embeddings = []
for model in models:
    emb = model.encode(sequence)
    embeddings.append(emb)

ensemble_emb = torch.cat(embeddings, dim=0)
print(f"Ensemble embedding shape: {ensemble_emb.shape}")
```

### 4. Using Task-Specific Models for Embeddings

**Key Insight**: You can extract embeddings from your fine-tuned models!

```python
# Example: Use your fine-tuned TFB model for embeddings
from omnigenbench import OmniModelForSequenceClassification

# Load your fine-tuned model
tfb_model = OmniModelForSequenceClassification(
    model="./my_tfb_finetuned_model",
    num_labels=919
)

# Extract task-adapted embeddings!
# These embeddings are optimized for TFB-related features
tfb_adapted_emb = tfb_model.encode(sequence, agg="mean")

# Use for similarity search in TFB context
# Or as features for related tasks
```

## Performance Tips

### 1. Batch Processing

Always use batch encoding for multiple sequences:

```python
# Slow (sequential)
embeddings = [model.encode(seq) for seq in sequences]  # ❌

# Fast (batched)
embeddings = model.batch_encode(sequences, batch_size=32)  # ✅
```

### 2. GPU Acceleration

```python
import torch

# Move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Embeddings automatically computed on GPU
embeddings = model.batch_encode(sequences)
```

### 3. Memory Management

For large datasets:

```python
import numpy as np

# Process in chunks
chunk_size = 1000
all_embeddings = []

for i in range(0, len(sequences), chunk_size):
    chunk = sequences[i:i+chunk_size]
    chunk_embs = model.batch_encode(chunk)
    all_embeddings.append(chunk_embs.cpu().numpy())  # Move to CPU

# Combine
final_embeddings = np.vstack(all_embeddings)
```

### 4. Caching Embeddings

Save embeddings to avoid recomputation:

```python
import pickle

# Save
embeddings = model.batch_encode(sequences)
with open("embeddings.pkl", "wb") as f:
    pickle.dump(embeddings, f)

# Load
with open("embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)
```

## Common Issues

### Issue 1: Embeddings All Similar

**Symptom**: All sequences have high similarity (> 0.95)

**Causes**:
- Using wrong aggregation method
- Model not properly loaded
- Sequences too short

**Solutions**:
- Try different aggregation: `agg="max"` instead of `"mean"`
- Verify model loaded correctly
- Use longer sequences (> 50 bp)

### Issue 2: Out of Memory

**Symptom**: CUDA OOM during batch encoding

**Solutions**:
```python
# Reduce batch size
embeddings = model.batch_encode(sequences, batch_size=8)

# Or move to CPU
embeddings = model.batch_encode(sequences, return_on_cpu=True)
```

### Issue 3: Slow Embedding Extraction

**Symptom**: Very slow for large datasets

**Solutions**:
1. Use GPU: `model.to("cuda")`
2. Increase batch size: `batch_size=64`
3. Use mixed precision: `torch.cuda.amp.autocast()`
4. Cache results: Save embeddings to disk

## Evaluation

### Intrinsic Evaluation

Test embedding quality directly:

1. **Sequence Similarity**: Similar sequences → similar embeddings
2. **Functional Grouping**: Same function → cluster together
3. **Mutation Sensitivity**: Single mutations → small embedding changes

### Extrinsic Evaluation

Test embeddings on downstream tasks:

```python
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

# Use embeddings as features
scores = cross_val_score(
    SVC(),
    embeddings,
    labels,
    cv=5,
    scoring='f1_macro'
)

print(f"Cross-validation F1: {scores.mean():.4f} ± {scores.std():.4f}")
```

## Related Tutorials

- **Attention Analysis**: `examples/attention_score_extraction/` - Visualize model attention
- **Variant Effect Prediction**: `examples/variant_effect_prediction/` - Use embeddings for VEP
- **RNA Design**: `examples/rna_sequence_design/` - Leverage embeddings for design

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
