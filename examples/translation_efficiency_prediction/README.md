# Translation Efficiency Prediction Examples

This directory contains comprehensive tutorials for predicting translation efficiency (TE) of mRNA sequences using OmniGenBench.

## Overview

**Translation Efficiency (TE) Prediction** is a **sequence classification** task where we predict whether an mRNA sequence has high or low translation efficiency. This is crucial for:
- Synthetic biology and genetic engineering
- Understanding post-transcriptional regulation
- Optimizing codon usage for protein expression
- Plant biotechnology applications

## What is Translation Efficiency?

Translation efficiency measures how effectively an mRNA molecule is translated into protein. It depends on:
- **5' UTR structure**: Ribosome binding and scanning
- **Codon usage bias**: tRNA availability and codon optimization
- **mRNA secondary structure**: Accessibility to ribosomes
- **Start codon context**: Kozak sequence in eukaryotes
- **Poly(A) tail**: mRNA stability

## Dataset

We use plant-specific mRNA translation efficiency datasets:
- **Rice (Oryza sativa)** mRNA sequences
- **Binary labels**: High TE (1) vs. Low TE (0)
- **Sequence features**: Full-length mRNA including 5' UTR, CDS, and 3' UTR
- **Biological context**: Plant-specific codon usage patterns

## Files in this Directory

### Quick Start
- **`quickstart_te.ipynb`**: End-to-end example in a single notebook (**START HERE**)

### Detailed Step-by-Step Tutorials
- **`01_data_preparation.ipynb`**: Dataset loading, preprocessing, and exploratory analysis
- **`02_model_initialization.ipynb`**: Model selection and architecture setup
- **`03_model_training.ipynb`**: Fine-tuning strategies and training monitoring
- **`04_model_inference.ipynb`**: Prediction and deployment (if available)

## Quick Start

### Option 1: Jupyter Notebook (Recommended)

```bash
cd examples/translation_efficiency_prediction
jupyter notebook quickstart_te.ipynb
```

### Option 2: Python API

```python
from omnigenbench import (
    OmniModelForSequenceClassification,
    OmniDatasetForSequenceClassification,
  OmniTokenizer,
  Trainer
)

# 1. Initialize tokenizer
tokenizer = OmniTokenizer.from_pretrained(
    "yangheng/PlantRNA-FM",
    trust_remote_code=True
)

# 2. Initialize model with PlantRNA-FM for plant-specific features
model = OmniModelForSequenceClassification(
    "yangheng/PlantRNA-FM",
    tokenizer=tokenizer,
    num_labels=2,  # Binary: High TE vs Low TE
    trust_remote_code=True
)

# 3. Prepare your dataset
# Format: [{"sequence": "AUGC...", "label": 1}, ...]
dataset = OmniDatasetForSequenceClassification(
    train_file="your_train_data.json",
    tokenizer=tokenizer,
    max_length=512
)

# 4. Train
trainer = Trainer(
  model=model,
  train_dataset=dataset,
  epochs=10,
  batch_size=16
)
metrics = trainer.train()

# 5. Predict on new sequences
test_sequence = "AUGCGAUCUCGAGCUACGUCGAUG"
inputs = tokenizer(test_sequence, return_tensors="pt", max_length=512, truncation=True)
outputs = model(**inputs)
prediction = outputs.logits.argmax(dim=-1)
print(f"Predicted TE: {'High' if prediction == 1 else 'Low'}")
```

### Option 3: Command-Line Interface

```bash
# Training
ogb autotrain \
    --model yangheng/PlantRNA-FM \
    --task-type sequence_classification \
    --train-file ./data/rice_te_train.json \
    --test-file ./data/rice_te_test.json \
    --num-labels 2 \
    --output-dir ./te_model \
    --batch-size 16 \
    --epochs 10 \
    --learning-rate 2e-5 \
    --max-length 512

# Inference
ogb autoinfer \
    --model ./te_model \
    --sequence "AUGCGAUCUCGAGCUACGUCGAUG..." \
    --output-file predictions.json
```

## Key Concepts

### 1. Why PlantRNA-FM?

We use **PlantRNA-FM** (Plant RNA Foundation Model) because:
- Pre-trained on large-scale plant RNA data
- Captures plant-specific codon usage bias
- Understands plant RNA secondary structures
- Optimized for plant genetic contexts

### 2. Sequence Classification Architecture

The model combines:
- **Backbone**: Pre-trained PlantRNA-FM (learns sequence representations)
- **Classification Head**: Linear layer for binary prediction
- **Output**: Logits for High TE and Low TE classes

### 3. Training Strategy

**Transfer Learning Workflow**:
1. **Frozen backbone** (first few epochs): Only train classification head
2. **Fine-tune all layers** (remaining epochs): Adapt model to TE task
3. **Learning rate scheduling**: Warmup + cosine decay

## Data Format

### Input Format

Your training data should be in JSON format:

```json
[
  {
    "sequence": "AUGCGAUCUCGAGCUACGUCGAUGCUAGCUCGAUGGCAUCCGAU",
    "label": 1,
    "gene_id": "Os01g0100100",
    "description": "High TE gene"
  },
  {
    "sequence": "AUGUGCUGCAUCGAUCGAUCGAUCGAUCGAUCGAUCGAUCGAUC",
    "label": 0,
    "gene_id": "Os01g0100200",
    "description": "Low TE gene"
  }
]
```

**Required fields**:
- `sequence`: RNA sequence (A, U, G, C or A, T, G, C)
- `label`: 1 (High TE) or 0 (Low TE)

**Optional fields** (preserved in output):
- `gene_id`, `description`, or any custom metadata

### Output Format

Predictions are saved as JSON:

```json
{
  "model": "yangheng/PlantRNA-FM",
  "task": "translation_efficiency_prediction",
  "results": [
    {
      "sequence": "AUGCGAUC...",
      "predicted_label": 1,
      "probabilities": [0.12, 0.88],
      "metadata": {
        "gene_id": "Os01g0100100"
      }
    }
  ]
}
```

## Performance Tips

### 1. Sequence Length Selection

**Recommendations**:
- **Full-length mRNA**: `max_length=2048` (includes 5' UTR, CDS, 3' UTR)
- **CDS only**: `max_length=1024` (faster, focuses on codon usage)
- **5' UTR + start of CDS**: `max_length=512` (captures initiation signals)

**Trade-offs**:
- Longer sequences = more context but slower training
- Shorter sequences = faster but may miss important features

### 2. Batch Size Optimization

| GPU VRAM | Max Length | Recommended Batch Size |
|----------|------------|------------------------|
| 8 GB     | 512        | 16                     |
| 16 GB    | 512        | 32                     |
| 16 GB    | 1024       | 16                     |
| 24 GB    | 2048       | 8                      |

### 3. Hyperparameter Tuning

**Good starting point**:
```python
learning_rate = 2e-5
num_epochs = 10
warmup_ratio = 0.1
weight_decay = 0.01
```

**For small datasets (< 1000 samples)**:
- Use higher dropout: `dropout=0.3`
- More regularization: `weight_decay=0.1`
- Early stopping: `patience=3`

### 4. Handling Class Imbalance

If you have unbalanced High/Low TE samples:

```python
import torch

# Option 1: Weighted loss with BCEWithLogitsLoss
# Estimate class weights from your dataset distribution
pos_weight = torch.tensor([2.0])  # >1.0 increases weight on positive class
loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

trainer = Trainer(
  model=model,
  train_dataset=dataset,
  epochs=10,
  batch_size=16,
  loss_fn=loss_fn
)

# Option 2: Balanced sampling at dataset construction (if supported)
dataset = OmniDatasetForSequenceClassification(
  data_file="train.json",
  tokenizer=tokenizer,
  balanced_sampling=True  # Oversample minority class
)
```

## Evaluation Metrics

For binary classification, we track:

| Metric | Description | When to Use |
|--------|-------------|-------------|
| **Accuracy** | Overall correct predictions | Balanced datasets |
| **Precision** | True positives / (True positives + False positives) | When false positives are costly |
| **Recall** | True positives / (True positives + False negatives) | When false negatives are costly |
| **F1 Score** | Harmonic mean of precision and recall | Imbalanced datasets |
| **AUC-ROC** | Area under ROC curve | Model discrimination ability |

**For TE prediction**, prioritize:
- **High Recall**: Don't miss high-TE genes for protein expression
- **F1 Score**: Balance precision and recall

## Expected Results

### With Default Settings (10 epochs, batch_size=16)

**Training Time**:
- CPU: ~2-3 hours (not recommended)
- Single GPU (RTX 3090): ~20-30 minutes
- Multi-GPU (2x A100): ~10 minutes

**Performance**:
- Accuracy: 85-90%
- F1 Score: 0.83-0.88
- AUC-ROC: 0.90-0.95

## Biological Interpretation

### What Makes High TE mRNAs?

After training, you can extract insights:

1. **Attention Visualization**: See which sequence regions matter most
```python
from omnigenbench import OmniModelForSequenceClassification

model = OmniModelForSequenceClassification(
    model="./te_model",
    num_labels=2
)

# Extract attention scores
attention_scores = model.extract_attention_scores(
    sequence="AUGCGAUC...",
    layer_indices=[0, 5, 11]  # Multiple layers
)
```

2. **Feature Importance**: Identify key motifs
3. **Codon Usage Analysis**: Compare predicted High vs Low TE sequences

## Common Issues and Solutions

### Issue 1: Model Predicts All Same Class

**Symptom**: Accuracy = 50% (or majority class frequency)

**Solutions**:
1. Check class balance: `dataset.label_distribution()`
2. Reduce learning rate: `--learning-rate 1e-5`
3. Add class weights or use focal loss
4. Ensure dataset is shuffled

### Issue 2: Overfitting

**Symptom**: Training accuracy >> validation accuracy

**Solutions**:
1. Add dropout: `--dropout 0.2`
2. Reduce model size: Use smaller backbone
3. Data augmentation: Random masking, reverse complement
4. Early stopping: `--patience 5`

### Issue 3: NaN Loss

**Symptom**: Loss becomes NaN during training

**Solutions**:
1. Lower learning rate: `--learning-rate 1e-6`
2. Enable gradient clipping: `--max-grad-norm 1.0`
3. Check for invalid sequences (non-AUGC characters)
4. Use mixed precision carefully: `--fp16 False`

## Advanced Topics

### 1. Cross-Species Transfer Learning

Train on one species, test on another:

```python
# Train on rice
model = OmniModelForSequenceClassification(
    model="yangheng/PlantRNA-FM",
    num_labels=2
)
# ... train on rice data ...

# Test on Arabidopsis
arabidopsis_dataset = OmniDatasetForSequenceClassification(
    data_file="arabidopsis_te_test.json",
    tokenizer=tokenizer
)
results = trainer.evaluate(arabidopsis_dataset)
```

### 2. Multi-Task Learning (Note)

Multi-task training APIs are not yet available in the public API. For now, train separate task-specific heads/models (e.g., one model for TE classification and another for mRNA stability regression), or extract embeddings and build lightweight downstream classifiers as needed.

### 3. Zero-Shot Prediction

Use embedding similarity without training:

```python
# Get embeddings for known High TE sequences
high_te_embeddings = model.encode(high_te_sequences)

# Compare new sequence to prototypes
new_embedding = model.encode(new_sequence)
similarity = cosine_similarity(new_embedding, high_te_embeddings.mean(dim=0))
```

## Advanced Configuration

### Training Parameter Tuning

For detailed guidance on optimizing training parameters, see the [TFB Prediction Advanced Training Configuration](../tfb_prediction/README.md#advanced-training-configuration), which covers:

- **Trainer selection**: `native`, `accelerate` (recommended), or `hf_trainer`
- **Learning rate guidelines**: Species-specific and model-size-specific recommendations
- **Batch size optimization**: Balance GPU memory and convergence speed
- **Multi-seed evaluation**: Ensure robust performance estimates

**Quick TE-Specific Recommendations**:

```bash
# For plant-specific models (PlantRNA-FM)
ogb autotrain \
    --dataset plant_te_data \
    --model yangheng/PlantRNA-FM \
    --learning-rate 2e-5 \
    --batch-size 16 \
    --num-epochs 15 \
    --trainer accelerate

# For general genomic models (OmniGenome)
ogb autotrain \
    --dataset plant_te_data \
    --model yangheng/OmniGenome-186M \
    --learning-rate 1e-5 \
    --batch-size 8 \
    --num-epochs 20 \
    --trainer accelerate
```

**TE-Specific Considerations**:
- **Longer sequences**: TE datasets often have longer mRNA sequences (500-2000 bp)
  - Use smaller batch sizes (8-16) to manage memory
  - Consider gradient accumulation for effective larger batches
- **Class imbalance**: If your dataset has unbalanced High/Low TE
  - Use weighted loss or class balancing
  - Focus on F1 score rather than accuracy
- **Cross-species generalization**: For transfer learning across species
  - Start with lower learning rates (1e-5)
  - Use longer training (20-30 epochs)
  - Validate on target species early and often

### Sequence Length Configuration

TE prediction requires careful sequence length management:

```python
from omnigenbench import OmniDatasetForSequenceClassification, OmniTokenizer

tokenizer = OmniTokenizer.from_pretrained("yangheng/PlantRNA-FM", trust_remote_code=True)

# Short sequences (5' UTR only, ~200 bp)
dataset_short = OmniDatasetForSequenceClassification(
    train_file="te_5utr.json",
    tokenizer=tokenizer,
    max_length=256,  # Accommodate 5' UTR
    batch_size=32    # Larger batch possible
)

# Full-length mRNA (5' UTR + CDS + 3' UTR, ~1500 bp)
dataset_full = OmniDatasetForSequenceClassification(
    train_file="te_fulllength.json",
    tokenizer=tokenizer,
    max_length=2048,  # Accommodate full mRNA
    batch_size=8      # Smaller batch due to memory
)
```

**Length Selection Guidelines**:
- **256 tokens**: 5' UTR analysis, fast training
- **512 tokens**: 5' UTR + partial CDS, balanced
- **1024 tokens**: Full CDS + UTRs, comprehensive
- **2048 tokens**: Full-length transcripts, memory-intensive

### Inference Optimization

For batch prediction on new sequences:

```bash
# Quick inference on small dataset
ogb autoinfer \
    --model yangheng/ogb_te_finetuned \
    --input-file new_sequences.json \
    --batch-size 32 \
    --output-file te_predictions.json

# Large-scale inference (genome-wide)
ogb autoinfer \
    --model yangheng/ogb_te_finetuned \
    --input-file genome_transcripts.csv \
    --batch-size 64 \
    --device cuda:0 \
    --output-file genome_te_predictions.json
```

See [AutoInfer Advanced Configuration](../autoinfer_examples/README.md#advanced-configuration) for more details on batch size tuning and device selection.

### Model Ensemble for Robust Predictions

Combine multiple models for better accuracy:

```python
from omnigenbench import ModelHub
import numpy as np

# Load multiple fine-tuned models
models = [
    ModelHub.load("yangheng/ogb_te_finetuned_seed0"),
    ModelHub.load("yangheng/ogb_te_finetuned_seed1"),
    ModelHub.load("yangheng/ogb_te_finetuned_seed2")
]

# Ensemble prediction
def ensemble_predict(sequence, models):
    predictions = []
    for model in models:
        output = model.inference(sequence)
        predictions.append(output['probabilities'])
    
    # Average probabilities
    avg_prob = np.mean(predictions, axis=0)
    return {"prediction": np.argmax(avg_prob), "confidence": np.max(avg_prob)}

# Use ensemble
result = ensemble_predict("AUGCGAUCUGC...", models)
print(f"Ensemble prediction: {'High TE' if result['prediction'] == 1 else 'Low TE'}")
print(f"Confidence: {result['confidence']:.4f}")
```

## Next Steps

After completing this tutorial, explore:

1. **RNA Design**: Use TE predictions to guide synthetic gene design
2. **Variant Effect Prediction**: Assess how mutations affect TE
3. **Codon Optimization**: Optimize coding sequences for high expression
4. **Attention Analysis**: Understand model's decision-making process

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
- **Examples**: Other tutorials in `examples/` directory
