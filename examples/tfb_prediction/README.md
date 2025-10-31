# Transcription Factor Binding (TFB) Prediction Examples

This directory contains comprehensive tutorials for predicting transcription factor binding sites in genomic sequences using OmniGenBench.

## Overview

Transcription Factor Binding (TFB) prediction is a **multi-label classification** task where we predict which transcription factors bind to a given DNA sequence. This is crucial for understanding gene regulation, enhancer activity, and chromatin accessibility.

## Dataset

We use the **DeepSEA** dataset, which contains:
- **440,000 training examples** (DNA sequences)
- **919 labels** (transcription factors)
- **Fixed sequence length**: 1000 bp
- **Binary labels**: 1 (binding), 0 (non-binding)

### Dataset Format

The dataset uses JSON format with the following structure:

```json
[
  {
    "sequence": "ATCGATCGATCG...",  // 1000 bp DNA sequence
    "labels": [0, 1, 0, 1, 0, ..., 0]  // 919 binary labels (one per TF)
  },
  {
    "sequence": "GCTAGCTAGCTA...",
    "labels": [1, 0, 0, 0, 1, ..., 1]
  }
]
```

**Key Points**:
- Each `sequence` must be exactly 1000 nucleotides (A, T, C, G)
- Each `labels` array must contain exactly 919 binary values
- Labels are ordered consistently across all samples (same TF at same position)
- Missing or ambiguous nucleotides (N, R, Y, etc.) should be avoided

### Custom TFB Datasets

To create your own TFB prediction dataset:

```python
import json

# Example: Create a TFB dataset with 3 transcription factors
tfb_data = []
for i in range(100):  # 100 training examples
    sample = {
        "sequence": generate_sequence(length=1000),  # Your sequence generation
        "labels": [
            int(has_tf1_binding),  # TF1: e.g., NF-kB
            int(has_tf2_binding),  # TF2: e.g., p53
            int(has_tf3_binding)   # TF3: e.g., STAT3
        ]
    }
    tfb_data.append(sample)

# Save to JSON
with open("custom_tfb_train.json", "w") as f:
    json.dump(tfb_data, f, indent=2)
```

**Best Practices**:
1. **Balance**: Try to have at least 50-100 positive examples per TF
2. **Quality**: Use validated ChIP-seq or similar experimental data
3. **Negative samples**: Include true negatives (experimentally verified non-binding)
4. **Sequence context**: Include sufficient flanking regions around binding sites

## Files in this Directory

### Quick Start
- **`quickstart_tfb.ipynb`**: End-to-end example in a single notebook (**START HERE**)

### Detailed Step-by-Step Tutorials
- **`01_data_preparation.ipynb`**: Dataset loading and preprocessing
- **`02_model_initialization.ipynb`**: Model setup and architecture explanation
- **`03_model_training.ipynb`**: Training configuration and fine-tuning
- **`04_model_inference.ipynb`**: Making predictions on new sequences
- **`05_advanced_dataset_creation.ipynb`**: Custom dataset creation patterns

### Visual Guide
- **`4-step workflow.png`**: Visual overview of the complete workflow

## Quick Start

### Option 1: Jupyter Notebook (Recommended for Learning)

```bash
cd examples/tfb_prediction
jupyter notebook quickstart_tfb.ipynb
```

Follow the cells sequentially to:
1. Install dependencies
2. Load the DeepSEA dataset
3. Initialize the model
4. Fine-tune on your data
5. Make predictions

### Option 2: Command-Line Interface

For production workflows, use the `ogb` CLI:

```bash
# Training
ogb autotrain \
    --model yangheng/PlantRNA-FM \
    --dataset deepsea_tfb_prediction \
    --output-dir ./tfb_model \
    --batch-size 16 \
    --num-epochs 10 \
    --learning-rate 2e-5

# Inference
ogb autoinfer \
    --model ./tfb_model \
    --sequence "ATCGATCG..." \
    --output-file predictions.json
```

## Key Concepts

### 1. Multi-Label Classification

Unlike binary classification (one label per sequence), TFB prediction assigns **multiple labels** to each sequence because:
- Multiple transcription factors can bind to the same region
- Binding sites can overlap
- Chromatin accessibility affects multiple factors

### 2. Model Architecture

We use `OmniModelForMultiLabelSequenceClassification` which:
- Loads a pre-trained genomic foundation model (e.g., PlantRNA-FM, OmniGenome)
- Adds a multi-label classification head (919 outputs)
- Uses **sigmoid activation** (not softmax) to handle multiple positive labels
- Loss function: **Binary Cross-Entropy with Logits**

### 3. Evaluation Metrics

For multi-label tasks, we use:
- **Macro-averaged F1**: Average F1 across all labels (handles class imbalance)
- **Micro-averaged F1**: Global F1 across all instances
- **Sample-wise metrics**: Precision, recall, F1 per sequence
- **Label-wise metrics**: Performance per transcription factor

## Python API Examples

### Basic Training

```python
from omnigenbench import (
    OmniModelForMultiLabelSequenceClassification,
    OmniDatasetForMultiLabelClassification,
    OmniTokenizer,
    Trainer
)

# 1. Load tokenizer and model
tokenizer = OmniTokenizer.from_pretrained("yangheng/PlantRNA-FM", trust_remote_code=True)
model = OmniModelForMultiLabelSequenceClassification(
    model="yangheng/PlantRNA-FM",
    num_labels=919,
    trust_remote_code=True
)

# 2. Load dataset from local JSON files
train_dataset = OmniDatasetForMultiLabelClassification(
    data_file="train.json",
    tokenizer=tokenizer,
    max_length=1000
)

valid_dataset = OmniDatasetForMultiLabelClassification(
    data_file="valid.json",
    tokenizer=tokenizer,
    max_length=1000
)

# 3. Train
trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    epochs=10,
    batch_size=16
)
metrics = trainer.train()
```

### Inference on New Sequences

```python
# Load fine-tuned model
model = OmniModelForMultiLabelSequenceClassification(
    model="./tfb_model",
    num_labels=919,
    trust_remote_code=True
)
tokenizer = OmniTokenizer.from_pretrained("./tfb_model", trust_remote_code=True)

# Predict
sequence = "ATCGATCGATCG" * 84  # Create 1000bp sequence (12 * 84 = 1008bp)
inputs = tokenizer(sequence, return_tensors="pt", max_length=1000, truncation=True)
outputs = model(**inputs)

# Get binding probabilities
import torch
probs = torch.sigmoid(outputs.logits)
predicted_labels = (probs > 0.5).int()  # Threshold at 0.5

print(f"Number of predicted binding sites: {predicted_labels.sum().item()}")
```

## Performance Tips

### 1. Memory Management

TFB prediction is memory-intensive due to:
- Long sequences (1000 bp)
- 919 output labels
- Large batch size

**Solutions**:
- Reduce batch size: `--batch-size 8` (default: 16)
- Enable gradient accumulation: `--gradient-accumulation-steps 4`
- Use mixed precision: `--fp16` (automatic with `ogb` CLI)
- Set `--max-examples 10000` for quick testing

### 2. Training Speed

**GPU Recommendations**:
- Minimum: 16GB VRAM (GTX 1080 Ti, RTX 3080)
- Recommended: 24GB VRAM (RTX 3090, A5000)
- Optimal: 40GB+ VRAM (A100, H100)

**Distributed Training**:
```bash
accelerate config  # Configure multi-GPU setup
ogb autotrain --trainer accelerate ...
```

### 3. Dealing with Class Imbalance

Many transcription factors have very few binding sites. Strategies:
- Use **focal loss** (set `--use-focal-loss`)
- Adjust class weights based on frequency
- Sample hard negatives during training

### 4. Hyperparameter Tuning

Start with these proven configurations:

**Fast Experimentation** (10 min on single GPU):
```bash
--batch-size 32 --num-epochs 3
```

**Production Quality** (2-4 hours):
```bash
--batch-size 16 --num-epochs 10 --learning-rate 2e-5
```

**Best Performance** (8-12 hours):
```bash
--batch-size 8 --num-epochs 20 --learning-rate 1e-5
```

## Common Issues and Solutions

### Issue 1: CUDA Out of Memory

**Symptom**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. Reduce batch size: `--batch-size 4`
2. Enable gradient checkpointing: `--gradient-checkpointing`
3. Use CPU offloading: `--offload-to-cpu`

### Issue 2: Dataset Download Slow

**Symptom**: DeepSEA download takes very long

**Solutions**:
1. Use `--max-examples 1000` for testing
2. Download manually and specify local path
3. Use cached version if available

### Issue 3: Low F1 Score

**Symptom**: Model accuracy is high but F1 is low

**Explanation**: This is expected for imbalanced multi-label tasks. Focus on:
- Macro-averaged F1 (better for rare labels)
- Per-label performance for your specific TFs of interest
- Precision-recall trade-off

## Expected Results

### With Default Settings (3 epochs, max_examples=1000)
- Training time: ~10 minutes (single GPU)
- Macro F1: 0.45-0.55
- Micro F1: 0.60-0.70

### With Full Dataset (10 epochs, full 440K examples)
- Training time: ~4 hours (single GPU)
- Macro F1: 0.65-0.75
- Micro F1: 0.80-0.85

## Advanced Training Configuration

After running the basic examples, you can customize training with these parameters:

### Trainer Selection

OmniGenBench supports three trainer types, each suited for different scenarios:

```bash
# Native PyTorch Trainer (simple, single GPU)
ogb autotrain \
    --dataset deepsea_tfb_prediction \
    --model yangheng/PlantRNA-FM \
    --trainer native \
    --batch-size 16 \
    --num-epochs 10

# Accelerate Trainer (recommended - supports distributed training)
ogb autotrain \
    --dataset deepsea_tfb_prediction \
    --model yangheng/PlantRNA-FM \
    --trainer accelerate \
    --batch-size 32 \
    --num-epochs 10

# HuggingFace Trainer (full HF ecosystem integration)
ogb autotrain \
    --dataset deepsea_tfb_prediction \
    --model yangheng/PlantRNA-FM \
    --trainer hf_trainer \
    --batch-size 16 \
    --num-epochs 10
```

**Trainer Comparison**:

| Trainer | Use Case | Multi-GPU | Features |
|---------|----------|-----------|----------|
| `native` | Simple experiments, debugging | No | Fast setup, minimal dependencies |
| `accelerate` | **Production (recommended)** | Yes | Distributed training, mixed precision |
| `hf_trainer` | HF integration needed | Yes | Full Trainer API, callbacks, logging |

### Learning Rate Tuning

The learning rate is the most critical hyperparameter:

```bash
# Conservative (safer for large models)
ogb autotrain --dataset DATASET --model MODEL --learning-rate 1e-5

# Default (balanced)
ogb autotrain --dataset DATASET --model MODEL --learning-rate 2e-5

# Aggressive (faster convergence, higher risk)
ogb autotrain --dataset DATASET --model MODEL --learning-rate 5e-5
```

**Guidelines**:
- **Large models (>100M params)**: Start with `1e-5` to `2e-5`
- **Medium models (10-100M params)**: Use `2e-5` to `5e-5`
- **Small models (<10M params)**: Can go up to `1e-4`
- **Fine-tuning from scratch**: Use lower rates (1e-5)
- **Transfer learning**: Can use slightly higher rates (5e-5)

**Signs of Poor Learning Rate**:
- Too high: Loss diverges, NaN values, unstable training
- Too low: Loss decreases very slowly, underfitting

### Batch Size Configuration

Batch size affects both memory usage and model convergence:

```python
# Python API with custom batch size
from omnigenbench import AutoTrain

trainer = AutoTrain(
    dataset="deepsea_tfb_prediction",
    model_name_or_path="yangheng/PlantRNA-FM",
    trainer="accelerate"
)

trainer.run(
    batch_size=8,        # Adjust based on GPU memory
    epochs=10,
    learning_rate=2e-5
)
```

**Batch Size Selection**:
- **8**: Small GPU (4-8GB), complex models
- **16**: Medium GPU (8-16GB), standard training
- **32**: Large GPU (16GB+), faster training
- **64+**: Multi-GPU, very fast training

**Trade-offs**:
- Larger batches: Faster training, more stable gradients, requires more memory
- Smaller batches: Less memory, noisier gradients (can be better for generalization)

### Epoch Control

Number of epochs determines how many times the model sees the entire dataset:

```bash
# Quick experiment (testing)
ogb autotrain --dataset DATASET --model MODEL --num-epochs 3

# Standard training
ogb autotrain --dataset DATASET --model MODEL --num-epochs 10

# Thorough training (large datasets)
ogb autotrain --dataset DATASET --model MODEL --num-epochs 20
```

**Best Practices**:
- Start with 3-5 epochs for testing
- Use 10-20 epochs for production
- Monitor validation loss - stop if it plateaus or increases
- Larger datasets typically need fewer epochs
- Smaller datasets may benefit from more epochs (with regularization)

### Advanced Python API Configuration

For full control over training:

```python
from omnigenbench import AutoTrain, AutoConfig

# Load dataset config
config = AutoConfig({
    "task_type": "multi_label_classification",
    "num_labels": 919,
    "train_file": "train.json",
    "test_file": "test.json",
    "max_length": 1000,
    "batch_size": 16,
    "epochs": 10,
    "learning_rate": 2e-5,
    "seeds": [0, 1, 2],  # Multi-seed evaluation for robustness
})

# Initialize trainer
trainer = AutoTrain(
    dataset="deepsea_tfb_prediction",
    model_name_or_path="yangheng/PlantRNA-FM",
    trainer="accelerate",
    overwrite=False  # Don't overwrite existing results
)

# Run with config overrides
trainer.run(
    batch_size=32,                 # Override config batch size
    learning_rate=5e-5,            # Override config learning rate
    epochs=15,                     # Override config epochs
    gradient_accumulation_steps=2  # Effective batch size = 32 * 2
)
```

**Additional Advanced Parameters**:
- `gradient_accumulation_steps`: Simulate larger batch sizes
- `warmup_steps`: Gradual learning rate increase at start
- `weight_decay`: Prevent overfitting (typically 0.01-0.1)
- `save_steps`: Control checkpoint frequency
- `eval_steps`: Evaluation frequency during training
- `seeds`: Multiple runs for statistical robustness

### Multi-Seed Evaluation

For robust performance estimates:

```python
config = AutoConfig({
    "seeds": [0, 1, 2, 3, 4],  # 5 independent runs
    # ... other parameters
})

trainer.run()  # Automatically runs 5 times with different seeds
```

**When to Use Multi-Seed**:
- Publishing results: Report mean ± std across 3-5 seeds
- Model comparison: Ensure differences are statistically significant
- Small datasets: Higher variance requires more seeds
- Large datasets: Single seed often sufficient (1-2 seeds for validation)

## Next Steps

After completing this tutorial, explore:
1. **Custom Dataset**: Use `05_advanced_dataset_creation.ipynb` to create your own TFB dataset
2. **Variant Effect Prediction**: Apply TFB models to assess mutation impacts
3. **Attention Analysis**: Use `examples/attention_score_extraction/` to visualize which sequence regions the model focuses on
4. **Transfer Learning**: Fine-tune on plant-specific or tissue-specific TF binding

## Citation

If you use this example in your research, please cite:

```bibtex
@article{yang2025omnigenome,
  title={OmniGenome: Unified Genomic Foundation Models for Multi-Task Learning},
  author={Yang, Heng and others},
  journal={bioRxiv},
  year={2025}
}
```

## Support

- **Issues**: [GitHub Issues](https://github.com/yangheng95/OmniGenBench/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yangheng95/OmniGenBench/discussions)
- **Documentation**: [Getting Started Guide](../../docs/GETTING_STARTED.md)
