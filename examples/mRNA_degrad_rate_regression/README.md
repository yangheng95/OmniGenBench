# mRNA Degradation Rate Prediction Examples

Predict mRNA degradation rates using sequence-based regression models - a crucial task for understanding gene expression dynamics and designing stable therapeutics.

## Overview

mRNA degradation rate prediction is a **regression task** where we predict the half-life or stability of mRNA molecules based on their sequence. This is important for:
- **Drug design**: Creating stable mRNA vaccines/therapeutics
- **Gene expression**: Understanding post-transcriptional regulation
- **Synthetic biology**: Designing mRNAs with desired stability
- **RNA therapeutics**: Optimizing delivery and expression

## Quick Start

### Jupyter Notebook

```bash
cd examples/mRNA_degrad_rate_regression
jupyter notebook mRNA_degrade_regression.ipynb
```

The tutorial covers:
- Loading mRNA degradation datasets
- Model initialization for regression
- Training regression models
- Evaluating prediction accuracy
- Making predictions on new sequences

### Python API

```python
from omnigenbench import OmniModelForSequenceRegression, OmniTokenizer
import torch

# Load tokenizer
tokenizer = OmniTokenizer.from_pretrained(
    "yangheng/OmniGenome-186M",
    trust_remote_code=True
)

# Load model for regression
model = OmniModelForSequenceRegression(
    "yangheng/OmniGenome-186M",
    tokenizer=tokenizer,
    num_labels=1,  # Single continuous value
    trust_remote_code=True
)

model = model.to(torch.device("cuda")).eval()

# Predict degradation rate
sequence = "AUGCGAUCUCGAGCUACGUCGAUGCUAGCUCGAUG"
prediction = model.predict(sequence)

print(f"Predicted degradation rate: {prediction['predictions'][0]:.4f}")
```

## Dataset Structure

mRNA degradation datasets typically contain:
- **sequence**: mRNA sequence (5'UTR + CDS + 3'UTR)
- **label**: Degradation rate (continuous value)

Example (`toy_datasets/RNA-mRNA/train.json`):
```json
[
  {
    "sequence": "AUGCGAUCUCGAGCUACGUCGAUG...",
    "label": 2.34
  },
  {
    "sequence": "CGGAUACGGCUAGUCUCGAGCUAC...",
    "label": 1.87
  }
]
```

## Training Example

```python
from omnigenbench import (
    OmniModelForSequenceRegression,
    OmniDatasetForSequenceRegression,
    OmniTokenizer,
    AutoTrain
)

# Initialize tokenizer
tokenizer = OmniTokenizer.from_pretrained(
    "yangheng/OmniGenome-186M",
    trust_remote_code=True
)

# Load dataset
train_dataset = OmniDatasetForSequenceRegression(
    train_file="toy_datasets/RNA-mRNA/train.json",
    tokenizer=tokenizer,
    max_length=512
)

# Initialize model
model = OmniModelForSequenceRegression(
    "yangheng/OmniGenome-186M",
    tokenizer=tokenizer,
    num_labels=1,
    trust_remote_code=True
)

# Train model
trainer = AutoTrain(
    model=model,
    train_dataset=train_dataset,
    config={
        "output_dir": "./mRNA_degradation_model",
        "num_train_epochs": 10,
        "per_device_train_batch_size": 16,
        "learning_rate": 2e-5,
        "save_strategy": "epoch",
        "eval_strategy": "epoch",
        "logging_steps": 50
    }
)

trainer.train()
```

## Evaluation Metrics

For regression tasks, use metrics like:

```python
from omnigenbench import RegressionMetric
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# During training
regression_metric = RegressionMetric()

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    
    mse = mean_squared_error(labels, predictions)
    mae = mean_absolute_error(labels, predictions)
    r2 = r2_score(labels, predictions)
    
    return {
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "rmse": mse ** 0.5
    }
```

## Use Cases

### 1. mRNA Vaccine Design

```python
# Evaluate stability of vaccine sequences
vaccine_sequences = [
    "AUGCCCGGGAAATTTGGGCCCAAA...",  # Candidate 1
    "CGGAUACGGCUAGUCUCGAGCUAC...",  # Candidate 2
]

predictions = model.batch_predict(vaccine_sequences)

for seq, pred in zip(vaccine_sequences, predictions):
    print(f"Sequence: {seq[:30]}...")
    print(f"Predicted stability: {pred['predictions'][0]:.4f}\n")
```

### 2. UTR Optimization

```python
# Test different 3'UTRs for stability
utr_variants = {
    "short": "AUG...CDS...UAAA",
    "long": "AUG...CDS...AUUUAAUAAAA",
    "optimized": "AUG...CDS...AUUAUUUAAA"
}

for name, seq in utr_variants.items():
    pred = model.predict(seq)
    print(f"{name}: {pred['predictions'][0]:.4f}")
```

### 3. Mutation Effect Analysis

```python
# Analyze how mutations affect stability
reference_seq = "AUGCGAUCUCGAGCUACGUCGAUG"
ref_stability = model.predict(reference_seq)['predictions'][0]

# Introduce mutation
mutant_seq = "AUGCGAUCCCGAGCUACGUCGAUG"  # U->C at position 9
mut_stability = model.predict(mutant_seq)['predictions'][0]

effect = mut_stability - ref_stability
print(f"Mutation effect: {effect:+.4f}")
print(f"{'Stabilizing' if effect > 0 else 'Destabilizing'}")
```

## Model Architectures

### Standard Regression Head

```python
# Single output neuron for regression
model = OmniModelForSequenceRegression(
    model="yangheng/OmniGenome-186M",
    num_labels=1
)
```

### Multi-Task Regression

```python
# Predict multiple properties simultaneously
model = OmniModelForSequenceRegression(
    model="yangheng/OmniGenome-186M",
    num_labels=3  # e.g., half-life, abundance, ribosome occupancy
)
```

## Files in This Directory

- **`mRNA_degrade_regression.ipynb`** - Interactive tutorial (⭐ **START HERE**)
- **`toy_datasets/`** - Example datasets:
  - `RNA-mRNA/` - mRNA degradation data
  - `Archive2/` - Alternative dataset
- **`README.md`** - This file

## Requirements

```bash
pip install omnigenbench torch transformers scikit-learn numpy pandas
```

## Best Practices

### 1. Feature Engineering
Consider including:
- Sequence length
- GC content
- Secondary structure stability
- Codon usage

### 2. Data Normalization
Normalize target values for better training:
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
labels_normalized = scaler.fit_transform(labels.reshape(-1, 1))

# Later, inverse transform predictions
predictions_original = scaler.inverse_transform(predictions)
```

### 3. Cross-Validation
Use k-fold CV for robust evaluation:
```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(sequences)):
    print(f"Training fold {fold+1}...")
    # Train model on train_idx
    # Evaluate on val_idx
```

### 4. Ensemble Methods
Combine multiple models for better predictions:
```python
# Train multiple models with different seeds
models = [train_model(seed=i) for i in range(5)]

# Average predictions
ensemble_pred = sum(m.predict(seq) for m in models) / len(models)
```

## Performance Optimization

### Batch Prediction
```python
# Efficient batch processing
predictions = model.batch_predict(
    sequences=large_sequence_list,
    batch_size=32
)
```

### GPU Utilization
```python
# Use mixed precision for faster training
model = model.to(torch.device("cuda")).half()
```

## Toy Datasets

Two example datasets are provided:

### RNA-mRNA Dataset
- **Size**: ~1000 sequences
- **Task**: Predict mRNA half-life
- **Labels**: Continuous degradation rates

### Archive2 Dataset
- **Size**: ~500 sequences
- **Task**: Alternative regression benchmark
- **Labels**: Normalized stability scores

## Related Examples

- [Translation Efficiency Prediction](../translation_efficiency_prediction/) - Related RNA regression task
- [RNA Structure Prediction](../rna_secondary_structure_prediction/) - Structure-based features
- [TFB Prediction](../tfb_prediction/) - Classification variant

---

**Next Steps**: Open the tutorial notebook to train your first mRNA degradation prediction model!
