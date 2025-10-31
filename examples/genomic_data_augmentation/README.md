# Genomic Data Augmentation Examples

Augment genomic sequence datasets to improve model robustness and generalization using biologically-informed transformations.

## Overview

Data augmentation is crucial for training robust genomic models, especially when datasets are small or imbalanced. OmniGenBench provides biologically-motivated augmentation techniques that preserve sequence semantics while increasing data diversity.

## Quick Start

### Jupyter Notebook

```bash
cd examples/genomic_data_augmentation
jupyter notebook RNA_Augmentation_Tutorial.ipynb
```

The tutorial demonstrates:
- Random nucleotide substitution
- Reverse complement transformation
- Sequence truncation and padding
- Custom augmentation pipelines
- Applying augmentations during training

### Python API

```python
from omnigenbench import OmniModelForAugmentation

# Initialize augmentation model (uses a masked language model under the hood)
augmentor = OmniModelForAugmentation(
    model_name_or_path="yangheng/OmniGenome-186M",
    noise_ratio=0.15,     # Proportion of tokens to mask per sequence
    instance_num=5,       # Number of augmented variants per input sequence
    batch_size=32         # Batched decoding for speed
)

# Augment a single sequence
original_seq = "AUGCGAUCUCGAGCUACGUCGAUG"
augmented_sequences = augmentor.augment(seq=original_seq, k=5)

for i, aug_seq in enumerate(augmented_sequences, 1):
    print(f"Augmentation {i}: {aug_seq}")
```

## Augmentation Techniques

### 1. MLM-Based Augmentation (Primary Method)
Uses masked language modeling to generate biologically plausible variations:
```python
# Single sequence augmentation
augmented_seq = augmentor.augment_sequence(seq=original_seq)

# Multiple variants of one sequence
augmented_variants = augmentor.augment(seq=original_seq, k=10)

# Batch augmentation for multiple sequences
sequences = ["AUGCGAUCUCGAGCUACGUCGAUG", "GCGCGCGCGCGCGCGCGCGC"]
augmented_batch = augmentor.augment_sequences(sequences)
```

### 2. Noise-Based Masking
The augmentation process applies random masking followed by MLM prediction:
```python
# Configure noise ratio (proportion of tokens to mask)
augmentor = OmniModelForAugmentation(
    model_name_or_path="yangheng/OmniGenome-186M",
    noise_ratio=0.2,     # Mask 20% of tokens
    instance_num=3       # Generate 3 variants per input
)

# Apply augmentation
augmented = augmentor.augment(seq=original_seq)
```

### 3. Batch Processing
Efficiently augment multiple sequences:
```python
from omnigenbench import OmniModelForAugmentation

augmentor = OmniModelForAugmentation(
    model_name_or_path="yangheng/OmniGenome-186M",
    batch_size=32,   # Process 32 masked instances at once
    noise_ratio=0.15,
    instance_num=5
)

# Augment list of sequences
sequences = ["AUGCGAUCUCGAGC", "GCGCGCGCGCGCGC", "TATATATATATAT"]
all_augmented = augmentor.augment_sequences(sequences)
print(f"Generated {len(all_augmented)} total augmented sequences")
```

### 4. File-Based Augmentation
Process sequences from files directly (expects one JSON object per line with key "seq"):
```python
# Augment sequences from input file and save to output file
augmentor.augment_from_file(
    input_file="toy_datasets/test.json",
    output_file="toy_datasets/augmented_sequences.json"
)
```

## Integration with Training

### Pre-Augment Dataset

```python
from omnigenbench import (
    OmniModelForAugmentation,
    OmniDatasetForSequenceClassification
)
import json

# Load original dataset
with open("toy_datasets/train.json", "r") as f:
    original_data = json.load(f)

# Initialize augmentor
augmentor = OmniModelForAugmentation(
    model_name_or_path="yangheng/OmniGenome-186M",
    noise_ratio=0.15,
    instance_num=3  # Generate 3 variants per original sequence
)

# Augment training data
augmented_data = []
for sample in original_data:
    seq = sample["sequence"]
    label = sample["label"]
    
    # Keep original
    augmented_data.append(sample)
    
    # Add augmented variants
    aug_seqs = augmentor.augment(seq, k=3)
    for aug_seq in aug_seqs:
        augmented_data.append({
            "sequence": aug_seq,
            "label": label
        })

# Save augmented dataset
with open("toy_datasets/augmented_train.json", "w") as f:
    json.dump(augmented_data, f, indent=2)

print(f"Original dataset size: {len(original_data)}")
print(f"Augmented dataset size: {len(augmented_data)}")
```

### Train with Augmented Data

```python
from omnigenbench import (
    OmniModelForSequenceClassification,
    OmniDatasetForSequenceClassification,
    Trainer
)

# Initialize tokenizer and model (see tutorials for tokenizer details)
from omnigenbench import OmniTokenizer
tokenizer = OmniTokenizer.from_pretrained("yangheng/OmniGenome-186M", trust_remote_code=True)
model = OmniModelForSequenceClassification(
    model="yangheng/OmniGenome-186M",
    num_labels=2,
    trust_remote_code=True
)

# Load augmented dataset
train_dataset = OmniDatasetForSequenceClassification(
    data_file="toy_datasets/augmented_train.json",
    tokenizer=tokenizer,
    max_length=512
)

# Train model with augmented data
trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    output_dir="./model_with_augmentation",
    epochs=10,
    batch_size=16
)
trainer.train()
```
    tokenizer=tokenizer,
    max_length=512
)

# Augment training data
augmented_data = []
for example in dataset:
    # Keep original
    augmented_data.append(example)
    
    # Add augmented versions
    aug_seqs = augmentor.augment(
        sequence=example['sequence'],
        num_augmentations=3,
        methods=["substitute", "reverse_complement"]
    )
    
    for aug_seq in aug_seqs:
        augmented_data.append({
            'sequence': aug_seq,
            'label': example['label']
        })

print(f"Original size: {len(dataset)}")
print(f"Augmented size: {len(augmented_data)}")
```

### Online Augmentation (During Training)

```python
from torch.utils.data import Dataset

class AugmentedDataset(Dataset):
    def __init__(self, base_dataset, augmentor, num_augmentations=2):
        self.base_dataset = base_dataset
        self.augmentor = augmentor
        self.num_augmentations = num_augmentations
    
    def __len__(self):
        return len(self.base_dataset) * (1 + self.num_augmentations)
    
    def __getitem__(self, idx):
        base_idx = idx // (1 + self.num_augmentations)
        aug_idx = idx % (1 + self.num_augmentations)
        
        example = self.base_dataset[base_idx]
        
        if aug_idx == 0:
            return example  # Return original
        else:
            # Apply augmentation
            aug_seq = self.augmentor.augment(example['sequence'], k=1)[0]
            
            return {
                'sequence': aug_seq,
                'label': example['label']
            }
```

## Augmentation Best Practices

### 1. Start Conservative
Begin with low augmentation rates and gradually increase:
```python
# Conservative: 10% substitution
aug_seq = augmentor.augment(sequence, methods=["substitute"], substitute_rate=0.1)

# Aggressive: 30% substitution (may lose biological meaning)
aug_seq = augmentor.augment(sequence, methods=["substitute"], substitute_rate=0.3)
```

### 2. Preserve Biological Constraints
- For **coding sequences**: Maintain reading frames (multiples of 3)
- For **RNA structures**: Consider base-pairing constraints
- For **regulatory elements**: Avoid disrupting known motifs

### 3. Balance Augmentation
Don't over-augment the dataset:
```python
# Rule of thumb: 2-5x augmentation for small datasets
# 1-2x for medium datasets
# Minimal for large datasets
```

### 4. Validate Augmented Sequences
Check that augmentations maintain label consistency:
```python
# Reverse complement should preserve functionality for most sequences
# But substitutions might change binding sites, splice sites, etc.
```

## Example Toy Dataset

The `toy_datasets/` directory contains example data:
- `train.json` - Training sequences with labels
- `valid.json` - Validation sequences
- `test.json` - Test sequences
- `augmented_sequences.json` - Pre-augmented examples
- `config.py` - Dataset configuration

## Files in This Directory

- **`RNA_Augmentation_Tutorial.ipynb`** - Interactive tutorial (⭐ **START HERE**)
- **`toy_datasets/`** - Example datasets for experimentation
- **`README.md`** - This file

## Requirements

```bash
pip install omnigenbench torch transformers numpy
```

## When to Use Augmentation

✅ **Good Use Cases**:
- Small training datasets (< 10K examples)
- Class imbalance problems
- Improving model robustness
- Transfer learning with limited target data

❌ **Avoid When**:
- Large datasets (> 100K examples) - may not help much
- Augmentations don't preserve labels
- Computational resources are very limited

## Related Examples

- [TFB Prediction](../tfb_prediction/) - Multi-label classification with augmentation
- [mRNA Degradation Rate](../mRNA_degrad_rate_regression/) - Regression with augmentation
- [RNA Structure Prediction](../rna_secondary_structure_prediction/) - Structure-aware augmentation

---

**Next Steps**: Open the tutorial notebook to experiment with genomic data augmentation!
