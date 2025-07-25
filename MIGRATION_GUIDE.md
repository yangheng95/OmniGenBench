# Migration Guide: Transitioning to HuggingFace datasets.Dataset

This guide explains how to migrate from the original PyTorch-based OmniDataset classes to the new HuggingFace `datasets.Dataset`-based implementations while preserving all custom behaviors including padding, multiprocessing, and caching.

## Overview

The refactored OmniDataset classes now inherit from HuggingFace's `datasets.Dataset` instead of `torch.utils.data.Dataset`, providing better integration with the HuggingFace ecosystem while maintaining all existing functionality.

### Key Benefits

1. **Better HuggingFace Integration**: Direct compatibility with HuggingFace transformers and datasets ecosystem
2. **Enhanced Caching**: Improved caching mechanisms with both JSON and pickle support
3. **Advanced Filtering/Mapping**: Built-in HuggingFace dataset operations (map, filter, select, etc.)
4. **Memory Efficiency**: Better memory management for large datasets
5. **Preserved Custom Behaviors**: All existing padding, tokenization, and preprocessing logic maintained

## Migration Examples

### 1. Sequence Classification

#### Before (Original PyTorch-based):
```python
from omnigenbench import OmniDatasetForSequenceClassification

dataset = OmniDatasetForSequenceClassification(
    data_source="path/to/data.json",
    tokenizer=tokenizer,
    max_length=512,
    label2id={"positive": 1, "negative": 0},
    cache=True,
    num_proc=4
)

# Use with PyTorch DataLoader
from torch.utils.data import DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

#### After (HuggingFace datasets.Dataset-based):
```python
from omnigenbench import OmniDatasetForSequenceClassificationHF
# or alternatively:
# from omnigenbench import HFOmniDatasetForSequenceClassification as OmniDatasetForSequenceClassificationHF

dataset = OmniDatasetForSequenceClassificationHF(
    data_source="path/to/data.json",
    tokenizer=tokenizer,
    max_length=512,
    label2id={"positive": 1, "negative": 0},
    cache=True,
    num_proc=4
)

# Option 1: Use as HuggingFace Dataset (recommended)
# Access HuggingFace dataset methods
filtered_dataset = dataset.filter(lambda x: len(x['input_ids']) > 100)
mapped_dataset = dataset.map(lambda x: {...})

# Option 2: Convert to PyTorch Dataset for backward compatibility
torch_dataset = dataset.to_torch_dataset()
from torch.utils.data import DataLoader
dataloader = DataLoader(torch_dataset, batch_size=32, shuffle=True, collate_fn=dataset.custom_collate_fn)
```

### 2. Token Classification

#### Before:
```python
from omnigenbench import OmniDatasetForTokenClassification

dataset = OmniDatasetForTokenClassification(
    data_source="path/to/token_data.json",
    tokenizer=tokenizer,
    max_length=256,
    label2id={"B": 1, "I": 2, "O": 0}
)
```

#### After:
```python
from omnigenbench import OmniDatasetForTokenClassificationHF

dataset = OmniDatasetForTokenClassificationHF(
    data_source="path/to/token_data.json",
    tokenizer=tokenizer,
    max_length=256,
    label2id={"B": 1, "I": 2, "O": 0}
)

# Use HuggingFace dataset features
print(dataset.features)  # View dataset schema
subset = dataset.select(range(100))  # Select first 100 examples
```

### 3. Using with HuggingFace Trainer

```python
from transformers import Trainer, TrainingArguments
from omnigenbench import OmniDatasetForSequenceClassificationHF

# Create datasets
train_dataset = OmniDatasetForSequenceClassificationHF(
    data_source="train.json",
    tokenizer=tokenizer,
    max_length=512
)

eval_dataset = OmniDatasetForSequenceClassificationHF(
    data_source="eval.json",
    tokenizer=tokenizer,
    max_length=512
)

# Use directly with HuggingFace Trainer
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    evaluation_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=train_dataset.custom_collate_fn,  # Use custom padding
)

trainer.train()
```

## Key Differences and New Features

### 1. Data Access Patterns

#### Original:
```python
# Access by index
item = dataset[0]
```

#### New (supports both):
```python
# HuggingFace style (recommended)
item = dataset[0]  # Still works
batch = dataset[0:10]  # Slice support

# HuggingFace dataset methods
filtered = dataset.filter(lambda x: x['labels'] != -100)
mapped = dataset.map(lambda x: {'new_field': x['input_ids'][:10]})
```

### 2. Custom Padding Preservation

The new implementation preserves all custom padding behaviors:

```python
dataset = OmniDatasetForSequenceClassificationHF(...)

# Custom collate function is available
dataloader = DataLoader(
    dataset.to_torch_dataset(), 
    batch_size=32, 
    collate_fn=dataset.custom_collate_fn
)
```

### 3. Enhanced Caching

```python
dataset = OmniDatasetForSequenceClassificationHF(
    data_source="large_dataset.json",
    tokenizer=tokenizer,
    cache=True,
    storage_format="json",  # or "pickle"
    num_proc=8  # Parallel processing
)
```

### 4. Multiprocessing Support

All multiprocessing capabilities are preserved:

```python
dataset = OmniDatasetForTokenClassificationHF(
    data_source="token_data.json",
    tokenizer=tokenizer,
    num_proc=4,  # Use 4 processes
    use_threading=False,  # Use processes instead of threads
    error_tolerance=0.1  # Tolerate 10% processing errors
)
```

## Backward Compatibility

The original classes remain available for backward compatibility:

```python
# Original classes still work
from omnigenbench import (
    OmniDatasetForSequenceClassification,  # Original PyTorch-based
    OmniDatasetForTokenClassification,     # Original PyTorch-based
)

# New HuggingFace-based classes
from omnigenbench import (
    OmniDatasetForSequenceClassificationHF,  # New HuggingFace-based
    OmniDatasetForTokenClassificationHF,     # New HuggingFace-based
)
```

## Best Practices for Migration

### 1. Gradual Migration
- Start with new projects using the HF classes
- Migrate existing projects incrementally
- Test thoroughly with your specific datasets

### 2. Leverage HuggingFace Features
```python
dataset = OmniDatasetForSequenceClassificationHF(...)

# Use built-in dataset operations
train_dataset = dataset.filter(lambda x: x['split'] == 'train')
val_dataset = dataset.filter(lambda x: x['split'] == 'val')

# Create balanced subsets
balanced_dataset = dataset.shuffle().select(range(1000))
```

### 3. Custom Preprocessing
```python
def custom_preprocessing(example):
    # Your custom logic here
    example['processed_sequence'] = example['sequence'].upper()
    return example

dataset = OmniDatasetForSequenceClassificationHF(...)
processed_dataset = dataset.map(custom_preprocessing)
```

## Troubleshooting

### Issue: "Cannot import HF dataset classes"
**Solution**: Ensure you have the latest version and datasets library installed:
```bash
pip install datasets>=2.0.0
pip install transformers>=4.0.0
```

### Issue: "Custom collate function not working"
**Solution**: Use the provided custom collate function:
```python
dataloader = DataLoader(
    dataset.to_torch_dataset(),
    batch_size=32,
    collate_fn=dataset.custom_collate_fn
)
```

### Issue: "Memory issues with large datasets"
**Solution**: Use memory-efficient loading:
```python
dataset = OmniDatasetForSequenceClassificationHF(
    data_source="large_file.json",
    max_examples=10000,  # Limit examples for testing
    storage_format="json",  # More memory efficient than pickle
    cache=True
)
```

## Performance Considerations

1. **Caching**: Always enable caching for repeated usage
2. **Parallel Processing**: Use `num_proc` for large datasets
3. **Storage Format**: Use JSON for better memory efficiency, pickle for speed
4. **Batch Size**: Optimize batch size based on your hardware

## Future Deprecation Notice

While the original PyTorch-based classes will remain available for backward compatibility, we recommend migrating to the HuggingFace-based classes for new projects and gradually updating existing ones to benefit from the enhanced features and better ecosystem integration.
