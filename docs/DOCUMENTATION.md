# OmniGenome Documentation

## Overview

OmniGenome is a comprehensive toolkit for genomic foundation models that provides automated benchmarking, training pipelines, and a flexible framework for building custom models and tasks. The framework is designed to handle various genomic data types and tasks including sequence classification, token classification, regression, and more.

## Key Components

### 1. Core Abstract Classes

#### OmniModel
The base class for all models in OmniGenome. It provides:
- Unified interface for model initialization, forward passes, and inference
- Support for different model types (pre-trained paths, PyTorch modules, configs)
- Automatic loss computation and prediction methods
- Model persistence and loading capabilities

```python
from omnigenome import OmniModel

# Initialize from pre-trained model
model = OmniModelForSequenceClassification("model_path", tokenizer)

# Forward pass
outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

# Prediction
predictions = model.predict("ATCGATCG")
```

#### OmniDataset
Abstract base class for all datasets in OmniGenome. Features:
- Support for various data formats (CSV, JSON, Parquet, TXT)
- Automatic tokenization and preprocessing
- Label mapping and data validation
- PyTorch-compatible dataset interface

```python
from omnigenome import OmniDatasetForSequenceClassification

# Initialize dataset
dataset = OmniDatasetForSequenceClassification("data.json", tokenizer, max_length=512)

# Access data
sample = dataset[0]
print(sample['input_ids'].shape)  # torch.Size([512])
```

#### OmniTokenizer
Wrapper class for tokenizers providing:
- Consistent interface across different tokenization strategies
- Support for custom tokenizer wrappers
- Automatic handling of special tokens and padding
- Sequence preprocessing options (U/T conversion, whitespace addition)

```python
from omnigenome import OmniSingleNucleotideTokenizer

# Initialize tokenizer
tokenizer = OmniSingleNucleotideTokenizer.from_pretrained("model_name")

# Tokenize sequences
inputs = tokenizer("ATCGATCG")
print(inputs['input_ids'].shape)  # torch.Size([1, seq_len])
```

#### OmniMetric
Abstract base class for evaluation metrics. Features:
- Integration with scikit-learn metrics
- Support for ignored labels and special cases
- Automatic metric computation and result formatting
- Compatibility with Hugging Face evaluation protocols

```python
from omnigenome import ClassificationMetric

# Initialize metric
metric = ClassificationMetric(ignore_y=-100)

# Compute metric
result = metric.accuracy_score(y_true, y_pred)
print(result)  # {'accuracy_score': 0.85}
```

### 2. Model Implementations

#### Classification Models
- **OmniModelForSequenceClassification**: For sequence-level classification tasks
- **OmniModelForTokenClassification**: For token-level classification tasks
- **OmniModelForMultiLabelSequenceClassification**: For multi-label classification

#### Regression Models
- **OmniModelForSequenceRegression**: For sequence-level regression tasks
- **OmniModelForTokenRegression**: For token-level regression tasks

#### Specialized Models
- **OmniModelForMLM**: For masked language modeling
- **OmniModelForSeq2Seq**: For sequence-to-sequence tasks
- **OmniModelForRNADesign**: For RNA design tasks
- **OmniModelForEmbedding**: For generating embeddings
- **OmniModelForAugmentation**: For data augmentation

### 3. Tokenizer Implementations

#### OmniSingleNucleotideTokenizer
Tokenizes genomic sequences at the nucleotide level:
- Each nucleotide (A, T, C, G, U) becomes a separate token
- Supports U/T conversion and whitespace addition
- Handles special tokens (BOS, EOS) automatically

#### OmniKmersTokenizer
Tokenizes sequences using k-mer approach:
- Groups consecutive nucleotides into k-mers
- Configurable k-mer size and overlap
- Useful for capturing local sequence patterns

#### OmniBPETokenizer
Uses Byte Pair Encoding for tokenization:
- Learns vocabulary from training data
- Efficient for large vocabularies
- Good for capturing complex patterns

### 4. Dataset Implementations

#### Task-Specific Datasets
- **OmniDatasetForSequenceClassification**: For sequence-level classification
- **OmniDatasetForTokenClassification**: For token-level classification
- **OmniDatasetForSequenceRegression**: For sequence-level regression
- **OmniDatasetForTokenRegression**: For token-level regression

#### Features
- Automatic data loading from various formats
- Label mapping and validation
- Sequence truncation and padding
- Support for secondary structure information

### 5. Metric Implementations

#### Classification Metrics
- **ClassificationMetric**: Comprehensive classification evaluation
- Supports all scikit-learn classification metrics
- Handles multi-class and binary classification

#### Regression Metrics
- **RegressionMetric**: Regression task evaluation
- Includes MSE, MAE, R², and other regression metrics

#### Ranking Metrics
- **RankingMetric**: For ranking and recommendation tasks
- Includes NDCG, MAP, and other ranking metrics

### 6. Auto Components

#### AutoBench
Automated benchmarking framework:
- Evaluates models across multiple benchmarks
- Supports multi-seed evaluation for robustness
- Automatic metric tracking and visualization
- Multiple trainer backends (native, accelerate, huggingface)

```python
from omnigenome import AutoBench

# Initialize benchmarking
bench = AutoBench("RGB", "model_name")

# Run evaluation
bench.run()

# View results
bench.mv.summary()
```

#### AutoTrain
Automated training framework:
- Handles model training with minimal configuration
- Supports various training strategies
- Automatic hyperparameter optimization
- Integration with different trainer backends

```python
from omnigenome import AutoTrain

# Initialize training
trainer = AutoTrain("RGB", "model_name")

# Run training
trainer.run()
```

### 7. Hub Components

#### ModelHub
Centralized model repository:
- Access to pre-trained genomic models
- Model metadata and performance information
- Automatic model downloading and caching

#### PipelineHub
Pre-built pipeline repository:
- Ready-to-use training and evaluation pipelines
- Standardized workflows for common tasks
- Easy pipeline customization and extension

#### BenchHub
Benchmark repository:
- Access to standardized genomic benchmarks
- Benchmark metadata and evaluation protocols
- Automatic benchmark downloading and setup

### 8. Utility Functions

#### Data Management
- **load_omni_dataset()**: Load datasets from various sources
- **download_model()**: Download models from the hub
- **download_benchmark()**: Download benchmarks from the hub
- **download_pipeline()**: Download pipelines from the hub

#### Environment Utilities
- **seed_everything()**: Set random seeds for reproducibility
- **env_meta_info()**: Collect environment metadata
- **RNA2StructureCache**: Cache for RNA structure predictions

#### File Operations
- **save_args()**: Save configuration arguments
- **print_args()**: Print configuration information
- **clean_temp_dir_pt_files()**: Clean temporary files

## Usage Examples

### Basic Model Usage

```python
from omnigenome import (
    OmniModelForSequenceClassification,
    OmniSingleNucleotideTokenizer,
    OmniDatasetForSequenceClassification
)

# Initialize components
tokenizer = OmniSingleNucleotideTokenizer.from_pretrained("model_name")
model = OmniModelForSequenceClassification("model_path", tokenizer)
dataset = OmniDatasetForSequenceClassification("data.json", tokenizer)

# Training
from omnigenome import Trainer
trainer = Trainer(model, train_dataset=dataset)
trainer.train()

# Inference
results = model.inference("ATCGATCG")
print(results['predictions'])  # Class label
print(results['confidence'])   # Confidence score
```

### Automated Benchmarking

```python
from omnigenome import AutoBench

# Initialize benchmarking
bench = AutoBench("RGB", "DNABERT-2")

# Run evaluation with custom parameters
bench.run(learning_rate=1e-4, batch_size=16)

# View results
bench.mv.summary(round=4)
```

### Custom Model Development

```python
from omnigenome import OmniModel

class CustomGenomicModel(OmniModel):
    def __init__(self, config_or_model, tokenizer, **kwargs):
        super().__init__(config_or_model, tokenizer, **kwargs)
        # Add custom layers
        self.custom_layer = torch.nn.Linear(self.config.hidden_size, 128)
    
    def forward(self, **inputs):
        # Custom forward pass
        hidden_states = self.last_hidden_state_forward(**inputs)
        custom_output = self.custom_layer(hidden_states)
        return {"custom_output": custom_output}
```

### Custom Tokenizer

```python
from omnigenome import OmniTokenizer

class CustomTokenizer(OmniTokenizer):
    def tokenize(self, sequence, **kwargs):
        # Custom tokenization logic
        tokens = sequence.split()  # Example: space-separated tokens
        return [tokens]
    
    def encode(self, sequence, **kwargs):
        # Custom encoding logic
        tokens = self.tokenize(sequence)
        return [self.base_tokenizer.convert_tokens_to_ids(t) for t in tokens]
```

## Best Practices

### 1. Model Development
- Inherit from appropriate base classes
- Implement required abstract methods
- Handle edge cases and error conditions
- Provide comprehensive docstrings

### 2. Data Processing
- Use appropriate dataset classes for your task
- Validate data formats and labels
- Handle missing or corrupted data gracefully
- Implement proper data augmentation when needed

### 3. Evaluation
- Use appropriate metrics for your task
- Handle class imbalance in classification tasks
- Consider multiple evaluation seeds for robustness
- Document evaluation protocols clearly

### 4. Performance Optimization
- Use appropriate precision (fp16, bf16) for your hardware
- Implement efficient data loading pipelines
- Consider model parallelism for large models
- Profile and optimize bottlenecks

## Contributing

When contributing to OmniGenome:

1. **Follow the existing code structure** and inherit from appropriate base classes
2. **Add comprehensive docstrings** for all public methods and classes
3. **Include usage examples** in docstrings
4. **Add unit tests** for new functionality
5. **Update documentation** when adding new features
6. **Follow the established naming conventions**

## Support

For questions and support:
- Check the examples directory for usage examples
- Review the test files for implementation details
- Open an issue on the GitHub repository
- Consult the inline documentation and docstrings

## License

OmniGenome is licensed under the Apache-2.0 License. See the LICENSE file for details. 