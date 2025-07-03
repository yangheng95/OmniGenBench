# OmniGenBench Quick Start Guide

## Overview

OmniGenBench is a comprehensive toolkit for genomic foundation models (GFMs) that provides automated benchmarking, custom fine-tuning, and various downstream applications for both RNA and DNA sequence analysis. This guide will help you get started with the toolkit quickly.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Automated Benchmarking](#automated-benchmarking)
4. [Custom Fine-tuning](#custom-fine-tuning)
5. [RNA Applications](#rna-applications)
6. [Advanced Features](#advanced-features)
7. [Supported Models](#supported-models)
8. [Benchmarks](#benchmarks)
9. [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites

- Python 3.10+
- PyTorch 2.5+
- CUDA-compatible GPU (recommended)

### Install OmniGenBench

```bash
# Install from PyPI
pip install omnigenome -U

# Or install from source
git clone https://github.com/yangheng95/OmniGenBench.git
cd OmniGenBench
pip install -e .
```

### Verify Installation

```python
import omnigenome
print(omnigenome.__version__)
```

## Quick Start

### 1. Basic Model Loading

```python
from omnigenome import OmniModelForSequenceClassification
from transformers import AutoTokenizer

# Load a pre-trained model
model_name = "yangheng/OmniGenome-52M"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = OmniModelForSequenceClassification(
    config_or_model=model_name,
    tokenizer=tokenizer,
    num_labels=2
)

# Make predictions
sequence = "AUGGCCUAA"
inputs = tokenizer(sequence, return_tensors="pt")
predictions = model(**inputs)
```

### 2. Sequence Classification

```python
from omnigenome import OmniModelForSequenceClassification, OmniDatasetForSequenceClassification

# Load dataset
dataset = OmniDatasetForSequenceClassification(
    train_file="path/to/train.json",
    test_file="path/to/test.json",
    tokenizer=tokenizer,
    max_length=512
)

# Train model
from omnigenome import Trainer
trainer = Trainer(
    model=model,
    train_dataset=dataset.train_dataset,
    eval_dataset=dataset.test_dataset,
    args=TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        learning_rate=2e-5
    )
)
trainer.train()
```

## Automated Benchmarking

### Command Line Interface

```bash
# Run automated benchmarking on RGB benchmark
autobench --model_name_or_path "yangheng/OmniGenome-186M" --benchmark "RGB" --trainer accelerate

# Available benchmarks: RGB, GB, PGB, GUE
# Available trainers: accelerate, huggingface
```

### Python API

```python
from omnigenome import AutoBench
import autocuda

# Initialize AutoBench
device = autocuda.auto_cuda()
auto_bench = AutoBench(
    benchmark="RGB",  # RNA Genome Benchmark
    model_name_or_path="yangheng/OmniGenome-186M",
    device=device,
    overwrite=True
)

# Run benchmark
auto_bench.run(
    epochs=3,
    batch_size=8,
    seeds=[42, 43, 44]
)
```

### Supported Benchmarks

| Benchmark | Description | Tasks |
|-----------|-------------|-------|
| **RGB** | RNA structure + function | 12 tasks (secondary structure, SNMR, degradation) |
| **BEACON** | RNA multi-domain | 13 tasks (base pairing, mRNA design) |
| **PGB** | Plant long-range DNA | 7 categories (PolyA, enhancer, chromatin) |
| **GUE** | DNA general tasks | 36 datasets (TF binding, promoter detection) |
| **GB** | Classic DNA classification | 9 datasets (enhancer, promoter classification) |

## Custom Fine-tuning

### 1. Token-level vs Sequence-level Tasks

```python
# For token-level regression (e.g., RNA degradation prediction)
from omnigenome import OmniModelForTokenRegression, OmniDatasetForTokenRegression

model = OmniModelForTokenRegression(
    config_or_model=model_name,
    tokenizer=tokenizer,
    num_labels=1
)

dataset = OmniDatasetForTokenRegression(
    train_file="path/to/train.json",
    test_file="path/to/test.json",
    tokenizer=tokenizer,
    max_length=512
)

# For sequence-level classification
from omnigenome import OmniModelForSequenceClassification, OmniDatasetForSequenceClassification

model = OmniModelForSequenceClassification(
    config_or_model=model_name,
    tokenizer=tokenizer,
    num_labels=3  # Number of classes
)
```

### 2. Custom Model Definition

```python
from omnigenome import OmniModel, OmniPooling
import torch

class CustomClassificationModel(OmniModel):
    def __init__(self, config_or_model, tokenizer, *args, **kwargs):
        super().__init__(config_or_model, tokenizer, *args, **kwargs)
        
        # Add custom layers
        self.pooler = OmniPooling(self.config)
        self.classifier = torch.nn.Linear(self.config.hidden_size, self.config.num_labels)
        self.loss_fn = torch.nn.CrossEntropyLoss()
    
    def forward(self, **inputs):
        labels = inputs.pop("labels", None)
        
        # Get model outputs
        last_hidden_state = self.last_hidden_state_forward(**inputs)
        
        # Pooling and classification
        pooled_output = self.pooler(inputs, last_hidden_state)
        logits = self.classifier(pooled_output)
        
        outputs = {"logits": logits}
        
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            outputs["loss"] = loss
            
        return outputs
```

### 3. Custom Tokenizer

```python
from omnigenome import OmniSingleNucleotideTokenizer
from transformers import AutoTokenizer

class CustomTokenizer(OmniSingleNucleotideTokenizer):
    def __init__(self, base_tokenizer=None, **kwargs):
        super().__init__(base_tokenizer, **kwargs)
        self.metadata["tokenizer_name"] = self.__class__.__name__
    
    def __call__(self, sequence, **kwargs):
        # Custom tokenization logic
        sequence = sequence.replace("U", "T")  # RNA to DNA conversion
        tokenized_inputs = self.base_tokenizer(sequence, **kwargs)
        
        # Add custom features
        # ... your custom logic here
        
        return tokenized_inputs
```

## RNA Applications

### 1. RNA Secondary Structure Prediction

```python
from omnigenome import OmniModelForStructuralImputation

# Load model for structure prediction
model = OmniModelForStructuralImputation(
    config_or_model="yangheng/OmniGenome-186M",
    tokenizer=tokenizer
)

# Predict structure
sequence = "AUGGCCUAA"
structure = model.predict(sequence)
print(f"Predicted structure: {structure}")
```

### 2. RNA Sequence Design

```python
from omnigenome import OmniModelForRNADesign

# Initialize RNA design model
model = OmniModelForRNADesign(
    config_or_model="yangheng/OmniGenome-186M",
    tokenizer=tokenizer
)

# Design RNA sequences for target structure
target_structure = "(((....)))"
designed_sequences = model.design(
    structure=target_structure,
    mutation_ratio=0.5,
    num_population=100,
    num_generation=100
)

print(f"Designed sequences: {designed_sequences}")
```

### 3. Command Line RNA Design

```bash
# Design RNA sequences via CLI
omnigenome rna_design \
    --structure "(((....)))" \
    --model "yangheng/OmniGenome-186M" \
    --mutation-ratio 0.5 \
    --num-population 100 \
    --num-generation 100 \
    --output-file "designed_sequences.json"
```

## Advanced Features

### 1. LoRA Fine-tuning

```python
from omnigenome import OmniLoraModel

# Create LoRA model for efficient fine-tuning
lora_model = OmniLoraModel(
    base_model=model,
    lora_config={
        "r": 16,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "v_proj"],
        "lora_dropout": 0.1
    }
)
```

### 2. Model Hub Integration

```python
from omnigenome import ModelHub

# Download models from OmniGenBench hub
model_hub = ModelHub()
available_models = model_hub.list_models()
print(f"Available models: {available_models}")

# Download specific model
model_path = model_hub.download_model("yangheng/OmniGenome-186M")
```

### 3. Pipeline Hub

```python
from omnigenome import PipelineHub

# Use pre-built pipelines
pipeline_hub = PipelineHub()
pipeline = pipeline_hub.load_pipeline("rna_structure_prediction")

# Run pipeline
result = pipeline("AUGGCCUAA")
```

### 4. Data Augmentation

```python
from omnigenome import OmniModelForAugmentation

# Initialize augmentation model
aug_model = OmniModelForAugmentation(
    config_or_model="yangheng/OmniGenome-186M",
    tokenizer=tokenizer
)

# Augment sequences
sequences = ["AUGGCCUAA", "GCCAUUGGC"]
augmented_sequences = aug_model.augment(sequences, num_augmentations=5)
```

## Supported Models

OmniGenBench provides plug-and-play evaluation for over **30 genomic foundation models**, covering both **RNA** and **DNA** modalities:

| Model | Parameters | Pre-training Corpus | Highlights |
|-------|------------|-------------------|------------|
| **OmniGenome** | 186M | 54B plant RNA+DNA tokens | Multi-modal, structure-aware encoder |
| **Agro-NT-1B** | 985M | 48 edible-plant genomes | Billion-scale DNA LM w/ NT-V2 k-mer vocab |
| **RiNALMo** | 651M | 36M ncRNA sequences | Largest public RNA LM; FlashAttention-2 |
| **DNABERT-2** | 117M | 32B DNA tokens, 136 species (BPE) | Byte-pair encoding; 2nd-gen DNA BERT |
| **RNA-FM** | 96M | 23M ncRNA sequences | High performance on RNA structure tasks |
| **RNA-MSM** | 96M | Multi-sequence alignments | MSA-based evolutionary RNA LM |
| **NT-V2** | 96M | 300B DNA tokens (850 species) | Hybrid k-mer vocabulary |
| **HyenaDNA** | 47M | Human chromosomes | Long-context autoregressive model (1Mb) |
| **SpliceBERT** | 19M | 2M pre-mRNA sequences | Fine-grained splice-site recognition |
| **Caduceus** | 1.9M | Human chromosomes | Ultra-compact DNA LM (RC-equivariant) |
| **RNA-BERT** | 0.5M | 4,000+ ncRNA families | Small BERT with nucleotide masking |

## Benchmarks

OmniGenBench supports five curated benchmark suites covering both **sequence-level** and **structure-level** genomics tasks across species:

| Suite | Focus | #Tasks / Datasets | Sample Tasks |
|-------|-------|------------------|--------------|
| **RGB** | RNA structure + function | 12 tasks (SN-level) | RNA secondary structure, SNMR, degradation prediction |
| **BEACON** | RNA (multi-domain) | 13 tasks | Base pairing, mRNA design, RNA contact maps |
| **PGB** | Plant long-range DNA | 7 categories | PolyA, enhancer, chromatin access, splice site |
| **GUE** | DNA general tasks | 36 datasets (9 tasks) | TF binding, core promoter, enhancer detection |
| **GB** | Classic DNA classification | 9 datasets | Human/mouse enhancer, promoter variant classification |

## Dataset Format

### JSON Format for Training

```json
[
    {
        "sequence": "AUGGCCUAA",
        "label": 1
    },
    {
        "sequence": "GCCAUUGGC", 
        "label": 0
    }
]
```

### For Token-level Tasks

```json
[
    {
        "sequence": "AUGGCCUAA",
        "labels": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    }
]
```

## Tutorials

### RNA Design
RNA design is a fundamental problem in synthetic biology, where the goal is to design RNA sequences that fold into a target structure. Find the tutorial in [RNA_Design_Tutorial.ipynb](examples/rna_sequence_design/RNA_Design_Tutorial.ipynb).

### RNA Secondary Structure Prediction
Learn how to predict RNA secondary structures using pre-trained models. Tutorial available in [Secondary_Structure_Prediction_Tutorial.ipynb](examples/rna_secondary_structure_prediction/Secondary_Structure_Prediction_Tutorial.ipynb).

### Custom Fine-tuning
Explore custom fine-tuning examples in the [custom_finetuning](examples/custom_finetuning/) directory.

### AutoBench Evaluation
Learn automated benchmarking in [AutoBench_Tutorial.ipynb](examples/autobench_gfm_evaluation/AutoBench_Tutorial.ipynb).

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```python
   # Reduce batch size
   auto_bench.run(batch_size=2)
   
   # Use gradient accumulation
   trainer = Trainer(
       args=TrainingArguments(
           gradient_accumulation_steps=4
       )
   )
   ```

2. **Model Loading Issues**
   ```python
   # Use trust_remote_code=True for custom models
   tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
   model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
   ```

3. **Tokenizer Compatibility**
   ```python
   # Create custom tokenizer wrapper if needed
   from omnigenome import OmniSingleNucleotideTokenizer
   custom_tokenizer = OmniSingleNucleotideTokenizer(base_tokenizer)
   ```

### Performance Tips

1. **Use LoRA for Efficient Fine-tuning**
   ```python
   lora_model = OmniLoraModel(base_model=model, lora_config={...})
   ```

2. **Enable Mixed Precision Training**
   ```python
   trainer = Trainer(
       args=TrainingArguments(
           fp16=True,  # For GPU training
           bf16=True   # For newer GPUs
       )
   )
   ```

3. **Use Accelerate for Distributed Training**
   ```bash
   autobench --trainer accelerate --model_name_or_path "model_name" --benchmark "RGB"
   ```

## Citation

If you use OmniGenBench in your research, please cite:

```bibtex
@article{yang2024omnigenbench,
      title={OmniGenBench: A Modular Platform for Reproducible Genomic Foundation Models Benchmarking}, 
      author={Heng Yang and Jack Cole, Yuan Li, Renzhi Chen, Geyong Min and Ke Li},
      year={2024},
      eprint={https://arxiv.org/abs/2505.14402},
      archivePrefix={arXiv},
      primaryClass={q-bio.GN},
      url={https://arxiv.org/abs/2505.14402}, 
}
```

## License

OmniGenBench is licensed under the Apache License 2.0. See the LICENSE file for more information.

## Contributing

We welcome contributions to OmniGenBench! If you have any ideas, suggestions, or bug reports, please open an issue or submit a pull request on GitHub.

## Support

- **Documentation**: [GitHub Repository](https://github.com/yangheng95/OmniGenBench)
- **Research Paper**: [arXiv](https://arxiv.org/abs/2505.14402)
- **Examples**: Check the `examples/` directory for detailed tutorials
- **Issues**: Report bugs and request features on GitHub

---

**Happy genomic modeling! 🧬** 