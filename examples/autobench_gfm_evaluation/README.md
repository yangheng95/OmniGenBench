# 🧬 Automated Genomic Foundation Model Benchmarking

This directory contains comprehensive examples for **automated evaluation and benchmarking** of Genomic Foundation Models (GFMs) using OmniGenBench's **AutoBench** framework.

## 📋 Table of Contents

- [Overview](#overview)
- [Benchmark Suites](#benchmark-suites)
- [Quick Start](#quick-start)
- [Available Examples](#available-examples)
- [LoRA Fine-tuning](#lora-fine-tuning)
- [Supported Models](#supported-models)
- [Results Structure](#results-structure)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

AutoBench provides a unified framework for systematically evaluating genomic foundation models across diverse tasks and benchmarks. Key features include:

- ✅ **Standardized Evaluation**: Consistent metrics across different models and tasks
- ✅ **Parameter-Efficient Fine-tuning**: Built-in LoRA support for efficient adaptation
- ✅ **Multi-Task Assessment**: Evaluate across multiple genomic understanding tasks
- ✅ **Automated Pipeline**: End-to-end evaluation with minimal manual intervention
- ✅ **Comprehensive Reporting**: Detailed performance analysis and comparisons

### Why Benchmark?

Benchmarking is essential for:
1. **Model Selection**: Compare different GFMs for your specific application
2. **Performance Validation**: Ensure reliable performance across diverse genomic tasks
3. **Research Advancement**: Contribute to understanding of genomic AI capabilities
4. **Efficiency Analysis**: Evaluate computational trade-offs

## 📊 Benchmark Suites

OmniGenBench supports multiple comprehensive benchmark suites:

| Benchmark | Full Name | Focus | Tasks | Genome Type |
|-----------|-----------|-------|-------|-------------|
| **RGB** | RNA Genome Benchmark | RNA biology | 12+ | RNA |
| **GUE** | Genomic Understanding Evaluation | DNA understanding | 36+ | DNA |
| **GB** | Genome Benchmark | Classic genomics | 9+ | DNA |
| **PGB** | Plant Genome Benchmark | Plant genomics | 7+ | DNA (Plant) |
| **BEACON** | Comprehensive RNA tasks | Multi-domain RNA | 13+ | RNA |

### Task Types

Each benchmark covers various task types:
- **Sequence Classification**: Promoter recognition, functional annotation
- **Token Classification**: Splice site prediction, binding site identification
- **Sequence Regression**: Expression level prediction, stability scores
- **Multi-label Classification**: Multi-functional sequence prediction

## 🚀 Quick Start

### Installation

```bash
# Install OmniGenBench with LoRA support
pip install omnigenbench peft bitsandbytes -U
```

### Basic Usage

```python
from omnigenbench import AutoBench

# Initialize AutoBench for a specific benchmark
bench = AutoBench(
    benchmark="RGB",                    # Choose: RGB, GUE, GB, PGB
    config_or_model="yangheng/OmniGenome-52M",
    overwrite=True,
    trainer='accelerate',               # or 'native', 'hf_trainer'
)

# Run benchmark
bench.run(
    batch_size=8,
    epochs=50,
    seeds=[0, 1, 2],                   # Multiple seeds for robust evaluation
    max_examples=None,                  # Use full dataset
)
```

### Quick Test Run

For rapid testing with limited data:

```python
bench.run(
    batch_size=8,
    epochs=1,
    seeds=0,
    max_examples=1000,  # Use only 1000 examples per task
)
```

## 📁 Available Examples

### 1. `benchmarking_with_lora.ipynb` / `.py`

**Comprehensive LoRA benchmarking tutorial** covering:

- ✅ Setup and configuration
- ✅ Single-model evaluation with LoRA
- ✅ Multi-model comparison
- ✅ Custom LoRA configurations
- ✅ Results analysis and interpretation

**Key Features:**
- Parameter-efficient fine-tuning (LoRA)
- Pre-configured settings for 10+ popular GFMs
- Automatic handling of model-specific requirements
- Batch evaluation across multiple models

**Usage:**
```bash
# Run as Jupyter notebook
jupyter notebook benchmarking_with_lora.ipynb

# Or run as Python script
python benchmarking_with_lora.py
```

## 🔧 LoRA Fine-tuning

### Why LoRA?

**Low-Rank Adaptation (LoRA)** enables efficient fine-tuning of large genomic foundation models:

- **Memory Efficient**: Only ~0.1-1% of parameters are trainable
- **Fast Training**: Significantly reduced training time
- **Easy Switching**: Multiple task-specific adapters can be stored separately
- **No Catastrophic Forgetting**: Base model weights remain frozen

### LoRA Configuration

Example configuration for OmniGenome:

```python
lora_config = {
    "r": 8,                              # Rank of update matrices
    "lora_alpha": 32,                    # Scaling factor
    "lora_dropout": 0.1,                 # Dropout probability
    "target_modules": ["key", "value", "dense"],  # Modules to adapt
    "bias": "none"                       # Bias handling
}

bench.run(
    batch_size=8,
    epochs=50,
    lora_config=lora_config,
)
```

### Pre-configured Models

The following models have optimized LoRA configurations:

```python
SUPPORTED_MODELS = {
    "OmniGenome-52M": {...},
    "OmniGenome-186M": {...},
    "DNABERT-2-117M": {...},
    "hyenadna-large-1m-seqlen-hf": {...},
    "nucleotide-transformer-v2-100m-multi-species": {...},
    "caduceus-ph_seqlen-131k_d_model-256_n_layer-16": {...},
    "rnafm": {...},
    "rnamsm": {...},
    "rnabert": {...},
    "SpliceBERT-510nt": {...},
    "evo-1-131k-base": {...},
}
```

## 🤖 Supported Models

### DNA Foundation Models

| Model | Parameters | Context Length | Best For |
|-------|------------|----------------|----------|
| OmniGenome-52M | 52M | 512-1024 | Fast prototyping |
| OmniGenome-186M | 186M | 512-1024 | Balanced performance |
| DNABERT-2-117M | 117M | 512 | DNA sequence tasks |
| HyenaDNA | 1.6M-1B | Up to 1M | Long sequences |
| Nucleotide Transformer | 50M-2.5B | 1000-6000 | Multi-species |
| Caduceus | Variable | Up to 131k | Bidirectional modeling |

### RNA Foundation Models

| Model | Parameters | Context Length | Best For |
|-------|------------|----------------|----------|
| RNAFM | 100M | 1024 | RNA structure/function |
| RNAMSM | Variable | 1024 | RNA sequence analysis |
| RNABERT | 110M | 440 | RNA classification |
| SpliceBERT | 19M | 510 | Splice site prediction |

### Multi-Modal Models

| Model | Parameters | Context Length | Best For |
|-------|------------|----------------|----------|
| Evo-1 | 7B | 8k-131k | General genomics |

## 📂 Results Structure

After running benchmarks, results are organized as follows:

```
autobench_evaluations/
├── RGB/                              # Benchmark name
│   ├── OmniGenome-52M/              # Model name
│   │   ├── task_1/                  # Individual task
│   │   │   ├── config.json          # Task configuration
│   │   │   ├── metrics.json         # Evaluation metrics
│   │   │   ├── predictions.json     # Model predictions
│   │   │   └── checkpoints/         # Model checkpoints
│   │   ├── task_2/
│   │   └── summary.json             # Overall performance
│   └── DNABERT-2-117M/
└── GUE/

autobench_logs/
├── RGB_OmniGenome-52M_20250102_143022.log
└── GUE_DNABERT-2-117M_20250102_150134.log
```

### Understanding Results

**Metrics JSON Structure:**
```json
{
  "task_name": "promoter_recognition",
  "model": "OmniGenome-52M",
  "metrics": {
    "accuracy": 0.9234,
    "f1_score": 0.9156,
    "precision": 0.9289,
    "recall": 0.9045
  },
  "training_time": 3600,
  "parameters": {
    "trainable": 196608,
    "total": 52000000,
    "lora_enabled": true
  }
}
```

## 🔬 Advanced Usage

### Custom Benchmark Configuration

```python
from omnigenbench import AutoBench

# Create custom benchmark configuration
bench = AutoBench(
    benchmark="RGB",
    config_or_model="yangheng/OmniGenome-186M",
    overwrite=True,
    trainer='accelerate',
    autocast='fp16',           # Mixed precision training
    device='cuda',
    output_dir='./my_results', # Custom output directory
)

# Run with custom training parameters
bench.run(
    batch_size=16,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    epochs=100,
    patience=10,               # Early stopping patience
    eval_steps=50,             # Evaluation frequency
    save_steps=100,            # Checkpoint frequency
    seeds=[0, 1, 2, 3, 4],    # Multiple runs for statistics
    lora_config={
        "r": 16,               # Higher rank for more capacity
        "lora_alpha": 64,
        "lora_dropout": 0.05,
        "target_modules": ["key", "value", "query", "dense"],
    }
)
```

### Multi-Model Comparison

```python
models_to_compare = [
    "yangheng/OmniGenome-52M",
    "yangheng/OmniGenome-186M",
    "zhihan1996/DNABERT-2-117M",
]

results = {}

for model in models_to_compare:
    bench = AutoBench(
        benchmark="RGB",
        config_or_model=model,
        overwrite=True,
    )
    
    metrics = bench.run(
        batch_size=8,
        epochs=50,
        seeds=[0, 1, 2],
    )
    
    results[model] = metrics

# Compare results
import pandas as pd
df = pd.DataFrame(results).T
print(df)
```

### Selective Task Evaluation

```python
# Evaluate only specific tasks within a benchmark
bench = AutoBench(
    benchmark="RGB",
    config_or_model="yangheng/OmniGenome-52M",
    tasks=["task_1", "task_3", "task_5"],  # Specify tasks
    overwrite=True,
)

bench.run(batch_size=8, epochs=50)
```

## 🛠️ Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM)

**Solution:**
```python
# Reduce batch size
bench.run(batch_size=4)  # or smaller

# Use gradient accumulation
bench.run(
    batch_size=4,
    gradient_accumulation_steps=4,  # Effective batch size: 16
)

# Enable mixed precision
bench = AutoBench(
    benchmark="RGB",
    config_or_model=model,
    autocast='fp16',  # Use FP16 precision
)
```

#### 2. Model Loading Errors

**Solution:**
```python
# Specify trust_remote_code for custom models
from transformers import AutoConfig, AutoModel

config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, config=config, trust_remote_code=True)

bench = AutoBench(
    benchmark="RGB",
    config_or_model=model,  # Pass model object directly
)
```

#### 3. LoRA Configuration Issues

**Solution:**
```python
# Check model architecture to find correct target modules
from transformers import AutoModel

model = AutoModel.from_pretrained("your-model")
print(model)  # Inspect model structure

# Common target modules by architecture:
# - BERT-based: ["query", "key", "value", "dense"]
# - GPT-based: ["c_attn", "c_proj"]
# - Hyena: ["in_proj", "out_proj"]
# - Mamba/Caduceus: ["in_proj", "x_proj", "out_proj"]
```

#### 4. Slow Training

**Solution:**
```python
# Use accelerate trainer
bench = AutoBench(
    benchmark="RGB",
    config_or_model=model,
    trainer='accelerate',  # Faster than 'native'
)

# Reduce evaluation frequency
bench.run(
    eval_steps=200,        # Evaluate less frequently
    logging_steps=50,      # Log less frequently
)

# Use smaller dataset for testing
bench.run(max_examples=1000)
```

### Environment Issues

If you encounter package conflicts:

```bash
# Create fresh conda environment
conda create -n omnigenbench python=3.10
conda activate omnigenbench

# Install in correct order
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate peft bitsandbytes
pip install omnigenbench -U
```

## 📊 Performance Tips

### Optimization Strategies

1. **Start Small**: Test with `max_examples=1000` and `epochs=1` first
2. **Use LoRA**: Dramatically reduces memory and training time
3. **Mixed Precision**: Enable `autocast='fp16'` for 2-3x speedup
4. **Batch Size**: Find optimal batch size for your GPU
5. **Gradient Accumulation**: Simulate larger batches without OOM
6. **Multiple Seeds**: Use 3-5 seeds for robust statistical analysis

### Recommended Settings by GPU

| GPU Memory | Batch Size | Model Size | Strategy |
|------------|------------|------------|----------|
| 8GB | 4-8 | ≤100M | FP16 + LoRA |
| 16GB | 8-16 | ≤500M | FP16 + LoRA |
| 24GB | 16-32 | ≤1B | FP16 + LoRA |
| 40GB+ | 32-64 | Any | FP16/BF16 + LoRA |

## 📚 Additional Resources

- **OmniGenBench Documentation**: [https://omnigenbench.readthedocs.io/](https://omnigenbench.readthedocs.io/)
- **Paper**: [OmniGenBench: Automating Large-scale Genomic Foundation Model Evaluation](https://arxiv.org/abs/xxxx.xxxxx)
- **GitHub Repository**: [https://github.com/yangheng95/OmniGenBench](https://github.com/yangheng95/OmniGenBench)
- **Model Hub**: [https://huggingface.co/yangheng](https://huggingface.co/yangheng)

## 🤝 Contributing

We welcome contributions! If you:
- Add support for new models
- Create new benchmark tasks
- Improve evaluation metrics
- Fix bugs or improve documentation

Please submit a pull request or open an issue on GitHub.

## 📝 Citation

If you use OmniGenBench in your research, please cite:

```bibtex
@article{omnigenbench2024,
  title={OmniGenBench: Automating Large-scale Evaluation for Genomic Foundation Models},
  author={Yang, Heng and others},
  journal={bioRxiv},
  year={2024}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Last Updated**: November 2025  
**Maintained By**: YANG, HENG (yangheng95)  
**Contact**: hy345@exeter.ac.uk
