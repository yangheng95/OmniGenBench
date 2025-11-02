# 🧬 RNA Secondary Structure Prediction Tutorials

This directory contains a comprehensive, modular tutorial series for RNA secondary structure prediction using OmniGenBench. The tutorials follow best practices in computational biology education and software design.

## 📚 Tutorial Structure

### Quick Start
- **[00_quickstart_rna_ssp.ipynb](00_quickstart_rna_ssp.ipynb)** - Complete workflow in ~30 lines of code
  - Perfect for experienced users who want to see the full pipeline
  - Demonstrates all 4 steps with minimal explanation
  - Great reference for quick implementation

### Detailed Step-by-Step Tutorials

#### 1. Data Preparation
- **[01_data_preparation.ipynb](01_data_preparation.ipynb)** - Understanding and preparing RNA structure data
  - Biological background: What is RNA secondary structure?
  - Task framing: Why this is token classification
  - Dataset overview: bpRNA structure annotations
  - Data loading with OmniDataset framework
  - Data quality checks and visualization

#### 2. Model Initialization  
- **[02_model_initialization.ipynb](02_model_initialization.ipynb)** - Loading and configuring the model
  - Foundation model concepts
  - OmniGenome architecture for token classification
  - Tokenizer setup and configuration
  - Model-task alignment principles

#### 3. Model Training
- **[03_model_training.ipynb](03_model_training.ipynb)** - Fine-tuning for structure prediction
  - Training loop concepts
  - AccelerateTrainer configuration
  - Evaluation metrics (F1, Accuracy)
  - Model checkpointing and saving

#### 4. Model Inference
- **[04_model_inference.ipynb](04_model_inference.ipynb)** - Predicting on new sequences
  - Loading trained models
  - Running predictions
  - Structure validation
  - Results interpretation

### Legacy Tutorials (For Reference)
- `Secondary_Structure_Prediction_Tutorial.ipynb` - Original comprehensive tutorial
- `ZeroShot_Structure_Prediction_Tutorial.ipynb` - Zero-shot prediction examples

## 🎯 Learning Path

### For Beginners
1. Start with [00_fundamental_concepts.ipynb](../00_fundamental_concepts.ipynb) (in parent directory)
2. Follow the Quick Start: [00_quickstart_rna_ssp.ipynb](00_quickstart_rna_ssp.ipynb)
3. Deep dive into each step: 01 → 02 → 03 → 04

### For Experienced Users
1. Quick Start: [00_quickstart_rna_ssp.ipynb](00_quickstart_rna_ssp.ipynb)
2. Reference specific tutorials as needed

## 🔬 What You'll Learn

### Biological Concepts
- RNA secondary structure and its importance
- Base-pairing patterns and structural elements
- Structure-function relationships

### Machine Learning Concepts
- Token classification task framing
- Foundation models for genomics
- Fine-tuning strategies
- Evaluation metrics for structured prediction

### OmniGenBench Framework
- `OmniDatasetForTokenClassification` usage
- `OmniModelForTokenClassification` configuration
- `AccelerateTrainer` for efficient training
- Model saving and loading via `ModelHub`

## 💻 Quick Start Example

```python
# Complete workflow in 4 steps
from omnigenbench import *

# 1. Load data
tokenizer = OmniTokenizer.from_pretrained("yangheng/OmniGenome-52M")
datasets = OmniDatasetForTokenClassification.from_hub(
    "RNA-SSP-Archive2", tokenizer=tokenizer, 
    label2id={"(": 0, ")": 1, ".": 2}
)

# 2. Initialize model
model = OmniModelForTokenClassification(
    "yangheng/OmniGenome-52M", tokenizer,
    label2id={"(": 0, ")": 1, ".": 2}
)

# 3. Train
trainer = AccelerateTrainer(
    model=model,
    train_dataset=datasets["train"],
    eval_dataset=datasets["valid"],
    compute_metrics=[ClassificationMetric().f1_score]
)
trainer.train()
trainer.save_model("my_rna_ssp_model")

# 4. Predict
model = ModelHub.load("my_rna_ssp_model")
result = model.inference("AUGCCGUGCAUUAA")
print(result["predictions"])  # ['(', '(', '(', '.', '.', '.', ')', ')', ')', '.', '.', '.', '.', '.']
```

## 🏗️ Design Philosophy

This tutorial series follows the OmniGenBench design philosophy:

1. **Modularity**: Each tutorial focuses on one concept
2. **Framework-First**: Leverage built-in OmniGenBench capabilities
3. **Educational**: Explain the "why" before the "how"
4. **Practical**: Real code that runs out-of-the-box
5. **Modern**: Use latest best practices (AccelerateTrainer, OmniDataset)

## 📊 Dataset Information

### bpRNA Dataset (RNA-SSP-Archive2)
- **Source**: Curated RNA secondary structure database
- **Size**: Thousands of experimentally validated structures
- **Format**: Sequence-structure pairs in dot-bracket notation
- **Split**: Pre-divided into train/valid/test sets

### Label Format
```
Sequence:   AUGCCGUGC
Structure:  .(((...)))
            
Legend:
  ( = Opening base pair (left bracket)
  ) = Closing base pair (right bracket)  
  . = Unpaired nucleotide
```

## 🛠️ Technical Requirements

### Software
- Python 3.8+
- PyTorch 1.12+
- CUDA-capable GPU (recommended)
- omnigenbench >= 0.3.23

### Installation
```bash
pip install omnigenbench -U
```

## 📖 Additional Resources

- **OmniGenBench Documentation**: [docs/GETTING_STARTED.md](../../docs/GETTING_STARTED.md)
- **Paper**: RNA-FM: Foundation Model for RNA Biology (if applicable)
- **GitHub Issues**: Report bugs or request features

## 🤝 Contributing

Found an issue or have a suggestion? Please:
1. Check existing issues
2. Open a new issue with detailed description
3. Or submit a pull request

## 📝 Citation

If you use these tutorials or OmniGenBench in your research:

```bibtex
@software{omnigenbench2024,
  title={OmniGenBench: A Unified Framework for Genomic Foundation Models},
  author={YANG, HENG and others},
  year={2024},
  url={https://github.com/yangheng95/OmniGenBench}
}
```

## 📧 Contact

- **Author**: YANG, HENG
- **Email**: hy345@exeter.ac.uk
- **Homepage**: https://yangheng95.github.io
- **GitHub**: https://github.com/yangheng95

---

**Last Updated**: 2025-01-11  
**Version**: 0.3.23alpha

