# **OmniGenBench**: From Static Genomic Foundation Models to Dynamic Research Ecosystem

## 1· The Old Paradigm: The Limitation of Static Genomic Foundation Models

Large-scale pre-trained Genomic Foundation Models (GFMs) ushered in a major leap forward, yet most arrived as **static artifacts**, competent but inflexible. A model released to predict transcription-factor binding, for example, did *only* that.

**The fundamental challenges were:**

- **Inflexibility** ,  Adapting a GFM to a new task (e.g., viral-promoter effects, enhancer-promoter interactions) required deep code-level surgery and heavy engineering effort.  
- **Fragmented Ecosystem** ,  A “one model, one task” landscape forced researchers to hunt for ever-new, single-purpose tools.  
- **Untapped Potential** ,  The rich, generalizable “genomic grammar” learned by GFMs remained locked away, unused beyond their narrow original scope.

---

## 2· The New Paradigm: OmniGenBench as the Catalyst for *Exploiting* Foundational Models

A **true** foundation model should be a *starting point*, not a finished product, an extensible platform that invites new, even unforeseen, biological questions.

**OmniGenBench** was built precisely to unlock this potential.  
It transforms static GFMs into **adaptable, multi-purpose research engines**, making extensibility the default so any scientist can innovate without touching core framework code.

---

## 3· The Mechanism: How OmniGenBench Achieves Seamless Extensibility

OmniGenBench uses a clean, object-oriented architecture, think flexible "lego bricks" rather than a rigid pipeline:

| Building Block | Purpose |
| -------------- | ------- |
| `OmniDataset`  | Load & preprocess *any* genomic data format |
| `OmniModel`    | Attach new task-specific "heads" to a pre-trained backbone |
| `Trainer`      | Handle model loading, batching, training loops, evaluation |

**To add a new task, a researcher merely:**

1. **Defines a custom `OmniDataset`** for their data.  
2. **Defines a custom `OmniModel`** (e.g., classification, regression, or bespoke architecture).  

The heavy lifting, backbone management, batching, distributed training, is abstracted away.

---

## 4· Data Template & Supported Formats

OmniGenBench supports flexible data loading for genomic machine learning tasks. To ensure compatibility, your data should follow a simple template and be saved in one of the supported formats.
For more details, see the documentation.

### Data Template: `{sequence, label}` Structure
Each data sample should be a dictionary with at least two keys:
- `sequence`: The biological sequence (DNA, RNA, or protein) as a string.
- `label`: The target value for the task (classification, regression, etc.).

#### Example for Classification
```json
[
  {"sequence": "ATCGATCGATCG", "label": "0"},
  {"sequence": "GCTAGCTAGCTA", "label": "1"}
]
```

#### Example for Regression
```json
[
  {"sequence": "ATCGATCGATCG", "label": 0.75},
  {"sequence": "GCTAGCTAGCTA", "label": -1.2}
]
```

OmniGenBench will automatically standardize common key names. For example, `seq` or `text` will be treated as `sequence`, and `label` will be standardized to `labels` internally.

### Supported Data Formats
OmniGenBench can load data from the following formats:
1. **JSON (`.json`)**: Recommended. A list of dictionaries as shown above. Also supports JSON Lines (`.jsonl`).
2. **CSV (`.csv`)**: Must have columns for `sequence` and `label`.
3. **Parquet (`.parquet`)**: Columns for `sequence` and `label`.
4. **FASTA (`.fasta`, `.fa`, etc.)**: Sequence data only. Labels must be provided separately or inferred.
5. **FASTQ (`.fastq`, `.fq`)**: Sequence and quality scores. Labels must be provided separately or inferred.
6. **BED (`.bed`)**: Genomic intervals. Sequence and label columns may need to be added.
7. **Numpy (`.npy`, `.npz`)**: Array of dictionaries with `sequence` and optional `label`.

For supervised tasks, ensure every sample has both a `sequence` and a `label`. For unsupervised or sequence-only tasks, only the `sequence` key is required.

---

## 5· From Theory to Practice: Demonstrating Unmatched Extensibility

OmniGenBench's flexibility is proven across diverse examples:

### Fine-Tuning for Multi-Label Classification
By adding a multi-label head and a custom dataset class, a standard GFM becomes a high-performance [transcription factor binding predictor](tfb_prediction/quickstart_tfb.ipynb), no framework edits required.

### Zero-Shot Variant Effect Prediction
In [Variant Effect Prediction](variant_effect_prediction/quickstart_vep.ipynb), embeddings from a frozen backbone are compared between reference and alternate alleles, enabling functional impact prediction **without any task-specific training**.

### Generative RNA Design with Evolutionary Algorithms
The [RNA Sequence Design](rna_sequence_design/) module combines genetic algorithms with MLM-guided mutations to design RNA sequences that fold into target secondary structures. This demonstrates OmniGenBench's extensibility beyond pure prediction tasks into **generative modeling for synthetic biology**.

**Quick Example**:
```python
from omnigenbench import OmniModelForRNADesign, OmniTokenizer

# Initialize tokenizer
tokenizer = OmniTokenizer.from_pretrained(
    "yangheng/OmniGenome-186M",
    trust_remote_code=True
)

# Initialize model
designer = OmniModelForRNADesign(
    "yangheng/OmniGenome-186M",
    tokenizer=tokenizer
)

# Design sequences that fold into target structure
target_structure = "(((...)))"  # Simple hairpin
sequences = designer.design(
    structure=target_structure,
    mutation_ratio=0.5,
    num_population=100,
    num_generation=100
)

print(f"Designed {len(sequences)} sequences:")
for seq in sequences[:5]:
    print(f"  {seq}")
```

**Or via CLI**:
```bash
ogb rna_design \
    --structure "(((...)))" \
    --num-population 200 \
    --num-generation 150 \
    --output-file results.json
```

### Genomic Embeddings and Attention Analysis
All OmniModel types support **embedding extraction and attention visualization**:

```python
from omnigenbench import OmniModelForEmbedding, OmniTokenizer

# Load tokenizer
tokenizer = OmniTokenizer.from_pretrained(
    "yangheng/OmniGenome-186M",
    trust_remote_code=True
)

# Initialize embedding model
model = OmniModelForEmbedding(
    "yangheng/OmniGenome-186M",
    tokenizer=tokenizer
)

# Extract embeddings
embedding = model.encode("ATCGATCGATCG", agg="mean")

# Extract attention patterns
attention = model.extract_attention_scores(
    sequence="ATCGATCGATCG",
    layer_indices=[0, 5, 11]
)
```

### Overcoming Practical Barriers  
With a single configuration dictionary, researchers apply [LoRA](autobench_gfm_evaluation/benchmarking_with_lora.ipynb) parameter-efficient fine-tuning and benchmarking, adapting **billion-parameter** models on a single GPU, previously out of reach for most labs.

---

## 6· API Usage Patterns: Choosing the Right Initialization Method

OmniGenBench provides flexible model initialization to suit different workflows. Understanding when to use each pattern is key to writing clean, maintainable code.

### Pattern 1: Constructor with Tokenizer Path (Recommended)

The standard pattern explicitly provides both model and tokenizer:

```python
from omnigenbench import OmniModelForSequenceClassification, OmniTokenizer

# Load tokenizer (always uses from_pretrained)
tokenizer = OmniTokenizer.from_pretrained(
    "yangheng/OmniGenome-186M",
    trust_remote_code=True
)

# Initialize model with constructor
model = OmniModelForSequenceClassification(
    "yangheng/OmniGenome-186M",  # Path as first positional argument
    tokenizer=tokenizer,
    num_labels=2,
    trust_remote_code=True
)

# Make predictions
predictions = model.predict("ATCGATCGATCG")
```

**Benefits**:
- ✅ Explicit control over tokenizer configuration
- ✅ Recommended pattern in the framework
- ✅ Works with all model types
- ✅ Allows tokenizer reuse across models

**Use when**: This is the standard pattern for most workflows.

### Pattern 2: Custom Tokenizer Settings (Advanced)

For specialized preprocessing requirements:

```python
from omnigenbench import OmniModelForSequenceClassification, OmniTokenizer

# Load tokenizer with custom settings
tokenizer = OmniTokenizer.from_pretrained(
    "yangheng/OmniGenome-186M",
    trust_remote_code=True,
    u2t=True,  # Convert U to T for DNA/RNA compatibility
    add_whitespace=False
)

# Initialize model with custom tokenizer
model = OmniModelForSequenceClassification(
    "yangheng/OmniGenome-186M",
    tokenizer=tokenizer,
    num_labels=2
)
```

**Use when**:
- 🧬 Working with mixed DNA/RNA sequences (u2t/t2u conversion)
- ⚙️ Custom tokenizer preprocessing needed
- 🔄 Reusing tokenizer across multiple models
- 🎨 Fine-grained control over tokenization parameters

### Important: OmniTokenizer Always Uses from_pretrained

Tokenizers are consistently loaded via the `from_pretrained()` classmethod:

```python
from omnigenbench import OmniTokenizer

# Always use from_pretrained for tokenizers
tokenizer = OmniTokenizer.from_pretrained(
    "yangheng/OmniGenome-186M",
    trust_remote_code=True
)
```

### Quick Decision Guide

| Scenario | Recommended Pattern | Example |
|----------|---------------------|---------|
| Standard fine-tuning | Constructor + tokenizer | `Model(path, tokenizer, num_labels=2)` |
| Quick inference | Constructor + tokenizer | `Model(path, tokenizer).predict(seq)` |
| Custom tokenization | Constructor + custom tokenizer | `OmniTokenizer.from_pretrained(..., u2t=True)` |
| RNA/DNA conversion | Custom tokenizer | `tokenizer = OmniTokenizer.from_pretrained(..., u2t=True)` |
| Reusing tokenizer | Constructor + shared tokenizer | Load tokenizer once, pass to multiple models |

---

## 7· Advanced Configuration and Parameter Tuning

After running basic examples, you can optimize performance by configuring various parameters. OmniGenBench follows a **"start simple, configure later"** philosophy.

### Inference Configuration

For optimizing model inference performance, see the [AutoInfer Advanced Configuration](autoinfer_examples/README.md#advanced-configuration) guide:

- **Batch size tuning**: Optimize memory usage and throughput
- **Device selection**: Choose between CPU, GPU, or specific CUDA devices
- **Output formats**: Customize result saving and organization

**Quick Reference**:
```bash
# Optimized for large GPU
ogb autoinfer --model MODEL --input-file data.json --batch-size 64 --device cuda:0

# CPU-only inference
ogb autoinfer --model MODEL --input-file data.json --batch-size 16 --device cpu
```

### Training Configuration

For fine-tuning models with custom hyperparameters, see the [TFB Prediction Advanced Training Configuration](tfb_prediction/README.md#advanced-training-configuration):

- **Trainer selection**: Choose between `native`, `accelerate`, or `hf_trainer`
- **Learning rate tuning**: Guidelines for different model sizes
- **Batch size optimization**: Balance memory and convergence
- **Epoch control**: Determine optimal training duration
- **Multi-seed evaluation**: Ensure statistical robustness

**Quick Reference**:
```bash
# Production training with Accelerate
ogb autotrain \
    --dataset DATASET \
    --model MODEL \
    --trainer accelerate \
    --learning-rate 2e-5 \
    --batch-size 32 \
    --num-epochs 10
```

### RNA Design Parameter Tuning

For designing RNA sequences with optimal algorithm parameters, see the [RNA Design Parameter Tuning Guide](rna_sequence_design/README.md#parameter-tuning-guide):

- **Mutation ratio**: Control exploration vs. exploitation
- **Population size**: Balance diversity and computation time
- **Generation count**: Set maximum iterations
- **Combined strategies**: Pre-configured parameter sets for different scenarios

**Quick Reference**:
```bash
# Fast prototyping
ogb rna_design --structure "(((...)))" --mutation-ratio 0.7 --num-population 50 --num-generation 50

# High-quality design (recommended)
ogb rna_design --structure "(((...)))" --mutation-ratio 0.3 --num-population 200 --num-generation 200
```

### Embedding Configuration

For extracting and using genomic embeddings effectively, see the [Genomic Embeddings Advanced Configuration](genomic_embeddings/README.md#advanced-embedding-configuration):

- **Aggregation strategies**: Choose between `mean`, `max`, `cls`, `last`
- **Batch processing**: Optimize throughput for large datasets
- **Device management**: Control hardware usage
- **Layer-specific embeddings**: Extract features from different transformer layers
- **Task-adapted embeddings**: Use fine-tuned models for specialized embeddings

**Quick Reference**:
```python
# Standard embedding extraction
embeddings = model.batch_encode(sequences, batch_size=32, agg="mean")

# Task-adapted embeddings from fine-tuned model
tfb_model = OmniModelForSequenceClassification("yangheng/ogb_tfb_finetuned", num_labels=919)
specialized_emb = tfb_model.encode(sequence, agg="mean")
```

### Configuration Priority

When a parameter is specified in multiple places, the priority order is:

1. **Runtime arguments** (highest priority) - Command-line flags or function arguments
2. **Config file** - Dataset-specific `config.py` files
3. **Default values** (lowest priority) - Framework defaults

**Example**:
```python
# config.py has learning_rate=2e-5
# This overrides it to 5e-5 at runtime
trainer.run(learning_rate=5e-5)
```

### Where to Find More Configuration Options

- **CLI Help**: Run `ogb COMMAND --help` for all available options
- **API Documentation**: See [docs/api_reference.rst](../docs/api_reference.rst)
- **Example Notebooks**: Explore Jupyter notebooks in each example directory
- **AutoConfig Class**: See `omnigenbench/auto/config/auto_config.py` for full parameter list

---

## 8· Conclusion: A Unique and Necessary Framework for Modern Genomics

OmniGenBench is **not** just a model hub or a training-script wrapper, it is a paradigm shift. By turning GFMs into living, extensible platforms, it empowers scientists to move beyond pre-defined tasks and invent novel solutions to pressing biological problems. To our knowledge, **no other genomics framework offers this depth of seamless, practical extensibility**, making OmniGenBench essential for realizing the full promise of foundation models in genomics.
