# RNA Secondary Structure Prediction Examples

Predict and visualize RNA secondary structures (2D folding patterns) from sequences using deep learning and traditional methods.

## Overview

RNA secondary structure prediction determines how an RNA molecule folds by identifying:
- **Base pairs**: Watson-Crick and wobble pairs (A-U, G-C, G-U)
- **Loops**: Hairpin, bulge, internal, and multi-branch loops
- **Stems**: Stacked base pairs forming helical regions

This is crucial for understanding RNA function, stability, and interactions.

## Files in This Directory

- **`Secondary_Structure_Prediction_Tutorial.ipynb`** - Standard structure prediction (**START HERE**)
- **`ZeroShot_Structure_Prediction_Tutorial.ipynb`** - Zero-shot prediction without training
- **`Secondary_Structure_Prediction.py`** - Python script implementation
- **`enhanced_ssp_demo.py`** - Advanced features demonstration
- **`toy_datasets/`** - Example datasets for training and evaluation

## Quick Start

### Option 1: Jupyter Notebooks

#### Standard Prediction
```bash
jupyter notebook Secondary_Structure_Prediction_Tutorial.ipynb
```

Covers:
- Training structure prediction models
- Evaluating on benchmark datasets
- Dot-bracket notation
- Visualization of predictions

#### Zero-Shot Prediction
```bash
jupyter notebook ZeroShot_Structure_Prediction_Tutorial.ipynb
```

Covers:
- Prediction without fine-tuning
- Using pre-trained models directly
- Ensemble methods
- Comparison with ViennaRNA

### Option 2: Python Scripts

```bash
# Run basic structure prediction
python Secondary_Structure_Prediction.py

# Run enhanced demo with visualization
python enhanced_ssp_demo.py
```

## Python API Examples

### Basic Structure Prediction

```python
from omnigenbench import OmniModelForTokenClassification, OmniTokenizer
import torch

# Load tokenizer
tokenizer = OmniTokenizer.from_pretrained(
    "yangheng/OmniGenome-186M",
    trust_remote_code=True
)

# Load model for structure prediction
model = OmniModelForTokenClassification(
    "yangheng/OmniGenome-186M",
    tokenizer=tokenizer,
    num_labels=3,  # 3 classes: unpaired (.), paired-left ((, paired-right ))
    trust_remote_code=True
)

model = model.to(torch.device("cuda")).eval()

# Predict structure
sequence = "GGCCUUAGCUCAGCGGUUAGAGCGCACCCCUGAUAAGGGUGAGGUCGCUGAUUCGAUCCAGCUAAGGCCACCA"
prediction = model.predict(sequence)

structure = prediction['predictions'][0]  # Dot-bracket notation
print(f"Sequence:  {sequence}")
print(f"Structure: {structure}")
```

### Zero-Shot Prediction

Zero-shot prediction uses pre-trained models **without fine-tuning** to predict RNA structures. This is useful when:
- You lack labeled training data
- You need quick predictions
- You want to assess model's intrinsic understanding

**Method 1: Using ViennaRNA with Pre-trained Embeddings**

```python
from omnigenbench import OmniModelForEmbedding, OmniTokenizer
import RNA  # ViennaRNA Python binding

# Load pre-trained model for embeddings
tokenizer = OmniTokenizer.from_pretrained(
    "yangheng/OmniGenome-186M",
    trust_remote_code=True
)

model = OmniModelForEmbedding(
    "yangheng/OmniGenome-186M",
    tokenizer=tokenizer,
    trust_remote_code=True
)

# Extract embeddings
sequence = "GGCCUUAGCUCAGCGGUUAGAGCGCACCCCUGAUAAGGGUGAGGUCGCUGAUUCGAUCCAGCUAAGGCCACCA"
embedding = model.encode(sequence, agg="tokens")  # Token-level embeddings

# Use ViennaRNA for structure prediction
structure, mfe = RNA.fold(sequence)
print(f"Sequence:  {sequence}")
print(f"Structure: {structure}")
print(f"MFE:       {mfe:.2f} kcal/mol")
```

**Method 2: Ensemble with Deep Learning Models**

Combine ViennaRNA's thermodynamic model with learned representations:
sequence = "GGCCUUAGCUCAGCGGUUAGAGCGCACCCCUGAUAAGGGUGAGGUCGCUGAUUCGAUCCAGCUAAGGCCACCA"
embedding = model.encode(sequence, agg="tokens")  # Token-level embeddings

# Use ViennaRNA for structure prediction
structure, mfe = RNA.fold(sequence)
print(f"Structure: {structure}")
print(f"Free energy: {mfe:.2f} kcal/mol")
```

### Training a Structure Prediction Model

```python
from omnigenbench import (
    OmniModelForTokenClassification,
    OmniDatasetForTokenClassification,
    OmniTokenizer,
    AutoTrain
)

# Initialize tokenizer
tokenizer = OmniTokenizer.from_pretrained(
    "yangheng/OmniGenome-186M",
    trust_remote_code=True
)

# Load dataset with structure annotations
train_dataset = OmniDatasetForTokenClassification(
    train_file="toy_datasets/Archive2/train.json",
    tokenizer=tokenizer,
    max_length=512,
    label_type="structure"  # Structure prediction task
)

# Initialize model
model = OmniModelForTokenClassification(
    model="yangheng/OmniGenome-186M",
    num_labels=3,  # . ( )
    trust_remote_code=True
)

# Train
trainer = AutoTrain(
    model=model,
    train_dataset=train_dataset,
    config={
        "output_dir": "./rna_structure_model",
        "num_train_epochs": 20,
        "per_device_train_batch_size": 8,
        "learning_rate": 2e-5,
        "save_strategy": "epoch"
    }
)

trainer.train()
```

## Structure Notation

### Dot-Bracket Notation

RNA structures are represented in dot-bracket format:
- `.` - Unpaired nucleotide
- `(` - Start of base pair (5' side)
- `)` - End of base pair (3' side)

**Example**:
```
Sequence:  GGCCUUAGCU
Structure: ((((...))))
           ||||   ||||
           Stem   Stem
```

### Extended Notations

For pseudoknots and complex structures:
- `[ ]` - Pseudoknot pairs
- `{ }` - Additional pseudoknot pairs
- `< >` - Tertiary interactions

## Visualization

### Plot Structure

```python
import matplotlib.pyplot as plt

def visualize_structure(sequence, structure):
    """Simple ASCII visualization of RNA structure."""
    fig, ax = plt.subplots(figsize=(15, 3))
    
    # Plot sequence
    for i, (base, struct) in enumerate(zip(sequence, structure)):
        color = 'red' if struct == '.' else 'blue'
        ax.text(i, 0, base, ha='center', va='center', color=color, fontsize=12)
        ax.text(i, -0.5, struct, ha='center', va='center', fontsize=10)
    
    ax.set_xlim(-1, len(sequence))
    ax.set_ylim(-1, 1)
    ax.axis('off')
    plt.title("RNA Secondary Structure")
    plt.tight_layout()
    plt.savefig("structure_viz.png", dpi=300, bbox_inches='tight')
    plt.show()

visualize_structure(sequence, structure)
```

### Advanced Visualization with ViennaRNA

```python
import ViennaRNA as RNA

# Generate structure layout
structure_plot = RNA.plot_structure(structure, sequence)
```

## Evaluation Metrics

### Base-Pair Distance

```python
def base_pair_distance(pred_struct, true_struct):
    """Count mismatched base pairs."""
    pred_pairs = extract_pairs(pred_struct)
    true_pairs = extract_pairs(true_struct)
    
    tp = len(pred_pairs & true_pairs)
    fp = len(pred_pairs - true_pairs)
    fn = len(true_pairs - pred_pairs)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {"precision": precision, "recall": recall, "f1": f1}
```

### Matthews Correlation Coefficient (MCC)

```python
from sklearn.metrics import matthews_corrcoef

def structure_mcc(pred_struct, true_struct):
    """Compute MCC for structure prediction."""
    # Convert to binary labels (paired vs unpaired)
    pred_binary = [0 if c == '.' else 1 for c in pred_struct]
    true_binary = [0 if c == '.' else 1 for c in true_struct]
    
    return matthews_corrcoef(true_binary, pred_binary)
```

## Benchmarks

### Standard Datasets

The `toy_datasets/` directory contains:

#### Archive2 Dataset
- **Source**: Archiveii benchmark
- **Size**: ~2,500 structures
- **Length**: 50-400 nucleotides
- **Diversity**: Various RNA families

#### RNA-mRNA Dataset
- **Source**: mRNA structures
- **Size**: ~1,000 structures
- **Focus**: 5'UTR and 3'UTR regions

## Use Cases

### 1. RNA Design Validation

```python
# Check if designed sequence folds correctly
designed_seq = "GGCCUUAGCUCAGCGGUUAGAGCGCACCCCUGAUAAGGGUGAGGUCGCUGAUUCGAUCCAGCUAAGGCCACCA"
target_structure = "(((((((..(((((.......))))).(((((.......)))))....(((((.......)))))))))))).."

pred_structure = model.predict(designed_seq)['predictions'][0]

similarity = calculate_similarity(pred_structure, target_structure)
print(f"Structure similarity: {similarity:.2%}")
```

### 2. Mutation Effect on Structure

```python
# Analyze how mutations affect folding
reference_seq = "GGCCUUAGCU"
mutant_seq = "GGCCAUAGCU"  # U->A mutation

ref_struct = model.predict(reference_seq)['predictions'][0]
mut_struct = model.predict(mutant_seq)['predictions'][0]

print(f"Reference: {ref_struct}")
print(f"Mutant:    {mut_struct}")
```

### 3. RNA Family Classification

```python
# Use structure features for classification
sequences = [...]  # Your sequences
structures = [model.predict(seq)['predictions'][0] for seq in sequences]

# Extract structure features
features = extract_structure_features(structures)
# Use for clustering or classification
```

## Advanced Features

### Ensemble Prediction

Combine multiple models for better accuracy:
```python
models = [
    load_model("model_1"),
    load_model("model_2"),
    load_model("model_3")
]

# Ensemble voting
predictions = [m.predict(sequence) for m in models]
consensus = vote(predictions)
```

### Constraint-Based Prediction

Incorporate known base pairs as constraints:
```python
# Use ViennaRNA with constraints
known_pairs = [(0, 9), (1, 8), (2, 7)]  # Indices of known pairs
constrained_structure = RNA.fold_constrained(sequence, known_pairs)
```

## Gradio Web Interface

The `.gradio/` directory contains a web interface for interactive structure prediction:

```bash
python enhanced_ssp_demo.py
```

This launches a Gradio app where you can:
- Input sequences manually
- Upload FASTA files
- Visualize predictions interactively
- Compare with ViennaRNA predictions

## Requirements

```bash
pip install omnigenbench torch transformers ViennaRNA matplotlib numpy
```

For advanced visualization:
```bash
pip install forgi drawrna  # Optional
```

## Best Practices

### 1. Sequence Preprocessing
- Remove non-standard nucleotides
- Convert DNA to RNA (T->U)
- Trim to reasonable length (< 500 nt for best results)

### 2. Model Selection
- Use fine-tuned models for specific RNA families
- Ensemble multiple models for critical predictions
- Compare with physics-based methods (ViennaRNA)

### 3. Post-Processing
- Filter impossible structures (e.g., isolated base pairs)
- Apply pseudoknot removal for standard notation
- Validate thermodynamic stability

## Related Examples

- [RNA Sequence Design](../rna_sequence_design/) - Inverse problem (structure -> sequence)
- [Genomic Embeddings](../genomic_embeddings/) - Feature extraction
- [mRNA Degradation](../mRNA_degrad_rate_regression/) - Structure affects stability

---

**Next Steps**: Open a tutorial notebook to predict RNA secondary structures interactively!
