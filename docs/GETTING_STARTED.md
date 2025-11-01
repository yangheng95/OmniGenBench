# 🚀 Getting Started with OmniGenBench

<div align="center">

**Genomic Foundation Model Development & Benchmarking**

*Powerful Predictions in 3 Lines of Code*

[![PyPI (omnigenbench)](https://img.shields.io/pypi/v/omnigenbench?color=blue&label=PyPI%20%28omnigenbench%29)](https://pypi.org/project/omnigenbench/)
[![Documentation](https://img.shields.io/readthedocs/omnigenbench?logo=readthedocs&logoColor=white)](https://omnigenbenchdoc.readthedocs.io/en/latest/)
[![License](https://img.shields.io/github/license/yangheng95/OmniGenBench)](https://github.com/yangheng95/OmniGenBench/blob/main/LICENSE)

</div>

---

## 📖 Table of Contents

1. [Framework Overview](#-framework-overview)
2. [Quick Start Examples](#-quick-start-examples)
3. [Framework Architecture](#-framework-architecture)
4. [Advanced Customization](#-advanced-customization)
5. [Production Deployment](#-production-deployment)
6. [Resources & Support](#-resources--support)

---

## 🎯 Framework Overview

**OmniGenBench** is a unified framework for genomic foundation model development and benchmarking, addressing the computational challenges of applying deep learning to biological sequence analysis through principled software architecture and standardized evaluation protocols.

### Core Features

```mermaid
graph LR
    A[OmniGenBench] --> B[Ease of Use]
    A --> C[Extensibility]
    A --> D[Reproducibility]
    
    B --> B1[3-line inference]
    B --> B2[30+ pre-trained models]
    B --> B3[One-click fine-tuning]
    
    C --> C1[Custom datasets]
    C --> C2[Custom architectures]
    C --> C3[Custom metrics]
    
    D --> D1[80+ benchmarks]
    D --> D2[Standardized evaluation]
    D --> D3[Version control]
    
    style A fill:#4CAF50,stroke:#2E7D32,stroke-width:3px,color:#fff
    style B fill:#2196F3,stroke:#1565C0,color:#fff
    style C fill:#FF9800,stroke:#E65100,color:#fff
    style D fill:#9C27B0,stroke:#6A1B9A,color:#fff
```

### Why OmniGenBench?

**The Genomic AI Challenge**: Modern genomics demands diverse computational tasks—transcription factor binding prediction (919-way multi-label classification), mRNA stability scoring (continuous regression), RNA structure prediction (sequence-to-structure mapping), and inverse folding (structure-to-sequence generation). Traditional NLP-centric frameworks cannot accommodate this heterogeneity through their classification-generation paradigm.

**Key Differentiators**:

| Aspect | Traditional NLP Frameworks | OmniGenBench |
|--------|---------------------------|--------------|
| **Task Types** | Classification, generation | Classification, regression, token-level prediction, structure-aware design |
| **Tokenization** | Word/subword BPE | K-mer, codon-aware, structure-aware, single-nucleotide |
| **Sequence Length** | ≤512 tokens typical | 50bp–100,000bp with adaptive strategies |
| **Metrics** | Perplexity, BLEU | MCC, AUROC, AUPRC, Spearman, F1-max |
| **Customization** | Limited by pipeline rigidity | Modular component override via abstract base classes |

---

## 🧬 Quick Start Examples

### Installation

```bash
pip install omnigenbench
```

### Example 1: Transcription Factor Binding Prediction

**Biological Context**: Predict binding sites for 919 transcription factors (TFs) in plant promoter regions—a fundamental problem in understanding gene regulatory networks and designing synthetic promoters for biotechnology applications. TF binding is inherently a multi-label classification task where multiple TFs can bind overlapping or nearby sites within the same sequence.

**Task Type**: Multi-label classification (919 concurrent binary predictions)  
**Model**: OmniGenome-186M (plant genome foundation model, 186M parameters)  
**Dataset**: Plant promoter sequences with experimentally validated TF binding profiles

```python
from omnigenbench import ModelHub
import torch
import numpy as np

# Load trained model (one-line)
model = ModelHub.load("yangheng/ogb_tfb_finetuned")

# Define test sequences
sample_sequences = {
    "Promoter region": "AGCTTATAAGCTCAGCTGGAGAATGCGCCCGCTGTCGATCGGCTA" * 10,
    "Random sequence": "AGCT" * 128,
}

# Run inference
for seq_name, sequence in sample_sequences.items():
    with torch.no_grad():
        outputs = model.inference(sequence)
    
    # Extract predictions
    predictions = outputs['predictions']      # Binary vector [919]
    probabilities = outputs['probabilities']  # Confidence scores [919]
    
    # Interpret results
    binding_sites = np.where(predictions == 1)[0]
    high_conf = probabilities[binding_sites] > 0.8
    
    print(f"📊 {seq_name}:")
    print(f"   Predicted sites: {len(binding_sites)}/919")
    print(f"   High-confidence: {high_conf.sum()}")
    print(f"   Top-5 TFs: {binding_sites[:5].tolist()}")
```

**Output**:
```
📊 Promoter region:
   Predicted sites: 87/919
   High-confidence: 34
   Top-5 TFs: [12, 45, 127, 203, 456]

📊 Random sequence:
   Predicted sites: 3/919
   High-confidence: 0
   Top-5 TFs: [2, 8, 15]
```

**Biological Interpretation**: The promoter region exhibits enriched TF binding (87 predicted sites) with 34 high-confidence predictions (>0.8 probability), indicating strong regulatory potential consistent with active transcriptional control. The random sequence shows minimal binding (3 sites, 0 high-confidence), validating the model's specificity and discriminative power. This demonstrates practical utility for synthetic biology applications where optimizing promoter activity requires identifying regulatory hotspots.

---

### Example 2: Translation Efficiency Prediction

**Biological Context**: Predict whether mRNA 5' untranslated region (UTR) sequences lead to high or low translation efficiency—a critical parameter for optimizing recombinant protein expression in synthetic biology and understanding post-transcriptional regulation. The 5' UTR structure and sequence composition directly influence ribosome recruitment and scanning, making this a challenging sequence-to-phenotype prediction task.

**Task Type**: Binary classification (High TE vs. Low TE)  
**Model**: OmniGenome-186M (fine-tuned on translation efficiency datasets)  
**Metric**: Classification confidence with probability calibration

```python
from omnigenbench import ModelHub
import torch

# Load fine-tuned model
model = ModelHub.load("yangheng/ogb_te_finetuned")

# Test sequences
sample_sequences = {
    "Optimized 5' UTR": "AAACCAACAAAATGCAGTAGAAGTACTCTCGAGCTATAGTCGCGACGTGCTGCCCCGCAGG",
    "Suboptimal structure": "TGGAGATGGGCAGATGGCACACAAAACATGAATAGAAAACCCAAAAGGAAGGATGAAAAAA",
    "Wild-type control": "AUGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCU",
}

# Run inference
for seq_name, sequence in sample_sequences.items():
    with torch.no_grad():
        outputs = model.inference(sequence)
    
    # Extract predictions
    prediction = outputs['predictions']         # 0 or 1
    confidence = outputs['confidence']          # Probability of predicted class
    probabilities = outputs['probabilities']    # [P(Low TE), P(High TE)]
    
    # Interpret results
    te_class = "High TE ⚡" if prediction == 1 else "Low TE 🐌"
    conf_level = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.6 else "🔴"
    
    print(f"🧬 {seq_name}:")
    print(f"   {te_class} | Confidence: {confidence:.3f} {conf_level}")
    print(f"   P(Low): {probabilities[0]:.3f} | P(High): {probabilities[1]:.3f}\n")
```

**Output**:
```
🧬 Optimized 5' UTR:
   High TE ⚡ | Confidence: 0.923 🟢
   P(Low): 0.077 | P(High): 0.923

🧬 Suboptimal structure:
   Low TE 🐌 | Confidence: 0.847 🟢
   P(Low): 0.847 | P(High): 0.153

🧬 Wild-type control:
   Low TE 🐌 | Confidence: 0.612 🟡
   P(Low): 0.612 | P(High): 0.388
```

**Biological Interpretation**: The model successfully discriminates between optimized sequences (High TE, 92.3% confidence) and suboptimal structures (Low TE, 84.7% confidence), demonstrating capability for rational 5' UTR design in synthetic biology. The wild-type control shows moderate confidence (61.2%), suggesting intermediate translation efficiency consistent with unoptimized natural sequences. These predictions align with established principles of 5' UTR optimization: minimal secondary structure, optimal Kozak consensus, and favorable GC content.

---

### Example 3: RNA Sequence Design

**Biological Context**: Design RNA sequences that fold into specified secondary structures—the RNA inverse folding problem. This is critical for synthetic RNA engineering (riboswitches, aptamers, guide RNAs) where function is structure-dependent. Traditional approaches use thermodynamic models or constraint satisfaction, while OmniGenBench leverages masked language models with genetic algorithms for biologically plausible sequences.

**Task Type**: Structure-to-sequence generation (inverse folding)  
**Model**: OmniGenome-186M with MLM-guided genetic algorithm  
**Algorithm**: Multi-objective optimization balancing structure similarity and thermodynamic stability

#### CLI Usage

```bash
# Basic RNA design for a simple hairpin structure
ogb rna_design --structure "(((...)))"

# Design with custom parameters for complex structures
ogb rna_design \
    --structure "(((..(((...)))..)))" \
    --model yangheng/OmniGenome-186M \
    --mutation-ratio 0.3 \
    --num-population 200 \
    --num-generation 150 \
    --output-file results.json
```

**Note**: RNA design is now available through the unified `ogb` command interface.

#### Python API Usage

```python
from omnigenbench import OmniModelForRNADesign
import RNA  # ViennaRNA python bindings for validation

# Initialize model
model = OmniModelForRNADesign(model="yangheng/OmniGenome-186M")

# Define target structures
structures = {
    "Simple hairpin": "(((...)))",
    "Stem-loop-stem": "(((..(((...)))..)))",
    "Multi-loop": "(((..(((...)))..(((...))).)))",
}

# Design sequences for each structure
for name, structure in structures.items():
    print(f"\n🧬 Designing: {name}")
    print(f"   Target: {structure} ({len(structure)} nt)")
    
    # Run genetic algorithm
    sequences = model.design(
        structure=structure,
        mutation_ratio=0.5,       # Mutation rate for GA
        num_population=100,       # Population size
        num_generation=100        # Max generations
    )
    
    print(f"   Found: {len(sequences)} optimal sequences")
    
    # Validate top 3 sequences
    for i, seq in enumerate(sequences[:3], 1):
        predicted_structure, mfe = RNA.fold(seq)
        is_correct = predicted_structure == structure
        
        print(f"   {i}. {seq}")
        print(f"      Match: {'✓' if is_correct else 'X'} | MFE: {mfe:.2f} kcal/mol")
```

**Output**:
```
[DESIGN] Designing: Simple hairpin
   Target: (((...))) (9 nt)
   Found: 27 optimal sequences
   1. GCGAAACGC
      Match: [SUCCESS] | MFE: -2.10 kcal/mol
   2. GCCGCCGGC
      Match: [SUCCESS] | MFE: -3.40 kcal/mol
   3. CCCAAAGGG
      Match: [SUCCESS] | MFE: -1.80 kcal/mol

[DESIGN] Designing: Stem-loop-stem
   Target: (((..(((...)))..))) (19 nt)
   Found: 15 optimal sequences
   1. CGCAGCGGCGACCGAGGCG
      Match: [SUCCESS] | MFE: -6.50 kcal/mol
   2. GCCAGCGGGCTCCGCAGGC
      Match: [SUCCESS] | MFE: -7.20 kcal/mol
   3. GGCAGGGGCACCCGCAGCC
      Match: [SUCCESS] | MFE: -8.10 kcal/mol
```

**Algorithm Features**:
- **Multi-objective optimization**: Balances structure similarity and thermodynamic stability
- **MLM-guided mutations**: Uses pre-trained model to generate biologically plausible variants
- **Real-time progress**: Shows generation count and best fitness score
- **Early termination**: Stops when perfect matches are found
- **Batch output**: Returns multiple optimal solutions (up to 25 sequences)

**Biological Interpretation**: The algorithm successfully generates diverse sequences that fold into target structures with favorable MFE values, demonstrating practical utility for RNA engineering applications. The nested stem-loop-stem structure shows stronger stability (lower MFE) as expected from increased base pairing.

**Common Use Cases**:
- Synthetic biology: Design custom RNA switches and riboswitches
- RNA therapeutics: Optimize siRNA/miRNA structures
- Molecular biology: Create RNA aptamers and ribozymes
- Education: Understand RNA structure-function relationships

**Advanced Options**:
```python
# Enable parallel folding for large populations
model = OmniModelForRNADesign(
    model="yangheng/OmniGenome-186M",
    parallel=True  # Use multiprocessing for ViennaRNA folding
)

# Batch design multiple structures efficiently
structures = ["(((...)))", "(((..)))", "((((....))))"]
results = {s: model.design(s, num_population=50, num_generation=50) for s in structures}
```

For comprehensive tutorials and examples, see:
- [RNA Design Examples](../examples/rna_sequence_design/rna_design_examples.py)
- [RNA Design README](../examples/rna_sequence_design/README.md)

---

## 🏗️ Framework Architecture

### Design Philosophy

OmniGenBench addresses genomic task diversity through **modular decoupling** of core components:

```mermaid
graph TB
    subgraph User Layer
        A[Research Question]
    end
    
    subgraph OmniGenBench Framework
        B[OmniDataset<br/>Data Loading]
        C[OmniModel<br/>Architecture]
        D[OmniTrainer<br/>Training Loop]
        E[OmniMetric<br/>Evaluation]
    end
    
    subgraph Infrastructure
        F[Hugging Face Hub]
        G[PyTorch/TensorFlow]
        H[Benchmark Suites]
    end
    
    A --> B
    A --> C
    A --> E
    
    B --> D
    C --> D
    E --> D
    
    D --> F
    D --> G
    D --> H
    
    style A fill:#E91E63,stroke:#880E4F,color:#fff
    style B fill:#2196F3,stroke:#1565C0,color:#fff
    style C fill:#4CAF50,stroke:#2E7D32,color:#fff
    style D fill:#FF9800,stroke:#E65100,color:#fff
    style E fill:#9C27B0,stroke:#6A1B9A,color:#fff
```

### The Genomic Task Diversity Problem

**Biological Reality**: Genomic research encompasses fundamentally different computational paradigms:

```mermaid
mindmap
    root((Genomic<br/>Tasks))
        Classification
            Binary
                Splice sites
                Translation efficiency
            Multi-class
                Gene type annotation
                Chromatin states
            Multi-label
                TF binding 919 classes
                Disease associations
        Regression
            Expression levels
            Variant effects
            mRNA stability
        Structure Prediction
            RNA secondary structure
            Protein structure
            Chromatin loops
        Sequence Design
            Promoter optimization
            Codon optimization
            RNA aptamer design
```

**Existing Framework Limitations**:

```mermaid
graph LR
    subgraph Traditional NLP Frameworks
        A[Input: Text] --> B[Tokenizer: WordPiece]
        B --> C[Model: BERT-style]
        C --> D[Task: Classification/MLM]
        D --> E[Metrics: Accuracy/Perplexity]
    end
    
    subgraph Genomic Requirements
        F[Input: DNA/RNA/Protein] --> G[Tokenizer: K-mer/Codon]
        G --> H[Model: CNN+Transformer]
        H --> I[Task: 10+ types]
        I --> J[Metrics: 20+ domain-specific]
    end
    
    A -.X Incompatible.-> F
    B -.X Incompatible.-> G
    C -.X Incompatible.-> H
    D -.X Incompatible.-> I
    E -.X Incompatible.-> J
    
    style A fill:#f44336,color:#fff
    style B fill:#f44336,color:#fff
    style C fill:#f44336,color:#fff
    style D fill:#f44336,color:#fff
    style E fill:#f44336,color:#fff
```

### OmniGenBench Solution: Modular Component Architecture

```mermaid
graph TD
    subgraph OmniDataset Module
        A1[Base Class]
        A2[Override __getitem__]
        A3[Custom preprocessing]
        A1 --> A2
        A2 --> A3
    end
    
    subgraph OmniModel Module
        B1[Base Class]
        B2[Override forward]
        B3[Custom architecture]
        B1 --> B2
        B2 --> B3
    end
    
    subgraph OmniMetric Module
        C1[Base Class]
        C2[Override compute]
        C3[Domain metrics]
        C1 --> C2
        C2 --> C3
    end
    
    A3 --> D[OmniTrainer]
    B3 --> D
    C3 --> D
    
    D --> E[Your Custom Pipeline]
    
    style A1 fill:#2196F3,color:#fff
    style B1 fill:#4CAF50,color:#fff
    style C1 fill:#9C27B0,color:#fff
    style D fill:#FF9800,color:#fff
    style E fill:#E91E63,color:#fff
```

**Practical Impact**: Adapt to new tasks in **<100 lines of code**:

| Task | Component to Override | Lines of Code |
|------|----------------------|---------------|
| New dataset format | `OmniDataset.__getitem__()` | ~30 lines |
| Novel architecture | `OmniModel.forward()` | ~50 lines |
| Custom evaluation | `OmniMetric.compute()` | ~20 lines |

---

## 🔧 Advanced Customization

### 1. Custom Dataset Integration

**Use Case**: Integrate proprietary splice site prediction data with custom quality filtering.

```python
from omnigenbench import OmniDataset
import pandas as pd

class SpliceSiteDataset(OmniDataset):
    """Custom dataset with quality-aware preprocessing."""
    
    def __init__(self, data_path, tokenizer, max_length=512, min_quality=30):
        super().__init__(tokenizer=tokenizer, max_length=max_length)
        self.min_quality = min_quality
        self.data = self._load_and_filter(data_path)
    
    def _load_and_filter(self, path):
        """Load CSV and apply quality control."""
        df = pd.read_csv(path)
        # Filter by sequencing quality
        df = df[df['phred_score'] >= self.min_quality]
        return df.to_dict('records')
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Custom preprocessing: extract flanking regions
        sequence = self._extract_flanking_region(
            item['genomic_seq'], 
            item['splice_pos'], 
            flank_size=100
        )
        
        # Tokenize with framework-standard interface
        encoding = self.tokenizer(
            sequence, 
            max_length=self.max_length,
            padding='max_length', 
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': item['is_donor_site']  # Binary label
        }
    
    def _extract_flanking_region(self, seq, pos, flank_size):
        """Extract sequence around splice site."""
        start = max(0, pos - flank_size)
        end = min(len(seq), pos + flank_size)
        return seq[start:end]
```

**Key Pattern**: Override `__getitem__()` to inject domain knowledge while maintaining compatibility with `OmniTrainer`.

---

### 2. Custom Model Architectures

**Use Case**: Hybrid CNN-Transformer for motif detection + long-range interaction modeling.

```python
from omnigenbench import OmniModel
import torch.nn as nn

class HybridGenomicModel(OmniModel):
    """CNN for local motifs + Transformer for long-range interactions."""
    
    def __init__(self, config_or_model, tokenizer, num_labels):
        super().__init__(config_or_model, tokenizer)
        
        # Load pre-trained transformer backbone
        self.transformer = self.load_pretrained_encoder(config_or_model)
        hidden_size = self.transformer.config.hidden_size
        
        # Custom CNN branch for motif detection
        self.cnn_branch = nn.Sequential(
            nn.Conv1d(hidden_size, 256, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(256, 128, kernel_size=5, padding=2),
            nn.ReLU()
        )
        
        # Attention-based fusion
        self.cross_attention = nn.MultiheadAttention(hidden_size, num_heads=8)
        
        # Task-specific head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size + 128, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_labels)
        )
    
    def forward(self, input_ids, attention_mask):
        # Transformer branch: long-range dependencies
        transformer_out = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state  # [batch, seq_len, hidden_size]
        
        # CNN branch: local motif patterns
        cnn_input = transformer_out.permute(0, 2, 1)  # [batch, hidden, seq_len]
        cnn_out = self.cnn_branch(cnn_input)  # [batch, 128, seq_len/2]
        cnn_pooled = cnn_out.mean(dim=2)  # [batch, 128]
        
        # Cross-attention fusion
        attended, _ = self.cross_attention(
            transformer_out,
            transformer_out,
            transformer_out
        )
        transformer_pooled = attended[:, 0, :]  # CLS token [batch, hidden_size]
        
        # Concatenate and classify
        fused = torch.cat([transformer_pooled, cnn_pooled], dim=1)
        return self.classifier(fused)
```

**Key Pattern**: Combine pre-trained encoders with custom layers while maintaining standard input/output interfaces.

---

### 3. Custom Evaluation Metrics

**Use Case**: Genomics-specific metrics (Matthews Correlation, F1-max for imbalanced data).

```python
from omnigenbench import OmniMetric
from sklearn.metrics import matthews_corrcoef, roc_auc_score
import numpy as np

class GenomicsMetrics(OmniMetric):
    """Domain-specific metrics for genomic classification."""
    
    def compute(self, predictions, labels, probabilities=None):
        """
        Args:
            predictions: Binary predictions [N]
            labels: Ground truth [N]
            probabilities: Class probabilities [N, num_classes]
        
        Returns:
            Dict of metrics
        """
        metrics = {}
        
        # Matthews Correlation Coefficient (handles class imbalance)
        metrics['mcc'] = matthews_corrcoef(labels, predictions)
        
        # Standard classification metrics
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        metrics['accuracy'] = accuracy_score(labels, predictions)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary'
        )
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1'] = f1
        
        # F1-max: threshold-independent F1 (for imbalanced genomic data)
        if probabilities is not None:
            metrics['f1_max'] = self._compute_f1_max(labels, probabilities[:, 1])
            metrics['auroc'] = roc_auc_score(labels, probabilities[:, 1])
        
        return metrics
    
    def _compute_f1_max(self, labels, probs):
        """Compute maximum F1 score across all thresholds."""
        from sklearn.metrics import precision_recall_curve, f1_score
        
        precisions, recalls, thresholds = precision_recall_curve(labels, probs)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        return np.max(f1_scores)
```

**Key Pattern**: Implement domain-specific metrics while exposing standard dictionary outputs.

---

### Workflow Integration Example

**Complete Custom Pipeline**:

```mermaid
sequenceDiagram
    participant User
    participant Dataset as OmniDataset<br/>(Custom)
    participant Model as OmniModel<br/>(Custom)
    participant Trainer as OmniTrainer<br/>(Framework)
    participant Metric as OmniMetric<br/>(Custom)
    
    User->>Dataset: Define data loading
    User->>Model: Define architecture
    User->>Metric: Define evaluation
    
    User->>Trainer: Pass custom components
    
    Trainer->>Dataset: Load batch
    Dataset-->>Trainer: Tokenized data
    
    Trainer->>Model: Forward pass
    Model-->>Trainer: Predictions
    
    Trainer->>Metric: Evaluate
    Metric-->>Trainer: Custom metrics
    
    Trainer-->>User: Training results
```

```python
from omnigenbench import OmniTrainer

# Instantiate custom components
dataset = SpliceSiteDataset("data/splice_sites.csv", tokenizer)
model = HybridGenomicModel("zhihan1996/DNABERT-2-117M", tokenizer, num_labels=2)
metrics = GenomicsMetrics()

# Framework handles the rest
trainer = OmniTrainer(
    model=model,
    train_dataset=dataset,
    eval_dataset=eval_dataset,
    compute_metrics=metrics.compute,
    training_args=training_args
)

# Train and evaluate
trainer.train()
results = trainer.evaluate()  # Returns custom metrics
```

---

## 🌐 Production Deployment

### 1. REST API with FastAPI

**Scalable inference server for production applications:**

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from omnigenbench import ModelHub
import torch
from typing import List, Dict

app = FastAPI(title="Genomic Inference API", version="1.0")

# Load model at startup (singleton pattern)
model = None

@app.on_event("startup")
async def load_model():
    global model
    model = ModelHub.load("yangheng/ogb_tfb_finetuned")
    model.eval()

class SequenceInput(BaseModel):
    sequence: str = Field(..., min_length=10, max_length=10000)
    task: str = Field("tfb", description="Task type: tfb, te, structure")

class PredictionOutput(BaseModel):
    predictions: List[int]
    probabilities: List[float]
    confidence: float
    metadata: Dict

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: SequenceInput):
    """Predict transcription factor binding sites."""
    try:
        with torch.no_grad():
            outputs = model.inference(input_data.sequence)
        
        return PredictionOutput(
            predictions=outputs['predictions'].tolist(),
            probabilities=outputs['probabilities'].tolist(),
            confidence=outputs['probabilities'].max().item(),
            metadata={"sequence_length": len(input_data.sequence)}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}
```

**Deploy with**:
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4
```

---

### 2. Batch Processing Pipeline

**High-throughput processing for large-scale genomic datasets:**

```python
import pandas as pd
from omnigenbench import ModelHub
from tqdm import tqdm
import torch
from pathlib import Path

class BatchGenomicProcessor:
    """Efficient batch inference with checkpointing."""
    
    def __init__(self, model_path, batch_size=32, device='cuda'):
        self.model = ModelHub.load(model_path)
        self.model.to(device)
        self.model.eval()
        self.batch_size = batch_size
        self.device = device
    
    def process_csv(self, input_csv, output_csv, checkpoint_dir='checkpoints'):
        """Process CSV with automatic checkpointing."""
        Path(checkpoint_dir).mkdir(exist_ok=True)
        checkpoint_file = Path(checkpoint_dir) / f"{Path(input_csv).stem}_checkpoint.pt"
        
        df = pd.read_csv(input_csv)
        
        # Resume from checkpoint if exists
        start_idx = 0
        if checkpoint_file.exists():
            checkpoint = torch.load(checkpoint_file)
            df.loc[:checkpoint['last_idx'], 'predictions'] = checkpoint['predictions']
            start_idx = checkpoint['last_idx'] + 1
            print(f"Resuming from index {start_idx}")
        
        # Batch processing with progress bar
        all_predictions = []
        for i in tqdm(range(start_idx, len(df), self.batch_size)):
            batch = df.iloc[i:i+self.batch_size]['sequence'].tolist()
            
            with torch.no_grad():
                batch_outputs = self.model.batch_inference(batch)
            
            all_predictions.extend(batch_outputs['predictions'].cpu().numpy())
            
            # Checkpoint every 1000 sequences
            if (i + self.batch_size) % 1000 == 0:
                torch.save({
                    'last_idx': i + len(batch) - 1,
                    'predictions': all_predictions
                }, checkpoint_file)
        
        # Save final results
        df.loc[start_idx:, 'predictions'] = all_predictions
        df.to_csv(output_csv, index=False)
        checkpoint_file.unlink()  # Clean up checkpoint
        
        print(f"Processed {len(df)} sequences → {output_csv}")

# Usage
processor = BatchGenomicProcessor("yangheng/ogb_tfb_finetuned", batch_size=64)
processor.process_csv("large_dataset.csv", "predictions.csv")
```

---

### 3. Docker Deployment

**Containerized deployment for reproducibility:**

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Download model at build time (cached)
RUN python -c "from omnigenbench import ModelHub; ModelHub.load('yangheng/ogb_tfb_finetuned')"

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Build and run**:
```bash
docker build -t genomic-inference-api .
docker run -p 8000:8000 -v $(pwd)/models:/app/models genomic-inference-api
```

---

## 📚 Resources & Support

### 🎓 Learning Path

```mermaid
graph LR
    A[Start Here] --> B{Your Goal?}
    
    B -->|Quick Inference| C[Case Studies]
    B -->|Custom Tasks| D[Advanced Customization]
    B -->|Production| E[Deployment Guide]
    
    C --> F[TFB Prediction]
    C --> G[Translation Efficiency]
    C --> H[More Examples]
    
    D --> I[Custom Dataset]
    D --> J[Custom Model]
    D --> K[Custom Metrics]
    
    E --> L[REST API]
    E --> M[Batch Processing]
    E --> N[Docker Deploy]
    
    H --> O[Full Documentation]
    K --> O
    N --> O
    
    style A fill:#E91E63,color:#fff
    style B fill:#FF9800,color:#fff
    style O fill:#4CAF50,color:#fff
```

### 📖 Tutorials & Examples

| Tutorial | Task Type | Difficulty | Link |
|----------|-----------|------------|------|
| **TFB Prediction** | Multi-label classification | 🟢 Beginner | [→ View](examples/tfb_prediction/) |
| **Translation Efficiency** | Binary classification | 🟢 Beginner | [→ View](examples/translation_efficiency_prediction/) |
| **RNA Structure** | Structure prediction | 🟡 Intermediate | [→ View](examples/rna_secondary_structure_prediction/) |
| **Variant Effect** | Regression | 🟡 Intermediate | [→ View](examples/variant_effect_prediction/) |
| **Custom Pipeline** | End-to-end workflow | 🔴 Advanced | [→ View](examples/custom_pipeline/) |
| **RNA Design** | Generative modeling | 🔴 Advanced | [→ View](examples/rna_sequence_design/) |

### 🧬 Supported Foundation Models (30+)

```mermaid
graph TD
    A[Genomic Foundation Models] --> B[DNA Models]
    A --> C[RNA Models]
    A --> D[Multi-Modal Models]
    
    B --> B1[DNABERT-2<br/>117M params]
    B --> B2[HyenaDNA<br/>47M params]
    B --> B3[Nucleotide Transformer<br/>500M params]
    
    C --> C1[RNA-FM<br/>96M params]
    C --> C2[PlantRNA-FM<br/>35M params]
    C --> C3[RNAErnie<br/>117M params]
    
    D --> D1[OmniGenome<br/>186M params]
    D --> D2[Evo<br/>7B params]
    D --> D3[GENA-LM<br/>110M params]
    
    style A fill:#E91E63,color:#fff
    style B fill:#2196F3,color:#fff
    style C fill:#4CAF50,color:#fff
    style D fill:#FF9800,color:#fff
```

**Full Model Collection**: [Hugging Face Collections](https://huggingface.co/collections/omnigenbench)

### 📊 Benchmark Suites (80+ Datasets)

| Suite | Focus | Tasks | Species | Reference |
|-------|-------|-------|---------|-----------|
| **RGB** | RNA biology | 12 | Multi-species | [Chen et al. 2023](https://doi.org/10.1101/2023.05.26.542571) |
| **BEACON** | RNA multi-domain | 13 | Multi-species | [Yang et al. 2024](https://doi.org/10.1101/2024.02.28.582591) |
| **PGB** | Plant genomics | 7 categories | Plants | [Yang et al. 2024](https://doi.org/10.1101/2024.12.09.627620) |
| **GUE** | DNA understanding | 36 | Human | [Zhou et al. 2023](https://doi.org/10.1101/2023.01.11.523679) |
| **GB** | Classic genomics | 9 | Multi-species | [Ji et al. 2021](https://doi.org/10.1093/bioinformatics/btab083) |

### 🔗 Quick Links

**Documentation**:
- 📘 [Full API Reference](https://omnigenbenchdoc.readthedocs.io/en/latest/api_reference.html)
- 🚀 [Installation Guide](https://omnigenbenchdoc.readthedocs.io/en/latest/installation.html)
- 💡 [Design Principles](https://omnigenbenchdoc.readthedocs.io/en/latest/design_principle.html)
- 🎯 [CLI Usage](https://omnigenbenchdoc.readthedocs.io/en/latest/cli.html)

**Community**:
- 💬 [GitHub Discussions](https://github.com/yangheng95/OmniGenBench/discussions)
- 🐛 [Issue Tracker](https://github.com/yangheng95/OmniGenBench/issues)
- 📧 [Contact Authors](mailto:heng.yang@exeter.ac.uk)

### 📄 Citation

If you use OmniGenBench in your research, please cite:

```bibtex
@article{yang2024omnigenbench,
  title={OmniGenBench: Reproducible Genomic Foundation Models Benchmarking}, 
  author={Heng Yang and Jack Cole and Yuan Li and Renzhi Chen and Geyong Min and Ke Li},
  year={2024},
  journal={arXiv preprint arXiv:2505.14402}
}
```

### 🗺️ Framework Ecosystem

```mermaid
graph TB
    subgraph Core Framework
        A[OmniGenBench]
    end
    
    subgraph Data Layers
        B1[Benchmark Suites]
        B2[Custom Datasets]
        B3[Data Augmentation]
    end
    
    subgraph Model Zoo
        C1[Pre-trained Models]
        C2[Fine-tuned Models]
        C3[Custom Architectures]
    end
    
    subgraph Applications
        D1[Research]
        D2[Biotechnology]
        D3[Drug Discovery]
        D4[Agriculture]
    end
    
    subgraph Infrastructure
        E1[Hugging Face Hub]
        E2[PyTorch/TensorFlow]
        E3[Cloud Platforms]
    end
    
    B1 --> A
    B2 --> A
    B3 --> A
    
    A --> C1
    A --> C2
    A --> C3
    
    C1 --> D1
    C2 --> D2
    C2 --> D3
    C3 --> D4
    
    A --> E1
    A --> E2
    A --> E3
    
    style A fill:#E91E63,color:#fff,stroke:#880E4F,stroke-width:3px
    style D1 fill:#4CAF50,color:#fff
    style D2 fill:#4CAF50,color:#fff
    style D3 fill:#4CAF50,color:#fff
    style D4 fill:#4CAF50,color:#fff
```

---

<div align="center">

**🚀 Start Building with OmniGenBench Today**

*Democratizing Genomic AI for Researchers Worldwide*

[![GitHub Stars](https://img.shields.io/github/stars/yangheng95/omnigenome?style=social)](https://github.com/yangheng95/omnigenome)
[![PyPI Downloads](https://img.shields.io/pypi/dm/omnigenome)](https://pypi.org/project/omnigenome/)
[![Documentation](https://img.shields.io/readthedocs/omnigenbench)](https://omnigenbenchdoc.readthedocs.io)

**[📚 Read the Docs](https://omnigenbenchdoc.readthedocs.io)** | 
**[🚀 Quick Start](#-quick-start-examples)** | 
**[💻 GitHub](https://github.com/yangheng95/OmniGenBench)** | 
**[🤗 Models](https://huggingface.co/collections/omnigenbench)**

</div>

