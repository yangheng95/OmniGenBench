.. _design_principle:

###########################
Architecture & Design Philosophy
###########################

OmniGenBench implements a principled software architecture for genomic foundation model development, grounded in three foundational tenets: **abstraction**, **modularity**, and **composability**. This design philosophy enables researchers to construct, extend, and integrate genomic machine learning pipelines with minimal technical friction while maintaining scientific rigor and computational reproducibility.

**Core Architectural Principle**: The framework is architected around four abstract base classes (``OmniModel``, ``OmniDataset``, ``OmniTokenizer``, ``OmniMetric``) that serve as formal interface contracts, ensuring all components expose consistent APIs while remaining extensible through inheritance-based customization.

**Design Goals**

The framework addresses four fundamental challenges that distinguish computational genomics from general-purpose deep learning:

1. **Heterogeneous Task Types**: Unlike text-centric NLP frameworks where classification and generation dominate, genomics requires diverse prediction modalities—multi-label classification (transcription factor binding with 919 concurrent labels), continuous regression (gene expression quantification), token-level classification (splice site detection), and structure-aware sequence generation (RNA inverse folding). Each task demands specialized loss functions, output formats, and evaluation protocols that cannot be unified under a single model interface.

2. **Tokenization Diversity**: Biological sequences demand specialized encoding strategies beyond standard subword tokenization:
   
   - **K-mer encoding**: Capture local sequence motifs with overlapping windows (vocabulary size 4^k for k-mers)
   - **Single-nucleotide encoding**: Character-level representation with vocabulary ~10 (A, C, G, T/U, N, special tokens)
   - **BPE (Byte-Pair Encoding)**: Learn subword units from corpus statistics for discovering recurrent patterns
   - **Codon-aware schemes**: Respect triplet boundaries in protein-coding regions to preserve translational reading frames
   
   No single tokenization strategy optimizes all genomic tasks—protein-coding regions benefit from codon-awareness, while regulatory regions require finer-grained nucleotide resolution.

3. **Domain-Specific Evaluation**: Standard NLP metrics (perplexity, BLEU score) fail to capture biological relevance. Genomics requires specialized measures aligned with domain practices:
   
   - **Matthews Correlation Coefficient (MCC)**: Balanced performance metric for highly imbalanced datasets (e.g., rare variant prediction where negatives outnumber positives 1000:1)
   - **Area Under Precision-Recall Curve (AUPRC)**: Prioritizes ranking quality for rare event detection with severe class imbalance
   - **Spearman/Pearson Correlation**: Monotonic and linear relationship measures for continuous biological phenomena (gene expression, degradation rates)
   - **F1-max**: Optimal threshold selection for binary classification, accounting for varying decision boundaries across tasks
   
   These metrics provide biologically meaningful evaluation that aligns with experimental validation practices.

4. **Reproducibility at Scale**: With 80+ benchmark tasks spanning DNA, RNA, and protein sequences across multiple species, systematic evaluation demands:
   
   - **Deterministic workflows**: Fixed random seeds with explicit control over all stochastic components (initialization, shuffling, dropout)
   - **Version control**: Explicit tracking of model checkpoints, dataset versions, and framework dependencies
   - **Statistical rigor**: Multi-seed averaging (typically 3-5 independent runs) with standard deviations and confidence intervals
   - **Significance testing**: Paired statistical tests (Wilcoxon signed-rank, paired t-test) for principled model comparison

This document explores the architectural patterns, abstract interface contracts, and extension mechanisms that balance power for advanced users with accessibility for practitioners.

***********************************
Architectural Foundation
***********************************

**Abstract Base Classes as Formal Contracts**

OmniGenBench employs the *Abstract Base Class (ABC) pattern* from software engineering to establish formal interface contracts for all major components. Each ABC defines:

- **Interface specification**: Required methods with explicit type signatures and semantic contracts
- **Behavioral invariants**: Expected input/output semantics, side effects, and exception handling
- **Extension points**: Protected methods and hooks for domain-specific customization through inheritance

This architectural choice yields four critical properties that distinguish OmniGenBench from ad-hoc genomic ML frameworks:

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: 🎯 Type Safety & Interface Consistency
      :shadow: md

      All components of a given type (models, datasets, metrics) expose identical interfaces with consistent method signatures, enabling static type checking and eliminating entire classes of runtime errors. This predictability reduces cognitive load during development and accelerates debugging through contract verification.

   .. grid-item-card:: 🔌 Plug-and-Play Extensibility
      :shadow: md

      Custom implementations inherit full framework integration by subclassing ABCs and implementing required abstract methods. Extension requires no modification to core framework code—follow the Open-Closed Principle where components are open for extension but closed for modification.

   .. grid-item-card:: 🧩 Compositional Flexibility
      :shadow: md

      Components adhering to ABC contracts are inherently interoperable through duck typing and polymorphism. Mix pre-built modules with custom implementations seamlessly—swap tokenizers, combine metrics, or pipeline datasets without coupling concerns or brittle dependencies.

   .. grid-item-card:: 📐 Architectural Clarity & Self-Documentation
      :shadow: md

      Clear separation of concerns through ABCs makes the codebase self-documenting with explicit component responsibilities. New contributors immediately understand the architectural boundaries and extension points, reducing onboarding time and maintenance burden.


***********************************
The Four Pillars: Core Abstractions
***********************************

OmniGenBench is architected around four fundamental abstract base classes, each addressing a distinct concern in the genomic machine learning pipeline. Mastering these abstractions is essential for both using the framework effectively and extending it with custom components.

**1. OmniModel: Unified Model Interface**

.. card:: OmniModel: Task-Polymorphic Model Interface
   :shadow: md

   The ``OmniModel`` abstract class serves as the foundation for all genomic foundation models, providing a task-agnostic interface that unifies diverse prediction paradigms while allowing task-specific specialization. Through composition with ``EmbeddingMixin``, it seamlessly integrates representation learning capabilities for transfer learning and interpretability.

   **Architectural Principles:**

   *   **Task-Specific Subclassing**: Rather than monolithic model classes, OmniGenBench provides task-specific model types (``OmniModelForSequenceClassification``, ``OmniModelForTokenClassification``, ``OmniModelForSequenceRegression``, etc.) that inherit common functionality while specializing loss computation, output formatting, and evaluation protocols.
   
   *   **Automatic Architecture Resolution**: Models automatically detect underlying transformer architectures from HuggingFace ``config.json`` metadata via ``auto_map`` or ``architectures`` fields, eliminating manual architecture specification and enabling seamless model switching.
   
   *   **Representation Learning Integration**: All models inherit ``encode()`` and ``extract_attention_scores()`` methods from ``EmbeddingMixin``, enabling zero-code extraction of sequence embeddings and attention patterns for downstream interpretability, clustering, and transfer learning applications.
   **Core Methods:**

   *   ``__init__(config_or_model, tokenizer, **kwargs)`` - Polymorphic initialization from HuggingFace configs, model paths, or PyTorch modules
   *   ``forward(**inputs)`` - Standard forward pass with automatic task-appropriate loss computation when labels provided
   *   ``predict(sequence)`` / ``inference(sequence)`` - High-level prediction interfaces with automatic tokenization and post-processing
   *   ``save_model(path)`` / ``load_model(path)`` - Model persistence with automatic tokenizer bundling for deployment
   *   ``encode(sequences)`` - Extract fixed-length embeddings for downstream tasks *(inherited from EmbeddingMixin)*
   *   ``extract_attention_scores(sequence)`` - Retrieve layer-wise attention matrices for interpretability *(inherited from EmbeddingMixin)*

   **Design Pattern:**

   .. code-block:: python

      from omnigenbench import OmniModelForSequenceClassification, OmniTokenizer

      # Task-specific instantiation with automatic architecture detection
      tokenizer = OmniTokenizer.from_pretrained("yangheng/OmniGenome-186M")
      model = OmniModelForSequenceClassification(
          "yangheng/OmniGenome-186M",
          tokenizer=tokenizer,
          num_labels=2,
          problem_type="single_label_classification"
      )
      
      # Training: forward pass with automatic loss computation
      outputs = model(input_ids=..., attention_mask=..., labels=...)
      loss = outputs['loss']  # Task-appropriate loss function automatically applied
      
      # Inference: high-level prediction interface with structured output
      result = model.inference("ATCGATCGATCGATCG")
      print(result)  # {'predictions': array([1]), 'probabilities': array([0.08, 0.92])}
      
      # Representation learning: extract embeddings for clustering or downstream tasks
      embeddings = model.encode(["ATCG", "GCTA"])  # Shape: (2, hidden_dim)

   .. tip::
      **When to Use Which Interface:**
      
      - ``forward()``: During training loops when gradients are required for backpropagation
      - ``predict()``: For batch inference with internal tokenization and numpy output  
      - ``inference()``: For single-sequence predictions with human-readable structured output
      - ``encode()``: For transfer learning, embedding extraction, and downstream feature engineering
..       .. code-block:: python

..          from omnigenbench import ClassificationMetric

..          metric = ClassificationMetric(ignore_y=-100)
..          y_true = [0, 1, -100, 1]
..          y_pred = [0, 1, 0, 0]
..          results = metric.compute_metric(y_true, y_pred)
..          print(results) # {'accuracy_score': 0.66, ...}


OmniGenBench is built around four fundamental abstract classes. Understanding these is key to mastering the library.

.. card:: OmniModel: Unified Model Interface
   :shadow: md

   The ``OmniModel`` abstract class serves as the foundation for all genomic foundation models, providing a task-agnostic interface that unifies diverse prediction paradigms. Through composition with ``EmbeddingMixin``, it seamlessly integrates representation learning capabilities.

   **Architectural Principles:**

   *   **Task-Specific Subclassing**: Rather than monolithic model classes, OmniGenBench provides task-specific model types (``OmniModelForSequenceClassification``, ``OmniModelForTokenClassification``, etc.) that inherit common functionality while specializing loss computation and output formatting.
   
   *   **Automatic Architecture Resolution**: Models auto-detect underlying architectures from HuggingFace ``config.json`` via ``auto_map`` or ``architectures`` fields, eliminating manual architecture specification.
   
   *   **Representation Learning Integration**: All models inherit ``encode()`` and ``extract_attention_scores()`` from ``EmbeddingMixin``, enabling seamless extraction of sequence embeddings and attention patterns for interpretability and transfer learning.

   **Core Methods:**

   *   ``__init__(config_or_model, tokenizer, **kwargs)`` - Flexible initialization from configs, paths, or PyTorch modules
   *   ``forward(**inputs)`` - Standard forward pass with task-specific loss computation
   *   ``predict(sequence)`` / ``inference(sequence)`` - High-level prediction interfaces
   *   ``save_model(path)`` / ``load_model(path)`` - Model persistence with tokenizer bundling
   *   ``encode(sequences)`` - Extract fixed-length embeddings for downstream tasks *(from EmbeddingMixin)*
   *   ``extract_attention_scores(sequence)`` - Retrieve layer-wise attention matrices *(from EmbeddingMixin)*

   **Design Pattern:**

   .. code-block:: python

      from omnigenbench import OmniModelForSequenceClassification, OmniTokenizer

      # Task-specific instantiation with automatic architecture detection
      tokenizer = OmniTokenizer.from_pretrained("yangheng/OmniGenome-186M")
      model = OmniModelForSequenceClassification(
          "yangheng/OmniGenome-186M",
          tokenizer=tokenizer,
          num_labels=2,
          problem_type="single_label_classification"
      )
      
      # Training: forward pass with automatic loss computation
      outputs = model(input_ids=..., attention_mask=..., labels=...)
      loss = outputs['loss']  # Task-appropriate loss function applied
      
      # Inference: high-level prediction interface
      result = model.inference("ATCGATCGATCGATCG")
      print(result)  # {'predictions': array([1]), 'probabilities': array([0.08, 0.92])}
      
      # Representation learning: extract embeddings (inherited from EmbeddingMixin)
      embeddings = model.encode(["ATCG", "GCTA"])  # Shape: (2, hidden_dim)

   .. tip::
      **When to Use Which Interface:**
      
      - ``forward()``: During training loops when you need gradients and loss
      - ``predict()``: For batch inference with tokenization handled internally  
      - ``inference()``: For single-sequence predictions with formatted output
      - ``encode()``: For transfer learning and downstream feature extraction

**2. OmniDataset: Polymorphic Data Handling**

.. card:: OmniDataset: Format-Agnostic Data Loading
   :shadow: md

   The ``OmniDataset`` abstract class provides a polymorphic interface for genomic data ingestion, abstracting away format-specific parsing logic while maintaining PyTorch DataLoader compatibility through the standard ``__getitem__`` and ``__len__`` protocols.

   **Architectural Principles:**

   *   **Format Agnosticism**: Unified API supports JSON, CSV, Parquet, FASTA, FASTQ, BED, and NumPy formats through pluggable format-specific parsers, enabling seamless dataset migration without code changes.
   
   *   **Integrated Tokenization Pipeline**: Tokenization occurs within the dataset ``__getitem__`` method, ensuring consistency across training and evaluation while enabling efficient caching of encoded sequences to minimize redundant computation.
   
   *   **Lazy Loading & Memory Efficiency**: Large genomic datasets (>1M sequences) are loaded incrementally with on-the-fly tokenization, minimizing memory footprint and enabling training on datasets larger than available RAM.
   
   *   **Automatic Label Management**: Bidirectional mapping between string labels and integer indices with support for multi-label scenarios, ignored indices (PyTorch's -100 convention), and label smoothing.

   **Core Methods:**

   *   ``__init__(data_path, tokenizer, max_length, **kwargs)`` - Initialize with data source and tokenization config
   *   ``__getitem__(index)`` - Retrieve tokenized sample as PyTorch tensors (``input_ids``, ``attention_mask``, ``labels``)
   *   ``__len__()`` - Dataset size for sampler configuration and progress tracking
   *   ``get_labels()`` - Retrieve unique label set for vocabulary construction
   *   ``get_label_mapping()`` - Obtain label-to-index dictionary for inverse mapping

   **Design Pattern:**

   .. code-block:: python

      from omnigenbench import OmniDatasetForSequenceClassification, OmniTokenizer

      # Polymorphic initialization - format auto-detected from extension
      tokenizer = OmniTokenizer.from_pretrained("yangheng/OmniGenome-186M")
      dataset = OmniDatasetForSequenceClassification(
          dataset_name_or_path="promoters.json",  # or .csv, .fasta, .parquet
          tokenizer=tokenizer,
          max_length=512,
          label2id={"negative": 0, "positive": 1}  # or auto-generated
      )
      
      # PyTorch DataLoader integration
      from torch.utils.data import DataLoader
      loader = DataLoader(dataset, batch_size=32, shuffle=True)
      
      # Inspect label mapping
      print(dataset.label2id)  # {'negative': 0, 'positive': 1}
      print(dataset.id2label)  # {0: 'negative', 1: 'positive'}
      
      # Access tokenized samples
      sample = dataset[0]
      print(sample.keys())  # dict_keys(['input_ids', 'attention_mask', 'labels'])

   .. important::
      **Data Format Convention**: All datasets expect at minimum a ``sequence`` field (or aliases like ``seq``, ``text``, ``dna``, ``rna``) and optionally a ``label`` field (or ``labels``, ``target``, ``y``). Field names are auto-standardized internally during loading.

   .. tip::
      **Custom Data Formats**: Extend by subclassing and overriding format-specific loading methods while inheriting tokenization and label management logic. See ``OmniDataset._load_json()``, ``_load_csv()``, ``_load_fasta()`` for examples.
**3. OmniTokenizer: Flexible Sequence Encoding**

.. card:: OmniTokenizer: Flexible Sequence Encoding
   :shadow: md

   The ``OmniTokenizer`` abstract class provides a unified interface for diverse biological sequence tokenization strategies, from character-level encodings to learned subword vocabularies.

   **Architectural Principles:**

   *   **Strategy Pattern**: Different tokenization algorithms (k-mer, BPE, single-nucleotide) implement the same interface, enabling runtime swapping without code changes.
   
   *   **Preprocessing Pipeline**: Built-in preprocessing hooks (RNA-to-DNA conversion, case normalization, special token insertion) ensure consistent sequence representation.
   
   *   **HuggingFace Compatibility**: Wraps HuggingFace tokenizers when available, providing backward compatibility while adding genomic-specific functionality.

   **Core Methods:**

   *   ``__init__(base_tokenizer, **kwargs)`` - Wrap existing tokenizer or initialize custom strategy
   *   ``tokenize(sequence, **kwargs)`` - Convert sequence to token list
   *   ``encode(sequence, **kwargs)`` - Tokenize and convert to integer indices with padding/truncation
   *   ``decode(token_ids, **kwargs)`` - Reverse tokenization to recover original sequence
   *   ``from_pretrained(model_name)`` - Load tokenizer matching pre-trained model

   **Design Pattern:**

   .. code-block:: python

      from omnigenbench import OmniTokenizer, OmniSingleNucleotideTokenizer, OmniKmersTokenizer

      # Pattern 1: Load tokenizer matching pre-trained model
      tokenizer = OmniTokenizer.from_pretrained("yangheng/OmniGenome-186M")
      
      # Pattern 2: Initialize with specific tokenization strategy
      single_nt_tokenizer = OmniSingleNucleotideTokenizer.from_pretrained(
          "yangheng/OmniGenome-186M"
      )
      
      kmer_tokenizer = OmniKmersTokenizer(
          kmer=6,  # 6-mer tokenization
          max_length=512
      )
      
      # Encode with automatic preprocessing
      inputs = tokenizer(
          "AUGCUAGC",  # RNA sequence with U
          max_length=128,
          padding="max_length",
          truncation=True,
          return_tensors="pt"
      )
      # Note: Automatic U-to-T conversion if tokenizer.u2t=True
      
      # Decode back to sequence
      decoded = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
      print(decoded)  # Original sequence (may have U→T conversion)

   .. tip::
      **Choosing Tokenization Strategies:**
      
      - **Single-nucleotide**: Fine-grained position-level tasks (splice sites, methylation)
      - **K-mer (k=3,6)**: Balance between resolution and context (general classification)
      - **BPE/Learned**: Transfer learning from large corpora (foundation models)
**4. OmniMetric: Domain-Specific Evaluation**

.. card:: OmniMetric: Domain-Specific Evaluation
   :shadow: md

   The ``OmniMetric`` abstract class standardizes evaluation protocols for genomic machine learning, providing a consistent interface to domain-specific performance measures.

   **Architectural Principles:**

   *   **Metric Composition**: Complex evaluation workflows combine multiple metrics through a unified interface, enabling comprehensive model assessment.
   
   *   **Ignored Label Handling**: Proper treatment of masked positions (PyTorch's -100 convention) prevents contamination of evaluation metrics.
   
   *   **Statistical Rigor**: Built-in support for confidence intervals, significance testing, and multi-seed aggregation for reproducible benchmarking.

   **Core Methods:**

   *   ``__init__(ignore_y=None, **kwargs)`` - Configure with ignored label values
   *   ``compute_metric(y_true, y_pred, **kwargs)`` - Calculate metric from predictions and targets
   *   ``get_metric_name()`` - Retrieve canonical metric identifier

   **Design Pattern:**

   .. code-block:: python

      from omnigenbench import ClassificationMetric, RegressionMetric
      import numpy as np

      # Classification: comprehensive evaluation suite with ignored labels
      clf_metric = ClassificationMetric(ignore_y=-100)
      
      y_true = np.array([0, 1, -100, 1, 0])  # -100 = masked/ignored position
      y_pred = np.array([0, 1, 0, 1, 1])
      
      results = clf_metric.compute(y_true, y_pred)
      # Returns dict with multiple metrics:
      # {
      #   'accuracy': 0.75,
      #   'precision': 0.67, 
      #   'recall': 1.0,
      #   'f1_score': 0.80,
      #   'matthews_corrcoef': 0.58  # Matthews Correlation Coefficient for imbalanced data
      # }
      
      # Regression: domain-appropriate metrics for continuous predictions
      reg_metric = RegressionMetric()
      y_true_reg = np.array([1.2, 3.4, 5.6])
      y_pred_reg = np.array([1.3, 3.2, 5.8])
      results = reg_metric.compute(y_true_reg, y_pred_reg)
      # Returns:
      # {
      #   'mse': 0.03,
      #   'mae': 0.15,
      #   'r2_score': 0.98,
      #   'pearson_correlation': 0.99,
      #   'spearman_correlation': 1.0  # Rank-order correlation
      # } 
                                           y_pred=[1.1, 3.5, 5.4])
      # Returns: {'mse': 0.03, 'mae': 0.13, 'r2': 0.98, 'spearman': 1.0}

   .. important::
      **Genomics-Specific Metrics**: Unlike NLP (perplexity, BLEU), genomics prioritizes:
      
      - **MCC**: Handles class imbalance better than accuracy
      - **AUPRC**: More informative than AUROC for rare positives  
      - **Spearman ρ**: Captures monotonic relationships in expression data


***********************************
Extension Patterns & Best Practices
***********************************

**The Open-Closed Principle in Practice**

OmniGenBench embraces the *Open-Closed Principle*: the framework is open for extension but closed for modification. You add functionality through inheritance and composition, not by editing core code. This ensures your customizations remain compatible with framework updates.

**Extension Pattern 1: Custom Model Architectures**

.. card:: Extension Pattern 1: Custom Model Architectures
   :shadow: md

   **Use Case**: Integrate novel architectures (Mamba, Hyena, custom CNNs) while inheriting AutoBench/AutoTrain compatibility.

   **Implementation Strategy**: Subclass task-specific model types and override ``forward()`` to inject custom layers or attention mechanisms.

   .. code-block:: python

      from omnigenbench import OmniModelForSequenceClassification
      import torch.nn as nn

      class HybridCNNTransformer(OmniModelForSequenceClassification):
          """Custom architecture combining CNN feature extraction with transformer."""
          
          def __init__(self, config, tokenizer, **kwargs):
              super().__init__(config, tokenizer, **kwargs)
              
              # Add custom layers while preserving base architecture
              self.conv_layers = nn.Sequential(
                  nn.Conv1d(config.hidden_size, 256, kernel_size=7, padding=3),
                  nn.ReLU(),
                  nn.MaxPool1d(2)
              )
              
              # Custom classifier head
              self.custom_classifier = nn.Linear(256, self.config.num_labels)
          
          def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
              # Base transformer encoding
              outputs = self.base_model(
                  input_ids=input_ids,
                  attention_mask=attention_mask,
                  output_hidden_states=True
              )
              
              # Custom processing pipeline
              hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden)
              conv_features = self.conv_layers(hidden_states.transpose(1, 2))
              pooled = conv_features.mean(dim=2)  # Global average pooling
              logits = self.custom_classifier(pooled)
              
              # Leverage inherited loss computation
              loss = None
              if labels is not None:
                  loss = self.compute_loss(logits, labels)
              
              return type('Output', (), {
                  'loss': loss,
                  'logits': logits,
                  'hidden_states': outputs.hidden_states
              })()

   **Key Principles:**
   
   - Preserve ``forward()`` signature for trainer compatibility
   - Use ``self.base_model`` to access pre-trained weights
   - Call ``self.compute_loss()`` for task-appropriate loss functions
   - Return outputs with ``loss`` and ``logits`` attributes

**Extension Pattern 2: Custom Data Loaders**

.. card:: Extension Pattern 2: Custom Data Formats
   :shadow: md

   **Use Case**: Integrate proprietary file formats (BigWig, custom HDF5, database connectors) into training pipelines.

   **Implementation Strategy**: Subclass task-specific dataset types and override ``_load_data()`` for custom parsing logic.

   .. code-block:: python

      from omnigenbench import OmniDatasetForSequenceClassification
      import h5py

      class HDF5GenomicDataset(OmniDatasetForSequenceClassification):
          """Load genomic data from HDF5 archives."""
          
          def _load_data(self, data_path):
              """Override to implement HDF5 parsing."""
              samples = []
              
              with h5py.File(data_path, 'r') as f:
                  sequences = f['sequences'][:]  # Numpy array
                  labels = f['labels'][:]
                  
                  for seq, label in zip(sequences, labels):
                      samples.append({
                          'sequence': seq.decode('utf-8'),  # Bytes to string
                          'label': int(label)
                      })
              
              return samples  # Return standardized format
          
          def __getitem__(self, idx):
              # Inherit tokenization from parent class
              return super().__getitem__(idx)

   **Key Principles:**
   
   - Return data in standardized ``{sequence, label}`` format
   - Inherit tokenization and batching from parent class
   - Use ``self.tokenizer`` for consistent encoding
   - Leverage ``self.label_mapping`` for label conversion

**Extension Pattern 3: Custom Tokenization**

.. card:: Extension Pattern 3: Custom Tokenization Schemes
   :shadow: md

   **Use Case**: Implement domain-specific tokenization (codon-aware, structure-informed, phylogeny-based).

   **Implementation Strategy**: Subclass ``OmniTokenizer`` and implement ``tokenize()`` with custom segmentation logic.

   .. code-block:: python

      from omnigenbench import OmniTokenizer

      class CodonAwareTokenizer(OmniTokenizer):
          """Tokenize DNA sequences respecting codon boundaries."""
          
          def __init__(self, vocab_file=None, **kwargs):
              super().__init__(vocab_file, **kwargs)
              self.codon_map = self._load_genetic_code()
          
          def tokenize(self, sequence, **kwargs):
              """Split into codons (triplets) for protein-coding regions."""
              # Preprocess: ensure uppercase, remove non-ATCG
              seq = sequence.upper().replace('U', 'T')
              
              # Segment into codons
              codons = []
              for i in range(0, len(seq) - 2, 3):
                  codon = seq[i:i+3]
                  if codon in self.codon_map:
                      codons.append(codon)
                  else:
                      codons.append('[UNK]')  # Unknown codon
              
              return codons
          
          def _load_genetic_code(self):
              """Standard genetic code mapping."""
              return {
                  'ATG': 'M', 'TAA': '*', 'TAG': '*', 'TGA': '*',
                  # ... full codon table
              }

   **Key Principles:**
   
   - Implement ``tokenize()`` to return list of tokens
   - Handle special cases (unknown tokens, special symbols)
   - Maintain consistency with ``encode()`` and ``decode()``
   - Document biological assumptions (reading frames, genetic code)

**Extension Pattern 4: Custom Metrics**

.. card:: Extension Pattern 4: Domain-Specific Metrics
   :shadow: md

   **Use Case**: Implement specialized evaluation measures (structural similarity, phylogenetic distance, functional enrichment).

   **Implementation Strategy**: Subclass ``OmniMetric`` and implement ``compute_metric()`` with custom calculation logic.

   .. code-block:: python

      from omnigenbench import OmniMetric
      import numpy as np
      from scipy.stats import spearmanr
      from sklearn.metrics import matthews_corrcoef

      class StructuralAccuracyMetric(OmniMetric):
          """Evaluate RNA secondary structure prediction accuracy."""
          
          def compute_metric(self, y_true, y_pred, **kwargs):
              """Calculate base-pair distance and F1 score."""
              # Filter ignored labels
              mask = (y_true != self.ignore_y)
              y_true_filtered = y_true[mask]
              y_pred_filtered = y_pred[mask]
              
              # Calculate metrics
              sensitivity = self._calc_sensitivity(y_true_filtered, y_pred_filtered)
              ppv = self._calc_ppv(y_true_filtered, y_pred_filtered)
              f1_score = 2 * (sensitivity * ppv) / (sensitivity + ppv + 1e-10)
              
              return {
                  'sensitivity': sensitivity,
                  'ppv': ppv,  # Positive predictive value
                  'f1_structure': f1_score,
                  'mcc': matthews_corrcoef(y_true_filtered, y_pred_filtered)
              }
          
          def _calc_sensitivity(self, y_true, y_pred):
              """True positive rate for base pairs."""
              true_pairs = (y_true == 1).sum()
              predicted_pairs = ((y_true == 1) & (y_pred == 1)).sum()
              return predicted_pairs / (true_pairs + 1e-10)
          
          def _calc_ppv(self, y_true, y_pred):
              """Precision for predicted base pairs."""
              predicted_pairs = (y_pred == 1).sum()
              correct_pairs = ((y_true == 1) & (y_pred == 1)).sum()
              return correct_pairs / (predicted_pairs + 1e-10)

   **Key Principles:**
   
   - Return dictionary of metric names → values
   - Handle ignored labels via ``self.ignore_y``
   - Include multiple related metrics for comprehensive evaluation
   - Document metric interpretation and biological significance


***********************************
Development Best Practices
***********************************

**Principles for Production-Ready Extensions**

When extending OmniGenBench for research or production use, adhere to these software engineering and scientific computing principles:

1. **Interface Adherence**
   
   Always inherit from the appropriate ABC and implement all required abstract methods. This ensures your custom components integrate seamlessly with AutoBench, AutoTrain, and other framework features.
   
   .. code-block:: python
   
      # ✓ Correct: Implements required interface
      class MyModel(OmniModelForSequenceClassification):
          def forward(self, **inputs):  # Required method
              ...
      
      # ✗ Incorrect: Missing required methods
      class MyModel(OmniModel):
          pass  # Will fail at runtime

2. **Comprehensive Documentation**
   
   Provide docstrings following NumPy/Google style conventions. Include:
   
   - **Purpose**: One-sentence summary of component functionality
   - **Parameters**: Type annotations and semantic descriptions
   - **Returns**: Expected output structure and data types
   - **Examples**: Minimal working code demonstrating usage
   - **Biological Context**: Domain-specific assumptions (e.g., "Assumes protein-coding regions")
   
   .. code-block:: python
   
      def compute_metric(self, y_true, y_pred, **kwargs):
          """
          Calculate Matthews Correlation Coefficient for imbalanced classification.
          
          MCC is preferred over accuracy for genomic tasks where positive class
          (e.g., binding sites) is rare compared to negative class (non-binding).
          
          Args:
              y_true (np.ndarray): Ground truth labels, shape (n_samples,)
              y_pred (np.ndarray): Predicted labels, shape (n_samples,)
              **kwargs: Additional metric-specific parameters
          
          Returns:
              dict: Metric results with keys:
                  - 'mcc': Matthews correlation coefficient [-1, 1]
                  - 'accuracy': Overall classification accuracy [0, 1]
          
          Example:
              >>> metric = ClassificationMetric(ignore_y=-100)
              >>> results = metric.compute_metric([0,1,1,0], [0,1,0,0])
              >>> print(results['mcc'])  # 0.577
          """

3. **Reproducibility & Testing**
   
   Write unit tests for all custom components. Use pytest and maintain >80% code coverage.
   
   .. code-block:: python
   
      # tests/test_custom_tokenizer.py
      import pytest
      from my_extension import CodonAwareTokenizer
      
      def test_codon_tokenization():
          """Verify codon boundary preservation."""
          tokenizer = CodonAwareTokenizer()
          
          # Test case: 9bp sequence = 3 codons
          sequence = "ATGAAATAG"
          tokens = tokenizer.tokenize(sequence)
          
          assert len(tokens) == 3
          assert tokens == ["ATG", "AAA", "TAG"]
      
      def test_non_divisible_length():
          """Handle sequences not divisible by 3."""
          tokenizer = CodonAwareTokenizer()
          sequence = "ATGAA"  # 5bp
          tokens = tokenizer.tokenize(sequence)
          assert len(tokens) == 1  # Only complete codons

4. **Error Handling & Validation**
   
   Validate inputs and provide informative error messages. Fail fast with clear diagnostics.
   
   .. code-block:: python
   
      def _load_data(self, data_path):
          if not Path(data_path).exists():
              raise FileNotFoundError(
                  f"Dataset not found: {data_path}\n"
                  f"Expected format: JSON with 'sequence' and 'label' fields"
              )
          
          data = json.load(open(data_path))
          
          # Validate required fields
          required_fields = {'sequence', 'label'}
          if not all(field in data[0] for field in required_fields):
              raise ValueError(
                  f"Missing required fields. Expected: {required_fields}, "
                  f"Found: {set(data[0].keys())}"
              )
          
          return data

5. **Performance Optimization**
   
   Profile bottlenecks before optimizing. Common optimization targets:
   
   - **Tokenization**: Cache tokenized sequences in ``__init__`` for static datasets
   - **Data Loading**: Use memory mapping (``np.memmap``) for large arrays
   - **Metric Computation**: Vectorize operations with NumPy instead of Python loops
   - **GPU Utilization**: Batch operations and minimize CPU-GPU transfers

6. **Version Control & Dependencies**
   
   Pin dependency versions in ``requirements.txt`` for reproducibility:
   
   .. code-block:: text
   
      # requirements.txt
      omnigenbench==0.3.23
      torch==2.6.0
      transformers==4.46.0
      # Custom dependencies
      viennarna==2.6.4  # RNA structure prediction

7. **Code Style Consistency**
   
   Follow project conventions:
   
   - Use ``black`` formatter with 88-character line length
   - Apply ``isort`` for import organization
   - Adhere to PEP 8 and type hint all public methods
   - Prefix private methods with underscore (``_load_data``)

**Common Pitfalls to Avoid**

❌ **Modifying Core Framework Code**
   - Fork and modify ``OmniModel`` directly
   - ✅ Instead: Subclass and override specific methods

❌ **Ignoring Type Contracts**
   - Return list from ``forward()`` when dict expected
   - ✅ Instead: Match expected return types from parent class

❌ **Hardcoding Assumptions**
   - Assume DNA sequences (breaks for RNA/protein)
   - ✅ Instead: Document assumptions and validate input types

❌ **Skipping Multi-Seed Evaluation**
   - Report single-run results
   - ✅ Instead: Use ``seeds=[0,1,2,3,4]`` in AutoBench for statistical rigor

❌ **Mixing Task Types**
   - Use ``OmniModelForClassification`` for regression
   - ✅ Instead: Choose task-specific model class matching problem type


***********************************
Trainer Backend Selection
***********************************

OmniGenBench provides three trainer backends, each optimized for different execution contexts. Understanding when to use each is critical for efficient workflows.

**Trainer Architecture Overview**

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────┐
   │                     Trainer Backends                        │
   ├──────────────────┬──────────────────┬──────────────────────┤
   │  Native Trainer  │ Accelerate Trainer│ HuggingFace Trainer │
   │                  │                   │                      │
   │  • Single GPU    │  • Multi-GPU/Node │  • Full HF ecosystem│
   │  • Explicit loop │  • Distributed    │  • Rich callbacks   │
   │  • Full control  │  • Auto-scaling   │  • Deepspeed/FSDP   │
   │  • Debugging     │  • Production     │  • Hub integration  │
   └──────────────────┴──────────────────┴──────────────────────┘

**1. Native Trainer** (``trainer="native"``)

Pure PyTorch training loop with explicit control over every step.

**When to Use:**
- Single-GPU development and debugging
- Custom training loops requiring fine-grained control
- Educational purposes to understand training mechanics
- Prototyping novel optimization strategies

**Characteristics:**
- **Pros**: Full visibility, easy debugging, minimal abstractions
- **Cons**: No automatic distributed training, manual device management
- **Default for**: ``AutoBench`` Python API (single-task focus)

**Example:**

.. code-block:: python

   from omnigenbench import AutoTrain
   
   # Native trainer for explicit control
   trainer = AutoTrain(
       dataset="my_promoters",
       config_or_model="yangheng/OmniGenome-186M",
       trainer="native",  # Explicit single-GPU training
       device="cuda:0"
   )
   trainer.run(epochs=50, batch_size=32)

**2. Accelerate Trainer** (``trainer="accelerate"``)

Leverages HuggingFace Accelerate for transparent distributed training.

**When to Use:**
- Multi-GPU training without manual parallelization
- Distributed training across multiple nodes
- Production benchmarking requiring scalability
- Mixed-precision training (FP16/BF16)

**Characteristics:**
- **Pros**: Seamless multi-GPU scaling, minimal code changes
- **Cons**: Less explicit control, requires ``accelerate`` config
- **Default for**: ``AutoTrain`` Python API, CLI commands (``ogb autobench``, ``ogb autotrain``)

**Example:**

.. code-block:: python

   from omnigenbench import AutoBench
   
   # Accelerate trainer for distributed evaluation
   bench = AutoBench(
       benchmark="RGB",
       config_or_model="yangheng/OmniGenome-186M",
       trainer="accelerate"  # Multi-GPU if available
   )
   bench.run(seeds=[0, 1, 2, 3, 4])

**Configuration** (optional ``accelerate_config.yaml``):

.. code-block:: yaml

   compute_environment: LOCAL_MACHINE
   distributed_type: MULTI_GPU
   mixed_precision: fp16
   num_processes: 4
   gpu_ids: [0,1,2,3]

**3. HuggingFace Trainer** (``trainer="hf_trainer"``)

Full integration with HuggingFace Trainer API and ecosystem.

**When to Use:**
- Leveraging HuggingFace Hub for model versioning
- Using advanced features (DeepSpeed, FSDP, gradient checkpointing)
- Callbacks for early stopping, learning rate scheduling
- Integration with Weights & Biases, TensorBoard

**Characteristics:**
- **Pros**: Rich ecosystem, production-ready, extensive callbacks
- **Cons**: More abstraction layers, heavier dependencies
- **Default for**: None (opt-in only)

**Example:**

.. code-block:: python

   from omnigenbench import AutoTrain
   from transformers import TrainingArguments
   
   # HuggingFace Trainer with custom arguments
   training_args = TrainingArguments(
       output_dir="./results",
       num_train_epochs=50,
       per_device_train_batch_size=16,
       gradient_accumulation_steps=2,
       fp16=True,
       logging_steps=100,
       save_strategy="epoch",
       evaluation_strategy="epoch",
       load_best_model_at_end=True
   )
   
   trainer = AutoTrain(
       dataset="my_dataset",
       config_or_model="yangheng/OmniGenome-186M",
       trainer="hf_trainer",
       training_args=training_args
   )
   trainer.run()

**Default Trainer Selection Matrix**

.. list-table:: Default Trainer by Entry Point
   :widths: 30 25 45
   :header-rows: 1

   * - Entry Point
     - Default Trainer
     - Rationale
   * - ``AutoBench`` (Python API)
     - ``native``
     - Single-task focus, explicit evaluation control
   * - ``AutoTrain`` (Python API)
     - ``accelerate``
     - Training benefits from distributed capabilities
   * - ``ogb autobench`` (CLI)
     - ``accelerate``
     - Production benchmarking at scale
   * - ``ogb autotrain`` (CLI)
     - ``accelerate``
     - Production training with multi-GPU support

.. important::
   **Overriding Defaults**: Always specify ``trainer`` parameter explicitly to avoid confusion:
   
   .. code-block:: python
   
      # Explicit is better than implicit
      bench = AutoBench(..., trainer="native")    # Single-GPU
      bench = AutoBench(..., trainer="accelerate") # Multi-GPU

**Performance Comparison**

.. code-block:: text

   Benchmark: RGB (12 tasks), Model: OmniGenome-186M
   
   Native Trainer (1x A100):     47 min total
   Accelerate Trainer (4x A100): 14 min total (3.4x speedup)
   HF Trainer (4x A100):         16 min total (with callbacks overhead)

**Migration Guide: Legacy to Unified CLI**

.. code-block:: bash

   # OLD: Legacy standalone commands
   autobench --config_or_model "model" --benchmark "RGB"
   autotrain --dataset "data" --model "model"
   
   # NEW: Unified ogb command (recommended)
   ogb autobench --model "model" --benchmark "RGB" --trainer accelerate
   ogb autotrain --dataset "data" --model "model" --trainer accelerate

.. tip::
   **Choosing the Right Trainer:**
   
   - **Quick experiments**: ``native`` for fast iteration
   - **Production benchmarking**: ``accelerate`` for scalability
   - **Advanced features**: ``hf_trainer`` for ecosystem integration
   - **Debugging**: ``native`` for full visibility