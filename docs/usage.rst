.. _usage:

###########
User Guide
###########

**Core Workflows for Genomic Foundation Models**

This guide demonstrates the primary usage patterns for OmniGenBench, covering the complete machine learning lifecycle: automated benchmarking with ``AutoBench``, model fine-tuning with ``AutoTrain``, and production deployment via inference APIs. Each workflow is designed to minimize boilerplate code while providing flexibility for advanced customization through configurable parameters and extension points.

**Prerequisites**: This guide assumes you have installed OmniGenBench (see :doc:`installation`). All examples use models from the HuggingFace Hub and standardized benchmark datasets that are automatically downloaded and cached on first use.

***********************************
Model & Dataset Downloading
***********************************

**Eliminating Git-LFS Dependencies for Production-Grade Model Acquisition**

OmniGenBench provides enhanced model downloading infrastructure that eliminates hard dependencies on Git-LFS while providing superior reliability, performance, and error handling. The system implements a hybrid strategy that prioritizes direct HTTP downloads via HuggingFace Hub's official API, with automatic fallback to Git-based cloning for edge cases.

**Design Philosophy**: Git-LFS pointer file corruption has been a persistent source of silent model loading failures. Users without properly configured Git-LFS installations would successfully "clone" models but receive only 100-byte pointer files instead of multi-gigabyte weight tensors—leading to models initializing with random weights and producing nonsensical predictions.

**Key Improvements**:

* **Zero Git-LFS Dependency**: Direct HTTPS downloads via ``huggingface_hub`` eliminate Git/LFS configuration requirements
* **Automatic Integrity Verification**: Post-download validation detects LFS pointer corruption
* **Performance Gains**: 33% faster downloads through CDN optimization
* **Reduced Storage**: 20% disk savings by omitting ``.git/`` repository metadata
* **Graceful Degradation**: Automatic fallback to Git clone preserves backward compatibility

Download Strategy Architecture
===============================

**Primary Method: HuggingFace Hub API (Recommended)**

Direct HTTPS downloads using ``huggingface_hub.snapshot_download()``, bypassing Git entirely.

**Advantages**:

* Only requires ``huggingface_hub>=0.20.0`` Python package—no system-level Git/LFS binaries
* CDN-accelerated transfer with automatic geo-routing
* Automatic SHA256 verification for each file chunk
* Resume support for interrupted downloads
* Compact storage excluding Git history (20-25% size reduction)

**Example**:

.. code-block:: python

   from huggingface_hub import snapshot_download

   # Download entire model repository via HTTPS
   local_path = snapshot_download(
       repo_id="yangheng/OmniGenome-186M",
       cache_dir="__OMNIGENOME_DATA__/models/",
       local_dir_use_symlinks=False,
       resume_download=True,
   )

**Fallback Method: Git Clone with LFS**

Required only when users need full Git history or when HF Hub API is unavailable.

.. warning::
   **Git-LFS Pointer File Hazard**
   
   If Git-LFS is **not** installed, ``git clone`` will substitute large files with pointer files:
   
   .. code-block:: text
   
      version https://git-lfs.github.com/spec/v1
      oid sha256:b437d27531abc123...
      size 41943280
   
   PyTorch will fail to load these pointer files as model weights, resulting in **random initialization** 
   and incorrect inference. This failure is silent—no exception raised, but predictions are meaningless.

Usage Patterns
==============

**Automatic Strategy Selection (Recommended)**

.. code-block:: python

   from omnigenbench import ModelHub

   # Automatic strategy: try HF Hub API, fallback to Git clone
   model = ModelHub.load("yangheng/OmniGenome-186M")

**Explicit HF Hub API Usage**

.. code-block:: python

   from omnigenbench import ModelHub

   # Force HuggingFace Hub API (production environments)
   model = ModelHub.load(
       "yangheng/OmniGenome-186M",
       use_hf_api=True
   )

**Direct API Access for Fine-Grained Control**

.. code-block:: python

   from omnigenbench.src.utility.model_hub.hf_download import download_from_hf_hub

   # Download with custom configuration
   path = download_from_hf_hub(
       repo_id="yangheng/ogb_tfb_finetuned",
       cache_dir="/custom/cache/",
       force_download=False,
   )

**Selective File Download** (bandwidth optimization):

.. code-block:: python

   # Download only essential files
   path = download_from_hf_hub(
       repo_id="yangheng/OmniGenome-186M",
       allow_patterns=["*.json", "*.bin"],
       ignore_patterns=["*.msgpack", "*.h5"],
   )

**Downloading Benchmarks and Datasets**

The same robust API works for benchmark datasets:

.. code-block:: python

   from omnigenbench.src.utility.hub_utils import download_benchmark

   # Download benchmark with HF Hub API (recommended)
   benchmark_path = download_benchmark(
       "RGB",  # Short name
       use_hf_api=True
   )
   
   # Or specify full HuggingFace dataset repository
   benchmark_path = download_benchmark(
       "yangheng/OmniGenBench_RGB",
       use_hf_api=True
   )
   
   # Force re-download to update cached benchmark
   benchmark_path = download_benchmark(
       "RGB",
       force_download=True,
       use_hf_api=True
   )

**Automatic Method Selection in AutoBench**:

.. code-block:: python

   from omnigenbench import AutoBench

   # AutoBench automatically uses robust HF Hub API for benchmark downloads
   bench = AutoBench(
       benchmark="RGB",  # Automatically downloaded via HF Hub API
       model_name_or_path="yangheng/OmniGenome-186M"
   )
   bench.run()

Download Integrity Verification
================================

All downloads include automatic validation:

.. code-block:: python

   from omnigenbench.src.utility.model_hub.hf_download import (
       download_from_hf_hub,
       verify_download_integrity
   )

   # Download and verify
   path = download_from_hf_hub("yangheng/OmniGenome-186M")
   is_valid = verify_download_integrity(path)
   
   if not is_valid:
       raise RuntimeError("Download corrupted—LFS pointer detected")

**Validation Checks**:

1. File existence: Verify all required files present
2. LFS pointer detection: Scan .bin files for Git-LFS pointer headers
3. Size validation: Flag suspiciously small files (<200 bytes)

Troubleshooting Common Issues
==============================

**Problem: Model Produces Random Predictions**

**Symptoms**: Model loads successfully but predictions are nonsensical.

**Diagnosis**:

.. code-block:: python

   from omnigenbench.src.utility.model_hub.hf_download import verify_download_integrity

   is_valid = verify_download_integrity(
       "__OMNIGENOME_DATA__/models/yangheng--ogb_tfb_finetuned"
   )
   
   if not is_valid:
       print("Git-LFS pointer detected—model weights not downloaded")

**Solution**: Re-download with HF Hub API:

.. code-block:: python

   from omnigenbench import ModelHub

   # Force HF Hub API re-download
   model = ModelHub.load(
       "yangheng/ogb_tfb_finetuned",
       use_hf_api=True,
       force_download=True
   )

**Problem: Network Connection Failures**

HF Hub API includes automatic resume capability—simply re-run the download command:

.. code-block:: python

   # Re-run automatically resumes from last verified chunk
   path = download_from_hf_hub(
       "yangheng/OmniGenome-186M",
       force_download=False
   )

Performance Benchmarks
======================

Tested with ``yangheng/OmniGenome-186M`` (~200MB model):

.. list-table::
   :header-rows: 1
   :widths: 30 20 25 25

   * - Method
     - Time
     - Dependencies
     - Risk Level
   * - HF Hub API
     - **30 seconds**
     - ``huggingface_hub``
     - None
   * - Git Clone (with LFS)
     - 45 seconds
     - ``git`` + ``git-lfs``
     - Low
   * - Git Clone (without LFS)
     - 5 seconds ⚠️
     - ``git`` only
     - **High** (pointer files)

**Storage Efficiency**:

* HF Hub API: 200 MB (model files only)
* Git Clone: 250 MB (model files + ``.git/`` metadata)
* **Savings**: 50 MB (20% reduction)

Best Practices
==============

**Recommended**:

1. Default to HF Hub API for all production workloads
2. Always validate downloads with ``verify_download_integrity()``
3. Use selective downloads (``allow_patterns``) to optimize bandwidth
4. Handle private models with HuggingFace access tokens

**Avoid**:

1. Implicit Git-LFS dependencies (use ``use_hf_api=True``)
2. Ignoring verification failures
3. Mixing download methods without cleanup

.. tip::
   For detailed troubleshooting, repository metadata queries, and migration guides, see the 
   complete API reference at :doc:`api/model_hub`.

*************************************
Workflow 1: Automated Benchmarking
*************************************

**Objective**: Evaluate pre-trained genomic foundation models on standardized benchmark suites with reproducible protocols and multi-seed statistical rigor.

The ``AutoBench`` class orchestrates the complete evaluation pipeline: benchmark dataset acquisition, model loading from HuggingFace Hub, distributed inference across tasks, metric calculation with domain-specific measures, and results serialization. It implements best practices for genomic ML evaluation, including multi-seed averaging for variance quantification and task-specific metric selection aligned with biological validation standards.

**Basic Usage Pattern**

Evaluate `yangheng/OmniGenome-186M <https://huggingface.co/yangheng/OmniGenome-186M>`_ (plant genome foundation model with 186M parameters) on the RGB benchmark (RNA Genome Benchmark: 12 RNA sequence understanding tasks):

.. code-block:: python

   from omnigenbench import AutoBench

   # Initialize benchmarking pipeline
   bench = AutoBench(
       benchmark="RGB",                              # Benchmark identifier (RGB, BEACON, PGB, GUE, GB)
       config_or_model="yangheng/OmniGenome-186M" # HF Hub model ID or local path
   )

   # Execute evaluation workflow
   # Automatically handles: data loading, tokenization, inference, metric computation
   bench.run()
   
   # Results saved to: ./autobench_results/RGB/OmniGenome-186M/
   # Output format: JSON with per-task metrics and aggregated statistics

**Statistical Rigor: Multi-Seed Evaluation**

For publication-quality results, evaluate with multiple random seeds to quantify variance:

.. code-block:: python

   from omnigenbench import AutoBench

   bench = AutoBench(
       benchmark="RGB",
       config_or_model="yangheng/OmniGenome-186M"
   )
   
   # Run with 5 independent initializations
   bench.run(seeds=[0, 1, 2, 3, 4])
   
   # Results include: mean ± std for each metric across seeds
   # Example output: MCC: 0.742 ± 0.015, F1: 0.863 ± 0.009

.. tip::
   **Why Multiple Seeds for Evaluation?** 
   
   Random initialization, data shuffling, and dropout stochasticity cause performance variance 
   across training runs. Multi-seed evaluation (typically 3-5 independent runs) provides:
   
   - **Statistical validity**: Mean and standard deviation enable hypothesis testing and confidence intervals
   - **Variance quantification**: Distinguish models with stable performance from those with volatile behavior
   - **Reproducibility verification**: Demonstrates results aren't artifacts of fortunate initialization
   - **Publication standards**: Required by most computational biology journals for method comparison

.. note::
   **Trainer Backend Selection Strategy**: 
   
   - **Python API** (``AutoBench``): Defaults to ``native`` trainer for single-GPU explicit control and debugging
   - **CLI** (``ogb autobench``): Defaults to ``accelerate`` trainer for distributed multi-GPU capabilities
   
   This design optimizes for different use cases:
   
   - API users typically prioritize explicit control over evaluation steps (``native`` backend)
   - CLI users typically prioritize production-scale distributed evaluation (``accelerate`` backend)
   
   For multi-GPU distributed evaluation via Python API, explicitly specify ``trainer="accelerate"``:
   
   .. code-block:: python
   
      # Single-GPU native trainer (Python API default)
      bench = AutoBench(benchmark="RGB", config_or_model="model")
      
      # Multi-GPU distributed evaluation (override default)
      bench = AutoBench(
          benchmark="RGB", 
          config_or_model="model", 
          trainer="accelerate"
      )
   
   **Available trainer backends:**
   
   - ``native``: Pure PyTorch evaluation loop, single-GPU, explicit control (Python API default for AutoBench)
   - ``accelerate``: HuggingFace Accelerate, multi-GPU/multi-node distributed with gradient accumulation (CLI default)
   - ``hf_trainer``: HuggingFace Trainer API with full ecosystem integration (callbacks, logging, checkpointing)

**********************
Training a New Model
**********************

OmniGenBench simplifies the training process with the ``AutoTrain`` class. You provide a dataset and a base model, and it handles the rest.

In this example, we'll fine-tune the `yangheng/OmniGenome-186M` model on a custom dataset named "MyCustomDataset".

.. code-block:: python

   from omnigenbench import AutoTrain

   # Initialize the trainer with your dataset and a base model
   trainer = AutoTrain(dataset="MyCustomDataset", config_or_model="yangheng/OmniGenome-186M")

   # Start the training process
   trainer.run()

.. tip::
   Your dataset should be prepared in a compatible format. Refer to the :ref:`Data Template <data-template>` section below for details on data formatting.

.. note::
   **Trainer Backend Selection**: 
   
   - **Python API** (``AutoTrain``): Defaults to ``accelerate`` trainer for distributed training efficiency
   - **CLI** (``ogb autotrain``): Also defaults to ``accelerate`` trainer
   
   This design choice recognizes that training typically benefits from distributed capabilities
   even on single-GPU systems (gradient accumulation, mixed precision, memory optimization).
   
   For single-GPU training or debugging, specify ``trainer="native"``:
   
   .. code-block:: python
   
      # Multi-GPU distributed training (default for AutoTrain)
      trainer = AutoTrain(dataset="MyData", config_or_model="model")
      
      # Single-GPU training with explicit control (for debugging)
      trainer = AutoTrain(
          dataset="MyData", 
          config_or_model="model", 
          trainer="native"
      )
   
   **Available trainer backends:**
   
   - ``accelerate``: HuggingFace Accelerate, multi-GPU/multi-node distributed (default for AutoTrain)
   - ``native``: Pure PyTorch training loop, single-GPU, explicit control
   - ``hf_trainer``: HuggingFace Trainer API integration with full ecosystem support

.. note::
   **CLI Alternative**: You can also train models from the command line:
   
   .. code-block:: bash
   
      ogb autotrain \
          --dataset ./my_dataset \
          --model yangheng/OmniGenome-186M \
          --epochs 50 \
          --batch-size 32 \
          --trainer accelerate
   
   See :doc:`cli` for all training options and configuration.

.. _data-template:

**********************
Data Template & Formats
**********************

OmniGenBench supports flexible data loading for genomic machine learning tasks. To ensure compatibility, your data should follow a simple template and be saved in one of the supported formats.

**Data Template: {sequence, label} Structure**

Each data sample should be a dictionary with at least two keys:

- ``sequence``: The biological sequence (DNA, RNA, or protein) as a string.
- ``label``: The target value for the task (classification, regression, etc.).

**Example for Classification** (JSON format):

.. code-block:: json

   [
     {"sequence": "ATCGATCGATCG", "label": "0"},
     {"sequence": "GCTAGCTAGCTA", "label": "1"}
   ]

**Example for Regression** (JSON format):

.. code-block:: json

   [
     {"sequence": "ATCGATCGATCG", "label": 0.75},
     {"sequence": "GCTAGCTAGCTA", "label": -1.2}
   ]

OmniGenBench will automatically standardize common key names. For example, ``seq`` or ``text`` will be treated as ``sequence``, and ``label`` will be standardized to ``labels`` internally.

**Supported Data Formats**

1. **JSON (`.json`)**: Recommended. A list of dictionaries as shown above. Also supports JSON Lines (`.jsonl`).
2. **CSV (`.csv`)**: Must have columns for ``sequence`` and ``label``.
3. **Parquet (`.parquet`)**: Columns for ``sequence`` and ``label``.
4. **FASTA (`.fasta`, `.fa`, etc.)**: Sequence data only. Labels must be provided separately or inferred.
5. **FASTQ (`.fastq`, `.fq`)**: Sequence and quality scores. Labels must be provided separately or inferred.
6. **BED (`.bed`)**: Genomic intervals. Sequence and label columns may need to be added.
7. **Numpy (`.npy`, `.npz`)**: Array of dictionaries with ``sequence`` and optional ``label``.

For supervised tasks, ensure every sample has both a ``sequence`` and a ``label``. For unsupervised or sequence-only tasks, only the ``sequence`` key is required.

*********************
Inference with a Model
*********************

Once you have a trained model, running inference is straightforward. There are two safe patterns depending on your assets:

1) Use task-specific OmniModel classes when you know the task type (recommended)

.. code-block:: python

   from omnigenbench import OmniModelForSequenceClassification, OmniTokenizer

   # Example: multi-label TF binding (919 tasks)
   tokenizer = OmniTokenizer.from_pretrained("yangheng/ogb_tfb_finetuned")
   model = OmniModelForSequenceClassification(
       "yangheng/ogb_tfb_finetuned",
       tokenizer=tokenizer,
       num_labels=919,  # or pass label2id if available
   )

   sequence = "ATCGATCGATCGATCGATCGATCGATCGATCG"
   outputs = model.inference(sequence)
   print(outputs.keys())  # e.g., dict with predictions/probabilities/logits

2) Use ModelHub to load fine-tuned OmniGenBench models with metadata

.. code-block:: python

   from omnigenbench import ModelHub

   # Load models saved with OmniGenBench (metadata enables task context restoration)
   model = ModelHub.load("yangheng/ogb_tfb_finetuned")
   result = model.inference("ATCGATCGATCG")  # Works when metadata.json is present
   
.. note::
   ``ModelHub.load()`` clones models from HuggingFace Hub to local cache (``__OMNIGENOME_DATA__/models/``)
   on first use, then loads from local files only. It returns a fully-configured task-specific model
   when ``metadata.json`` is present, otherwise returns a standard Transformers model with attached tokenizer.
   
   For models without OmniGenBench metadata, prefer instantiating task-specific OmniModel classes
   directly (Pattern 1) with explicit ``num_labels`` or ``label2id`` configuration.

.. note::
   **CLI Alternative**: You can also run inference from the command line:
   
   .. code-block:: bash
   
      ogb autoinfer --model yangheng/ogb_tfb_finetuned --sequence "ATCGATCGATCG"
   
   The CLI uses the same metadata-aware loading under the hood. See :doc:`cli` for complete options.


************************************
Embedding Extraction & Visualization
************************************

All OmniModel-based classes inherit ``EmbeddingMixin``, which provides built-in support for extracting sequence embeddings and visualizing attention patterns. These features are essential for:

* **Downstream Analysis**: Using genomic embeddings for clustering, classification, or similarity search
* **Model Interpretation**: Understanding what patterns the model learns via attention visualization
* **Transfer Learning**: Extracting features for training custom models

**Extracting Sequence Embeddings**

Generate fixed-length vector representations of genomic sequences:

.. code-block:: python

   from omnigenbench import OmniModelForEmbedding
   
   model = OmniModelForEmbedding("yangheng/OmniGenome-186M")
   sequences = ["ATCGATCGATCGATCG", "GCTAGCTAGCTAGCTA"]
   embeddings = model.batch_encode(sequences, agg="mean")
   print(embeddings.shape)  # (2, hidden_size)
   
   # Use embeddings for downstream tasks (clustering, similarity, etc.)

**Extracting Attention Scores**

Visualize which positions in the sequence the model attends to:

.. code-block:: python

   from omnigenbench import OmniModelForSequenceClassification, OmniTokenizer
   
   tokenizer = OmniTokenizer.from_pretrained("yangheng/OmniGenome-186M")
   # Any OmniModel subclass works; num_labels can be a placeholder for analysis-only use
   model = OmniModelForSequenceClassification(
       "yangheng/OmniGenome-186M", tokenizer=tokenizer, num_labels=2
   )
   
   sequence = "ATCGATCGATCGATCG"
   result = model.extract_attention_scores(sequence)
   attn = result["attentions"]  # (num_layers, num_heads, seq_len, seq_len)
   print(attn.shape)

**Example Notebooks**

For complete tutorials with visualization examples:

* **Embedding Tutorial**: ``examples/genomic_embeddings/RNA_Embedding_Tutorial.ipynb``
* **Attention Analysis**: ``examples/attention_score_extraction/Attention_Analysis_Tutorial.ipynb``

.. tip::
   **Embedding Applications**:
   
   - **Sequence Similarity**: Use cosine similarity between embeddings to find similar sequences
   - **Clustering**: Group sequences by biological function using k-means or hierarchical clustering
   - **Feature Extraction**: Use embeddings as input features for traditional ML models
   - **Visualization**: Project high-dimensional embeddings to 2D/3D using t-SNE or UMAP


.. ************************************
.. Downloading Benchmarks (Datasets)
.. ************************************

.. .. code-block:: python

..    from omnigenbench.utility.hub_utils import download_model, download_benchmark
..    download_model("OmniGenome-186M-SSP")
..    download_benchmark("RGB")



************************************
Managing Datasets and Models Manually
************************************

While the ``AutoBench`` and ``AutoTrain`` pipelines handle asset downloads automatically, you might need to download models or benchmark datasets manually in certain scenarios, such as:

*   Pre-loading assets in an environment with limited internet access.
*   Inspecting the contents of a benchmark dataset.
*   Scripting custom workflows.

The ``omnigenbench`` module provides simple functions for this purpose. These functions download files from the Hugging Face Hub and store them in a local cache for future use, avoiding redundant downloads.

.. tip::
   The first time you download an asset, it might take a while depending on its size and your connection speed. Subsequent calls for the same asset will be nearly instant as it will be loaded directly from your local cache.


To download a specific benchmark dataset, use the ``download_benchmark`` function. Provide the benchmark's name as an argument.

.. code-block:: python

   from omnigenbench import download_benchmark

   # Define the name of the benchmark to download
   benchmark_name = "RGB"

   # Download the dataset from the Hugging Face Hub
   local_path = download_benchmark(benchmark_name)

   print(f"Benchmark '{benchmark_name}' downloaded successfully to: {local_path}")


Similarly, the ``download_model`` function allows you to fetch a pre-trained model. Use the model's identifier from the Hub.

.. code-block:: python

   from omnigenbench import download_model

   # Define the model identifier from the Hugging Face Hub
   model_id = "OmniGenome-186M-SSP"

   # Download the model files
   local_path = download_model(model_id)

   print(f"Model '{model_id}' downloaded successfully to: {local_path}")


*************************
Common Pitfalls & Tips
*************************

**Task Type Matters**

Always use the appropriate task-specific model class for your problem. OmniGenBench provides specialized model classes for different genomic tasks:

.. code-block:: python

   # Binary/Multi-class/Multi-label Classification
   # Use for: Transcription factor binding, promoter classification, RNA type classification
   from omnigenbench import OmniModelForSequenceClassification
   model = OmniModelForSequenceClassification("yangheng/OmniGenome-186M", num_labels=2)
   
   # Regression Tasks
   # Use for: Expression levels, mRNA degradation rates, variant effect prediction
   from omnigenbench import OmniModelForRegression
   model = OmniModelForRegression("yangheng/OmniGenome-186M")
   
   # Per-nucleotide Predictions (Token Classification)
   # Use for: Splice site detection, secondary structure prediction, methylation sites
   from omnigenbench import OmniModelForTokenClassification
   model = OmniModelForTokenClassification("yangheng/OmniGenome-186M", num_labels=3)
   
   # RNA Sequence Design
   # Use for: Designing RNA sequences that fold into target structures
   from omnigenbench import OmniModelForRNADesign
   model = OmniModelForRNADesign("yangheng/OmniGenome-186M")
   sequences = model.design(structure="(((...)))")

.. tip::
   **Choosing the Right Task Type**:
   
   - **Classification**: When predicting discrete categories (e.g., high/low expression, present/absent)
   - **Regression**: When predicting continuous values (e.g., expression level: 0.5, 2.3, 10.1)
   - **Token Classification**: When predicting labels for each position in the sequence
   - **RNA Design**: When generating sequences for target secondary structures

.. important::
   **RNA Design Returns a List**: The RNA design model always returns a list of sequences 
   (up to 25 candidates), never a single sequence. Always handle the output as a list:
   
   .. code-block:: python
   
      # Correct: Handle as list
      sequences = model.design(structure="(((...)))")
      for seq in sequences:
          print(seq)
      
      # Incorrect: Assuming single sequence
      sequence = model.design(structure="(((...)))")  # This is a list!
      print(sequence.upper())  # Will fail!

**ModelHub vs Direct Instantiation**

Use ``ModelHub.load()`` for quick inference with OmniGenBench-saved fine-tuned models (loads model + tokenizer and restores task context when metadata is present):

.. code-block:: python

   model = ModelHub.load("yangheng/ogb_tfb_finetuned")
   outputs = model.inference("ATCGATCG")

Use direct class instantiation when you need custom configuration or when the HF repo has no OmniGenBench metadata:

.. code-block:: python

   # For training or custom configuration
   from omnigenbench import OmniModelForSequenceClassification
   model = OmniModelForSequenceClassification(
       config_or_model="yangheng/OmniGenome-186M",
       num_labels=919,  # Custom number of labels
       problem_type="multi_label_classification"
   )

**Data Format Requirements**

Ensure your data has the correct keys:

.. code-block:: python

   # Correct format
   data = [
       {"sequence": "ATCG", "label": 0},
       {"sequence": "GCTA", "label": 1}
   ]
   
   # Also accepted (auto-standardized)
   data = [
       {"seq": "ATCG", "labels": 0},  # 'seq' -> 'sequence', 'labels' -> 'label'
       {"text": "GCTA", "target": 1}  # 'text' -> 'sequence', 'target' -> 'label'
   ]

**GPU Memory Management**

For large models or long sequences:

.. code-block:: python

   # Reduce batch size
   bench = AutoBench(benchmark="RGB", config_or_model="large_model")
   bench.run(batch_size=4)  # Default is often 8-32
   
   # Use gradient checkpointing
   from omnigenbench import OmniModelForSequenceClassification
   model = OmniModelForSequenceClassification("model", gradient_checkpointing=True)
   
   # Use mixed precision
   bench = AutoBench(benchmark="RGB", config_or_model="model", autocast="bf16")

***************
What's Next?
***************

You've now seen the basic workflows in OmniGenBench! To dive deeper, explore these resources:

**Core Documentation:**

*   :doc:`cli` - Command-line interface for codeless operations
*   :doc:`design_principle` - Understanding the four abstract base classes (OmniModel, OmniDataset, OmniTokenizer, OmniMetric)
*   :doc:`api_reference` - Complete API reference for all classes and functions

**Detailed Guides (in API Reference):**

*   :doc:`api/trainers` - Comprehensive trainer guide (Native, Accelerate, HuggingFace)
*   :doc:`api/downstream_datasets` - Dataset classes, formats, and preprocessing
*   :doc:`api/downstream_models` - Model architectures and task-specific models
*   :doc:`api/commands` - CLI command reference with examples

**Quick Reference:**

.. code-block:: python

   # Model Loading (Recommended)
   from omnigenbench import ModelHub
   model = ModelHub.load("yangheng/OmniGenome-186M")
   
   # Automated Training (Recommended)
   from omnigenbench import AutoTrain
   trainer = AutoTrain(dataset="./my_dataset", config_or_model="yangheng/OmniGenome-186M")
   trainer.run()
   
   # Dataset Loading
   from omnigenbench import OmniDatasetForSequenceClassification
   dataset = OmniDatasetForSequenceClassification("data.json", tokenizer, max_length=512)
   
   # CLI Commands
   # ogb autobench --model yangheng/OmniGenome-186M --benchmark RGB
   # ogb autotrain --dataset data --model model
   # ogb autoinfer --model model --sequence "ATCG"