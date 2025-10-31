############
OmniGenBench
############

**A Unified Framework for Genomic Foundation Model Development and Benchmarking**

OmniGenBench is a comprehensive toolkit for developing, evaluating, and deploying genomic foundation models across diverse biological sequence analysis tasks. The framework implements a principled software architecture grounded in four core abstract base classes (``OmniModel``, ``OmniDataset``, ``OmniTokenizer``, ``OmniMetric``), providing task-specific abstractions, standardized evaluation protocols, and seamless integration with the HuggingFace ecosystem to address the unique computational challenges of genomic machine learning.

**Documentation Structure**

* :doc:`installation` - System requirements and installation procedures
* :doc:`usage` - Core workflows, model downloading, and API usage patterns
* :doc:`cli` - Command-line interface reference
* :doc:`design_principle` - Architectural principles and extension mechanisms
* :doc:`troubleshooting` - Common issues and solutions
* :doc:`api_reference` - Complete API specification



.. .. raw:: html

..    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.5em; margin-top: 1em;">
..        <a href="installation.rst" class="nav-card"> <h4 style="margin-top: 0; color: #2c3e50; font-size: 1.2em;">🚀 Installation</h4> <p style="margin-bottom: 0; color: #555; line-height: 1.6;">Start your journey by setting up OmniGenBench on your system.</p> </a>
..        <a href="usage.rst" class="nav-card"> <h4 style="margin-top: 0; color: #2c3e50; font-size: 1.2em;">📖 Basic Usage</h4> <p style="margin-bottom: 0; color: #555; line-height: 1.6;">Learn the fundamental workflows and how to run your first task.</p> </a>
..        <a href="cli.rst" class="nav-card"> <h4 style="margin-top: 0; color: #2c3e50; font-size: 1.2em;">🛠️ Command Usage (CLI)</h4> <p style="margin-bottom: 0; color: #555; line-height: 1.6;">Master the command-line interface for powerful and flexible operations.</p> </a>
..        <a href="design_principle.rst" class="nav-card"> <h4 style="margin-top: 0; color: #2c3e50; font-size: 1.2em;">🏗️ Design Principles</h4> <p style="margin-bottom: 0; color: #555; line-height: 1.6;">Understand the core architecture and design choices behind the library.</p> </a>
..        <a href="api_reference.rst" class="nav-card"> <h4 style="margin-top: 0; color: #2c3e50; font-size: 1.2em;">📚 API Reference</h4> <p style="margin-bottom: 0; color: #555; line-height: 1.6;">Get detailed information about all public classes, functions, and methods.</p> </a>
..    </div>


.. raw:: html

   <div class="nav-grid">
       <a href="installation.html" class="nav-card">
           <h4 class="nav-card-title">🚀 Installation</h4>
           <p class="nav-card-description">Start your journey by setting up OmniGenBench on your system.</p>
       </a>
       <a href="usage.html" class="nav-card">
           <h4 class="nav-card-title">📖 User Guide</h4>
           <p class="nav-card-description">Learn workflows, model downloading, and API usage patterns.</p>
       </a>
       <a href="cli.html" class="nav-card">
           <h4 class="nav-card-title">🛠️ CLI Commands</h4>
           <p class="nav-card-description">Master the command-line interface for codeless operations.</p>
       </a>
       <a href="design_principle.html" class="nav-card">
           <h4 class="nav-card-title">🏗️ Design Principles</h4>
           <p class="nav-card-description">Understand the core architecture and four abstract base classes.</p>
       </a>
       <a href="troubleshooting.html" class="nav-card">
           <h4 class="nav-card-title">🔧 Troubleshooting</h4>
           <p class="nav-card-description">Solutions to common issues and comprehensive FAQ.</p>
       </a>
       <a href="api_reference.html" class="nav-card">
           <h4 class="nav-card-title">📚 API Reference</h4>
           <p class="nav-card-description">Complete reference for all classes, functions, and methods.</p>
       </a>
   </div>



.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Get Started

   installation
   usage
   troubleshooting

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: CLI Interface

   cli

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Core Concepts

   design_principle

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: API Documentation

   api_reference

**Framework Capabilities**

* **30+ Pre-trained Models**: Production-ready genomic foundation models from HuggingFace Hub, supporting DNA and RNA sequences across multiple species (plants, animals, microbes)
* **80+ Standardized Benchmarks**: Comprehensive evaluation suites across five domains:
  
  * **RGB**: 12 RNA structure and function tasks
  * **BEACON**: 13 multi-domain RNA tasks
  * **PGB**: 7 plant genomics categories with long-range context
  * **GUE**: 36 DNA general understanding datasets
  * **GB**: 9 classic DNA classification benchmarks

* **Three-Line Inference API**: Minimal-code deployment pattern: load from HuggingFace Hub, predict on sequences, interpret results
* **Distributed Training Infrastructure**: Three trainer backends with complementary strengths:
  
  * ``native``: Pure PyTorch training loop for explicit control and debugging
  * ``accelerate``: HuggingFace Accelerate for multi-GPU/multi-node distributed training
  * ``hf_trainer``: HuggingFace Trainer API for full ecosystem integration

* **Extensible Architecture**: Plug-and-play customization through abstract base classes - extend with custom models, datasets, tokenizers, or metrics without modifying core code
* **Interpretability Tools**: Built-in embedding extraction (``encode()``), attention visualization (``extract_attention_scores()``), and similarity computation via ``EmbeddingMixin``
* **RNA Sequence Design**: Structure-to-sequence generation using genetic algorithms enhanced with masked language model guidance for target secondary structure realization

**Quick Start**

**Python API**: Three-line inference workflow for rapid deployment

.. code-block:: python

   from omnigenbench import ModelHub
   
   # Load fine-tuned model from HuggingFace Hub (auto-cached on first use)
   model = ModelHub.load("yangheng/ogb_tfb_finetuned")
   
   # Predict transcription factor binding sites (919-way multi-label classification)
   predictions = model.inference("ATCGATCGATCGATCG" * 20)
   # Returns: {'predictions': array([1, 0, 1, ...]), 
   #           'probabilities': array([0.92, 0.08, 0.87, ...])}
   
   # Interpret results: identify binding sites
   import numpy as np
   binding_tfs = np.where(predictions['predictions'] == 1)[0]
   print(f"Predicted binding sites: {len(binding_tfs)}/919 transcription factors")

**Command-Line Interface**: Production workflows with zero-code execution

.. code-block:: bash

   # Automated benchmarking: evaluate model across 12 RNA tasks with multi-seed averaging
   ogb autobench \
       --model yangheng/OmniGenome-186M \
       --benchmark RGB \
       --seeds 0 1 2 \
       --trainer accelerate
   # Output: Mean ± std per metric (e.g., MCC: 0.742 ± 0.015, F1: 0.863 ± 0.009)
   
   # Automated training: fine-tune on custom dataset with distributed training
   ogb autotrain \
       --dataset ./my_promoters \
       --model yangheng/OmniGenome-186M \
       --epochs 50 \
       --batch-size 32 \
       --trainer accelerate
   
   # Automated inference: batch prediction on genomic sequences
   ogb autoinfer \
       --model yangheng/ogb_tfb_finetuned \
       --input-file sequences.json \
       --output-file predictions.json
   
   # RNA sequence design: generate sequences realizing target secondary structure
   ogb rna_design \
       --structure "(((...)))" \
       --model yangheng/OmniGenome-186M \
       --num-population 200 \
       --num-generation 100