.. _cli:

#######################
Command Line Interface
#######################

.. rst-class:: lead

   OmniGenBench comes with a powerful and intuitive command-line interface (CLI) that allows you to run complex workflows like inference, training, benchmarking, and RNA design directly from your terminal, without writing any Python code.

All commands are now unified under the ``ogb`` command with subcommands. For backward compatibility, legacy standalone commands (``autotrain``, ``autobench``) are still available; prefer the unified ``ogb`` interface.

This page provides an overview of the most common commands. For a full list of all available commands and arguments, you can always run:

.. code-block:: bash

   ogb --help
   ogb autoinfer --help
   ogb autotrain --help
   ogb autobench --help
   ogb rna_design --help


**************************************
Unified Command Structure
**************************************

OmniGenBench provides a unified CLI through the ``ogb`` command:

.. code-block:: bash

   ogb <subcommand> [arguments]

Available subcommands:
   * ``autoinfer`` - Run inference with fine-tuned models
   * ``autotrain`` - Train or fine-tune genomic models
   * ``autobench`` - Benchmark models on standard datasets
   * ``rna_design`` - Design RNA sequences for target secondary structures

**Note**: For backward compatibility, you can still use standalone commands like ``autoinfer``, ``autotrain``, and ``autobench``, though the unified ``ogb`` interface is recommended.

.. tip::
   **Windows Users**: If you encounter Unicode/encoding errors with output characters,
   set your terminal encoding to UTF-8:
   
   .. code-block:: bash
   
      # PowerShell
      [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
      
      # Git Bash (recommended for Windows)
      export PYTHONIOENCODING=utf-8
   
   The ``ogb`` command structure has improved Windows compatibility.


**************************************
Automatic Inference Command
**************************************

The ``ogb autoinfer`` command allows you to run inference with fine-tuned models on arbitrary genomic sequences. It supports various input formats and can process single sequences or batch inference on datasets.

**Usage**

.. code-block:: bash

   ogb autoinfer --model <model_name_or_path> --sequence <sequence> --output-file <output.json>

**Example 1: Single Sequence Inference**

.. code-block:: bash

   ogb autoinfer \
     --model yangheng/ogb_tfb_finetuned \
     --sequence "ATCGATCGATCGATCGATCGATCGATCGATCG" \
     --output-file tfb_predictions.json

**Example 2: Batch Inference from JSON File**

.. code-block:: bash

   ogb autoinfer \
     --model yangheng/ogb_te_finetuned \
     --input-file sequences.json \
     --batch-size 64 \
     --output-file results.json \
     --device cuda:0

**Example 3: Inference from CSV File**

.. code-block:: bash

   ogb autoinfer \
     --model yangheng/ogb_tfb_finetuned \
     --input-file data.csv \
     --output-file predictions.json

.. note::
   The recommended interface is ``ogb autoinfer``. Legacy standalone ``autoinfer`` may not be available in all installations.

**Arguments**

*   ``--model`` **(Required)**
    Path or name of the fine-tuned model (e.g., ``yangheng/ogb_tfb_finetuned``).

*   ``--sequence`` *(Optional)*
    Input sequence(s). Can be a single sequence string, comma-separated sequences, or path to a text file.

*   ``--input-file`` *(Optional)*
    Path to JSON or CSV file with input data. 
    
    - JSON format: ``{"sequences": [...]}`` or ``{"data": [{"sequence": ..., ...}]}``
    - CSV format: Must have a ``sequence`` column

*   ``--output-file`` *(Optional, default: inference_results.json)*
    Output file to save predictions.

*   ``--batch-size`` *(Optional, default: 32)*
    Batch size for inference.

*   ``--device`` *(Optional)*
    Device to run inference on (e.g., ``cuda:0``, ``cpu``). Auto-detected if not specified.

**Input File Formats**

JSON with sequences list:

.. code-block:: json

   {
     "sequences": [
       "ATCGATCGATCGATCG",
       "GCGCGCGCGCGCGCGC"
     ]
   }

JSON with metadata:

.. code-block:: json

   {
     "data": [
       {"sequence": "ATCGATCGATCG", "gene_id": "gene_001", "label": "high"},
       {"sequence": "GCGCGCGCGCGC", "gene_id": "gene_002", "label": "low"}
     ]
   }

CSV format:

.. code-block:: csv

   sequence,gene_id,description
   ATCGATCGATCG,gene_001,5' UTR optimized
   GCGCGCGCGCGC,gene_002,5' UTR wild-type

**Output Format**

The output JSON file contains model predictions with metadata:

.. code-block:: json

   {
     "model": "yangheng/ogb_tfb_finetuned",
     "total_sequences": 2,
     "results": [
       {
         "sequence": "ATCGATCGATCG",
         "metadata": {"gene_id": "gene_001"},
         "predictions": [1, 0, 1, ...],
         "probabilities": [0.92, 0.15, 0.88, ...]
       }
     ]
   }


**************************************
Automatic Training Command
**************************************

The ``ogb autotrain`` command automates the entire training workflow from data loading to model evaluation and saving.

**Usage**

.. code-block:: bash

   ogb autotrain --dataset <dataset_name_or_path> --model <model_name> --output-dir <output_directory>

**Example 1: Basic Training**

.. code-block:: bash

   ogb autotrain \
     --dataset yangheng/tfb_promoters \
     --model zhihan1996/DNABERT-2-117M \
     --output-dir ./my_finetuned_model \
     --num-epochs 10 \
     --batch-size 32 \
     --learning-rate 5e-5

**Example 2: Training with Configuration File**

If your dataset directory contains a ``config.py`` file, it will be automatically loaded:

.. code-block:: bash

   ogb autotrain --dataset ./my_dataset --model yangheng/OmniGenome-186M

**Legacy Command (Still Available)**

.. code-block:: bash

   autotrain --dataset yangheng/tfb_promoters --model zhihan1996/DNABERT-2-117M

**Configuration File (config.py)**

Dataset configuration files use a dictionary-based approach. Here's a typical structure:

.. code-block:: python

   from omnigenbench import OmniDatasetForSequenceClassification, ClassificationMetric
   
   # Configuration dictionary
   config = {
       "task_type": "sequence_classification",
       "num_labels": 2,
       "train_file": "train.json",
       "test_file": "test.json",
       "max_length": 512,
       "batch_size": 8,
       "epochs": 50,
       "learning_rate": 2e-5,
       "seeds": [0, 1, 2],  # Multi-seed evaluation
       "compute_metrics": [ClassificationMetric().accuracy]
   }
   
   # Optional: Custom dataset class
   class Dataset(OmniDatasetForSequenceClassification):
       def __init__(self, dataset_name_or_path, tokenizer, max_length, **kwargs):
           super().__init__(dataset_name_or_path, tokenizer, max_length, **kwargs)

.. tip::
   Configuration files are optional. CLI arguments will override config file settings.
   See ``examples/autobench_gfm_evaluation/RGB/`` for more configuration examples.

**Arguments**

*   ``--dataset`` **(Required)**
    Name or path of the dataset to train on.

*   ``--model`` **(Required)**
    Name or path of the pre-trained model to fine-tune.

*   ``--output-dir`` *(Optional)*
    Directory to save the fine-tuned model.

*   ``--num-epochs`` *(Optional)*
    Number of training epochs.

*   ``--batch-size`` *(Optional)*
    Training batch size.

*   ``--learning-rate`` *(Optional)*
    Learning rate for optimization.

*   ``--trainer`` *(Optional, Default: accelerate)*
    The training backend to use. Defaults to `accelerate` for efficient distributed training.
    
    **Available Options**:
    
    - ``accelerate``: Distributed training via HuggingFace Accelerate (default, recommended)
    - ``native``: Pure PyTorch training loop (suitable for single-GPU or debugging)
    - ``hf_trainer``: HuggingFace Trainer API integration

*   ``--overwrite`` *(Optional)*
    A boolean flag. If set, it will overwrite any existing training results.


**************************************
Automatic Benchmarking Command
**************************************

The ``ogb autobench`` command is your primary tool for evaluating a model's performance on a standard benchmark dataset. It automates everything from data loading to metric calculation.

**Usage**

.. code-block:: bash

   ogb autobench --model <model_name_or_path> --benchmark <benchmark_name>

**Example**

Here's how to evaluate the ``yangheng/OmniGenome-186M`` model on the ``RGB`` benchmark:

.. code-block:: bash

   ogb autobench --model yangheng/OmniGenome-186M --benchmark RGB

**Legacy Command (Still Available)**

.. code-block:: bash

   autobench --model_name_or_path yangheng/OmniGenome-186M --benchmark RGB

**Arguments**

Here are the most common arguments for the ``autobench`` command:

*   ``--model``, ``-m`` **(Required)**
    The identifier of the model to evaluate. This can be a local path or a model name from the Hugging Face Hub (e.g., `yangheng/OmniGenome-186M`).

*   ``--benchmark``, ``-b`` **(Required)**
    The name of the benchmark dataset to use (e.g., `RGB`, `PGB`, `BEACON`, `GUE`, `GB`).
    
    **Supported Benchmarks**:
    
    - **RGB**: RNA structure and function (12 tasks)
    - **BEACON**: Multi-domain RNA understanding (13 tasks)
    - **PGB**: Plant genomics benchmarks (7 categories)
    - **GUE**: DNA general understanding evaluation (36 datasets)
    - **GB**: Classic DNA classification tasks (9 datasets)

*   ``--tokenizer``, ``-t`` *(Optional)*
    Path or name of a specific tokenizer to use. If not provided, it's inferred from the model.

*   ``--trainer`` *(Optional, Default: accelerate)*
    The training backend to use. 
    
    **Available Options**:
    
    - ``native``: Pure PyTorch training loop (suitable for single-GPU or debugging)
    - ``accelerate``: Distributed evaluation via HuggingFace Accelerate (default for CLI, recommended for multi-GPU)
    - ``hf_trainer``: HuggingFace Trainer API integration
    
    .. note::
       The CLI default is ``accelerate``, while the Python API ``AutoBench`` class defaults to ``native``.
       This design choice optimizes for different use cases: CLI users typically want distributed evaluation,
       while API users may prefer more control with the native trainer.

*   ``--overwrite`` *(Optional)*
    A boolean flag. If set, it will overwrite any existing results for this run.

*********************************
RNA Sequence Design Command
*********************************

The ``rna_design`` command provides a powerful tool for *in-silico* RNA sequence design. Given a target secondary structure in dot-bracket notation, it uses a genetic algorithm enhanced with masked language modeling to design RNA sequences that fold into that structure.

**Algorithm Overview**

The design process combines:

1. **Initialization**: Generate initial population using MLM-conditioned sampling
2. **Evolution**: Iteratively improve sequences through:
   
   - Multi-point crossover between parent sequences
   - MLM-guided mutations for biologically plausible variants
   - Multi-objective selection balancing structure similarity and thermodynamic stability

3. **Evaluation**: Structure prediction using ViennaRNA with fitness scoring
4. **Termination**: Early stopping when perfect matches are found or max generations reached

**Usage**

.. code-block:: bash

   ogb rna_design --structure "<dot_bracket_string>" [OPTIONS]

**Examples**

**Example 1: Basic Design for Simple Hairpin**

.. code-block:: bash

   ogb rna_design --structure "(((...)))"

**Example 2: Design with Custom Parameters**

.. code-block:: bash

   ogb rna_design \
       --structure "(((..(((...)))..)))" \
       --model yangheng/OmniGenome-186M \
       --mutation-ratio 0.3 \
       --num-population 200 \
       --num-generation 150 \
       --output-file complex_design.json

**Example 3: Fast Design for Quick Iteration**

.. code-block:: bash

   ogb rna_design \
       --structure "((((....))))" \
       --num-population 50 \
       --num-generation 30

**Legacy Command (Still Supported)**

For backward compatibility, the legacy command is also available:

.. code-block:: bash

   python -m omnigenbench.cli.omnigenome_cli rna_design --structure "(((...)))"

**Arguments**

*   ``--structure`` **(Required)**
    The target RNA secondary structure in dot-bracket notation.
    
    - Use ``(`` for opening base pairs, ``)`` for closing pairs, ``.`` for unpaired bases
    - Example patterns:
      
      * Simple hairpin: ``"(((...)))"`
      * Stem-loop-stem: ``"(((..(((...)))..)))"`
      * Multi-loop: ``"(((..(((...)))..(((...))).)))"`

*   ``--model`` *(Optional, Default: yangheng/OmniGenome-186M)*
    Pre-trained model name or path for MLM-guided mutations.
    
    - Larger models may produce better results but run slower
    - Recommended: ``yangheng/OmniGenome-186M`` for balance

*   ``--mutation-ratio`` *(Optional, Default: 0.5)*
    Mutation rate for genetic algorithm (0.0-1.0).
    
    - Lower values (0.1-0.2): More conservative, slower convergence
    - Higher values (0.5-0.8): More exploration, may be unstable
    - Recommended: Start with 0.5 (default), decrease for fine-tuning

*   ``--num-population`` *(Optional, Default: 100)*
    Population size for each generation.
    
    - Larger populations explore more solutions but run slower
    - Simple structures (< 20 nt): 100 is sufficient
    - Complex structures (> 50 nt): Consider 200-500

*   ``--num-generation`` *(Optional, Default: 100)*
    Maximum number of evolutionary generations.
    
    - Algorithm terminates early if perfect solution found
    - Recommended: 50-200 depending on difficulty
    - Monitor progress bar to see if more generations needed

*   ``--output-file`` *(Optional)*
    Path to JSON file to save designed sequences and parameters.
    
    Output format::
    
        {
          "structure": "(((...)))",
          "parameters": {
            "model": "yangheng/OmniGenome-186M",
            "mutation_ratio": 0.5,
            "num_population": 100,
            "num_generation": 100
          },
          "best_sequences": [
            "GCGAAACGC",
            "GCCGCCGGC",
            ...
          ]
        }

**Output**

The command outputs designed sequences to console and optionally saves to JSON file:

.. code-block:: text

   Designing RNA sequences: 100%|██████████| 50/50 [00:11<00:00,  4.52gen/s]
   ✅ Found 27 perfect matches
   
   Best RNA sequences for (((...))):
   - GCGAAACGC
   - GCCGCCGGC
   - CCCAAAGGG
   ...
   
   Results saved to results.json

**Performance Tips**

1. **Start Simple**: Test with small structures first (< 20 nt)
2. **Monitor Progress**: Real-time progress bar shows generation count and best score
3. **Early Termination**: Algorithm stops when perfect match found (Hamming distance = 0)
4. **GPU Acceleration**: Uses GPU automatically if available for MLM inference
5. **Adjust Parameters**: 
   
   - Poor convergence → Lower mutation ratio or increase population
   - Too slow → Reduce population/generations
   - Out of memory → Use CPU or reduce batch size

**Common Use Cases**

- 🧪 Synthetic biology: Design RNA switches and riboswitches
- 💊 RNA therapeutics: Optimize siRNA/miRNA structures
- 🔬 Molecular biology: Create RNA aptamers and ribozymes
- 📚 Education: Understand structure-function relationships

**See Also**

- Detailed documentation: :doc:`api/commands`
- Python API examples: ``examples/rna_sequence_design/``
- Tutorial notebook: ``examples/rna_sequence_design/RNA_Design_Tutorial.ipynb``