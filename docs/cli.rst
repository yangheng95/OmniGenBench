.. _cli:

#######################
Command Line Interface
#######################

.. rst-class:: lead

   OmniGenBench comes with a powerful and intuitive command-line interface (CLI) that allows you to run complex workflows like inference, training, benchmarking, and RNA design directly from your terminal, without writing any Python code.

All commands are now unified under the ``ogb`` command with subcommands. For backward compatibility, legacy standalone commands (``autoinfer``, ``autotrain``, ``autobench``) are still supported.

This page provides an overview of the most common commands. For a full list of all available commands and arguments, you can always run:

.. code-block:: bash

   ogb --help
   ogb autoinfer --help
   ogb autotrain --help
   ogb autobench --help


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

**Note**: For backward compatibility, you can still use standalone commands like ``autoinfer``, ``autotrain``, and ``autobench``.


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

**Legacy Command (Still Supported)**

.. code-block:: bash

   autoinfer --model yangheng/ogb_tfb_finetuned --sequence "ATCGATCGATCG"

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

**Legacy Command (Still Supported)**

.. code-block:: bash

   autotrain --dataset yangheng/tfb_promoters --model zhihan1996/DNABERT-2-117M

**Configuration File (config.py)**

.. code-block:: python

   from omnigenbench import AutoConfig

   config = AutoConfig(
       model="yangheng/OmniGenome-186M",
       num_epochs=10,
       batch_size=32,
       learning_rate=5e-5,
       output_dir="./my_model",
       save_steps=500,
       eval_steps=500
   )

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

**Legacy Command (Still Supported)**

.. code-block:: bash

   autobench --model_name_or_path yangheng/OmniGenome-186M --benchmark RGB

**Arguments**

Here are the most common arguments for the ``autobench`` command:

*   ``--model``, ``-m`` **(Required)**
    The identifier of the model to evaluate. This can be a local path or a model name from the Hugging Face Hub (e.g., `yangheng/OmniGenome-186M`).

*   ``--benchmark``, ``-b`` **(Required)**
    The name of the benchmark dataset to use (e.g., `RGB`, `PGB`, `BEACON`).

*   ``--tokenizer``, ``-t`` *(Optional)*
    Path or name of a specific tokenizer to use. If not provided, it's inferred from the model.

*   ``--trainer`` *(Optional)*
    The training backend to use. Defaults to `native`.
    *Choices*: `native`, `accelerate`, `hf_trainer`.

*   ``--overwrite`` *(Optional)*
    A boolean flag. If set, it will overwrite any existing results for this run.

*   ``--bs_scale`` *(Optional)*
    Batch size scaling factor. Allows you to adjust the batch size without setting it directly. Defaults to `1`.

*********************************
RNA Structure Design Command
*********************************

The ``rna_design`` command provides a powerful tool for *in-silico* RNA design. Given a target secondary structure, it uses an evolutionary algorithm powered by a generative model to design RNA sequences that are likely to fold into that structure.

**Usage**

.. code-block:: bash

   omnigenbench rna_design --structure "<dot_bracket_string>" --model <model_name_or_path>

**Example**

To design a sequence for a simple hairpin loop structure:

.. code-block:: bash

   omnigenbench rna_design --structure "((((...))))" --model yangheng/OmniGenome-186M --num-generation 50

**Arguments**

*   ``--structure`` **(Required)**
    The target RNA secondary structure specified in dot-bracket notation.

*   ``--model`` **(Required)**
    The generative model to use for scoring and guiding the design process.

*   ``--mutation-ratio`` *(Optional)*
    The mutation rate for the genetic algorithm. A float between 0 and 1.

*   ``--num-population`` *(Optional)*
    The size of the population in each generation of the evolutionary algorithm.

*   ``--num-generation`` *(Optional)*
    The total number of generations to run the evolution for. More generations can lead to better results but will take longer.

*   ``--output-file`` *(Optional)*
    Path to a file where the final designed sequences will be saved.