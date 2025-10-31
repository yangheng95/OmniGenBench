CLI Commands
============

Overview
--------

OmniGenBench provides a comprehensive command-line interface (CLI) for codeless operation, enabling users to quickly execute benchmarking, training, inference, and specialized genomic tasks directly from the terminal without writing Python scripts.

**Version**: OmniGenBench v0.3.23alpha

Key Features
~~~~~~~~~~~~

✅ **Unified CLI Entry Points**:
   - ``ogb`` - Main unified command (recommended)
   - ``python -m omnigenbench.cli.omnigenome_cli`` - Alternative entry point

✅ **Core Commands**:
   - ``autobench`` - Automated benchmarking on standard datasets
   - ``autotrain`` - Automated training/fine-tuning
   - ``autoinfer`` - Automated inference with fine-tuned models
   - ``rna_design`` - RNA sequence design for target structures

✅ **Supported Benchmarks**:
   - **RGB** - RNA Genomic Benchmark
   - **PGB** - Protein Genomic Benchmark
   - **GUE** - Genomics Understanding Evaluation
   - **GB** - General Benchmark
   - **BEACON** - Comprehensive benchmark suite

Quick Start Examples
--------------------

Unified Command (OGB CLI)
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Recommended**: Use the ``ogb`` command for all operations::

    # Automated benchmarking
    ogb autobench --model yangheng/OmniGenome-186M --benchmark RGB

    # Automated training
    ogb autotrain --dataset yangheng/tfb_promoters --model zhihan1996/DNABERT-2-117M

    # Automated inference
    ogb autoinfer --model yangheng/ogb_tfb_finetuned --sequence "ATCGATCGATCG"

Alternative Entry Point
~~~~~~~~~~~~~~~~~~~~~~~

Using the Python module (legacy)::

    # RNA design (use ogb rna_design instead, recommended)
    python -m omnigenbench.cli.omnigenome_cli rna_design --structure "(((...)))"

    # AutoInfer (use ogb autoinfer instead, recommended)
    python -m omnigenbench.cli.omnigenome_cli autoinfer --model yangheng/ogb_tfb_finetuned --sequence "ATCG"

Command Overview
----------------

AutoBench - Automated Benchmarking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Benchmark genomic foundation models on standard datasets::

    ogb autobench --model MODEL_NAME --benchmark BENCHMARK_NAME [OPTIONS]

**Key Arguments**:

- ``--model, -m`` (required): Model name or path (HF model ID or local path)
- ``--benchmark, -b`` (required): Benchmark name (RGB, PGB, GUE, GB, BEACON)
- ``--tokenizer, -t``: Tokenizer to use (default: same as model)
- ``--trainer``: Trainer type (default: accelerate)
- ``--bs_scale``: Batch size scale factor for GPU memory utilization
- ``--overwrite``: Overwrite existing results

**Examples**::

    # Benchmark OmniGenome on RGB
    ogb autobench --model yangheng/OmniGenome-186M --benchmark RGB

    # Benchmark DNABERT on GUE with custom trainer
    ogb autobench --model zhihan1996/DNABERT-2-117M --benchmark GUE --trainer accelerate

    # Benchmark with increased batch size
    ogb autobench --model yangheng/OmniGenome-186M --benchmark BEACON --bs_scale 4

AutoTrain - Automated Training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Automatically train or fine-tune models on genomic datasets::

    ogb autotrain --dataset DATASET_NAME --model MODEL_NAME [OPTIONS]

**Key Arguments**:

- ``--dataset`` (required): Dataset name or path
- ``--model`` (required): Pre-trained model name or path
- ``--tokenizer``: Tokenizer to use (default: same as model)
- ``--output-dir``: Directory to save fine-tuned model
- ``--num-epochs``: Number of training epochs
- ``--batch-size``: Training batch size
- ``--learning-rate``: Learning rate
- ``--trainer``: Trainer type (default: accelerate)
- ``--overwrite``: Overwrite existing output directory

**Examples**::

    # Basic training
    ogb autotrain --dataset yangheng/tfb_promoters --model zhihan1996/DNABERT-2-117M

    # Training with custom parameters
    ogb autotrain \\
        --dataset ./my_dataset \\
        --model yangheng/OmniGenome-186M \\
        --num-epochs 10 \\
        --batch-size 32 \\
        --learning-rate 5e-5 \\
        --output-dir ./my_model

AutoInfer - Automated Inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run inference with fine-tuned models on arbitrary sequences::

    ogb autoinfer --model MODEL_NAME [--sequence SEQ | --input-file FILE] [OPTIONS]

**Key Arguments**:

- ``--model`` (required): Fine-tuned model name or path
- ``--sequence``: Input sequence(s) (single sequence, comma-separated, or .txt file)
- ``--input-file``: Path to JSON/CSV file with input data
- ``--output-file``: Output file for predictions (default: inference_results.json)
- ``--batch-size``: Batch size for inference (default: 32)
- ``--device``: Device to use (e.g., 'cuda:0', 'cpu'; auto-detected if not specified)

**Input Formats**:

- **Single sequence**: ``--sequence "ATCGATCGATCG"``
- **Multiple sequences**: ``--sequence "ATCG,CGTA,TACG"``
- **Text file**: ``--sequence sequences.txt`` (one per line)
- **JSON file**: ``--input-file data.json``

  - Format 1: ``{"sequences": ["ATCG", "CGTA"]}``
  - Format 2: ``{"data": [{"sequence": "ATCG", "id": 1}, ...]}``
  - Format 3: ``["ATCG", "CGTA"]``

- **CSV file**: ``--input-file data.csv`` (must have 'sequence' column)

**Examples**::

    # Single sequence inference
    ogb autoinfer --model yangheng/ogb_tfb_finetuned --sequence "ATCGATCGATCG"

    # Multiple sequences
    ogb autoinfer --model yangheng/ogb_te_finetuned --sequence "ATCG,CGTA,TACG"

    # Batch inference from JSON
    ogb autoinfer --model yangheng/ogb_tfb_finetuned --input-file sequences.json --batch-size 64

    # CSV input with metadata
    ogb autoinfer --model yangheng/ogb_tfb_finetuned --input-file data.csv --device cuda:0

    # Text file input
    ogb autoinfer --model yangheng/ogb_tfb_finetuned --sequence sequences.txt --output-file results.json

RNA Design
~~~~~~~~~~

Design RNA sequences that fold into specific secondary structures using genetic algorithms combined with masked language modeling::

    ogb rna_design --structure STRUCTURE [OPTIONS]

**Algorithm Overview**:

The RNA design command uses an evolutionary algorithm enhanced with masked language modeling (MLM) to generate RNA sequences that fold into target secondary structures. The process involves:

1. **Initialization**: Generate initial population using MLM-guided sequence generation
2. **Evolution**: Iteratively improve sequences through:
   
   - Crossover: Multi-point recombination between parent sequences
   - Mutation: MLM-guided mutations to explore sequence space
   - Selection: Multi-objective optimization using NSGA-II-like sorting
   
3. **Evaluation**: Structure prediction using ViennaRNA with fitness scoring based on:
   
   - Structure similarity (Hamming distance to target)
   - Thermodynamic stability (Minimum Free Energy)

**Key Arguments**:

- ``--structure`` (required): Target RNA structure in dot-bracket notation (e.g., "(((...)))")
  
  - Use ``(`` for opening pairs, ``)`` for closing pairs, ``.`` for unpaired bases
  - Structure length determines sequence length
  - Example simple hairpin: ``"(((...)))`` (8 nucleotides)
  - Example complex structure: ``"(((..(((...)))..)))`` (18 nucleotides)

- ``--model``: Pre-trained model name or path (default: yangheng/OmniGenome-186M)
  
  - Can use HuggingFace model IDs or local paths
  - Larger models may produce better results but run slower
  - Recommended: yangheng/OmniGenome-186M for balance of speed and quality

- ``--mutation-ratio``: Mutation ratio for genetic algorithm (0.0-1.0, default: 0.5)
  
  - Controls the fraction of nucleotides mutated per generation
  - Lower values (0.1-0.3): More conservative, slower convergence
  - Higher values (0.5-0.8): More exploration, may be unstable
  - Recommended: Start with 0.5, decrease if not converging

- ``--num-population``: Population size (default: 100)
  
  - Number of candidate sequences per generation
  - Larger populations explore more solutions but run slower
  - Recommended: 100-500 depending on structure complexity
  - Simple structures (< 20 nt): 100 is sufficient
  - Complex structures (> 50 nt): Consider 200-500

- ``--num-generation``: Number of generations (default: 100)
  
  - Maximum evolutionary iterations
  - Algorithm terminates early if perfect solution found
  - Recommended: 50-200 depending on difficulty
  - Monitor output to see if more generations are needed

- ``--output-file``: Output JSON file to save results
  
  - Saves designed sequences and parameters
  - JSON format includes structure, parameters, and best sequences
  - If not specified, only prints to console

**Output Format**:

The command outputs designed sequences to console and optionally saves to JSON file::

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
        "GCUAAAGCC",
        ...
      ]
    }

**Examples**::

    # Basic RNA design for simple hairpin
    ogb rna_design --structure "(((...)))"

    # Design with custom parameters for better results
    ogb rna_design \
        --structure "(((...)))" \
        --model yangheng/OmniGenome-186M \
        --mutation-ratio 0.3 \
        --num-population 200 \
        --num-generation 150 \
        --output-file hairpin_design.json

    # Design complex structure (stem-loop-stem)
    ogb rna_design \
        --structure "(((..(((...)))..)))" \
        --num-population 300 \
        --num-generation 200 \
        --output-file complex_design.json

    # Fast design with smaller population
    ogb rna_design \
        --structure "((((....))))" \
        --num-population 50 \
        --num-generation 50 \
        --mutation-ratio 0.6

**Legacy Command** (still supported)::

    python -m omnigenbench.cli.omnigenome_cli rna_design --structure "(((...)))"

**Performance Tips**:

1. **Start Simple**: Test with small structures first to verify functionality
2. **Monitor Progress**: The algorithm shows progress with tqdm progress bar
3. **Early Termination**: Algorithm stops when perfect match is found (Hamming distance = 0)
4. **GPU Acceleration**: Uses GPU automatically if available for MLM inference
5. **Parallel Folding**: Enable with ``parallel=True`` in Python API for faster structure evaluation

**Troubleshooting**:

- **Poor Convergence**: Try lowering mutation ratio (0.2-0.3) or increasing population
- **Too Slow**: Reduce population/generations or use smaller model
- **No Perfect Match**: Some structures are difficult; best sequence is still returned
- **Out of Memory**: Reduce batch size in MLM inference or use CPU
- **Invalid Structure**: Ensure balanced parentheses in dot-bracket notation

**Common Structure Patterns**:

- Hairpin: ``"(((...)))"`
- Stem-Loop-Stem: ``"(((..(((...)))..)))"`
- Multi-loop: ``"(((...(((...)))..(((...))).)))"`
- Pseudoknot: Not supported (dot-bracket notation limitation)

**Related Python API**::

    from omnigenbench import OmniModelForRNADesign
    
    model = OmniModelForRNADesign(model="yangheng/OmniGenome-186M")
    sequences = model.design(
        structure="(((...)))",
        mutation_ratio=0.5,
        num_population=100,
        num_generation=100
    )
    print("Designed sequences:", sequences)

Available Benchmarks
--------------------

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Benchmark
     - Description
     - Key Tasks
   * - **RGB**
     - RNA Genomic Benchmark
     - RNA structure prediction, stability, interactions
   * - **PGB**
     - Protein Genomic Benchmark
     - Protein function, structure, interaction prediction
   * - **GUE**
     - Genomics Understanding Evaluation
     - Comprehensive genomic understanding tasks
   * - **GB**
     - General Benchmark
     - General genomic prediction tasks
   * - **BEACON**
     - Comprehensive Benchmark Suite
     - Long-range dependencies, context understanding

Trainer Types
-------------

All training and benchmarking commands support multiple trainer backends:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Trainer
     - Description
     - Use Case
   * - **accelerate**
     - HuggingFace Accelerate (default)
     - Multi-GPU distributed training
   * - **native**
     - Native PyTorch trainer
     - Single-GPU, debugging
   * - **hf_trainer**
     - HuggingFace Trainer
     - Full HF ecosystem integration

Example::

    # Use Accelerate trainer (recommended for multi-GPU)
    ogb autotrain --dataset data --model model --trainer accelerate

    # Use native trainer (for single-GPU)
    ogb autotrain --dataset data --model model --trainer native

Best Practices
--------------

✅ **Recommended Workflows**

1. **Model Benchmarking**::

    # Benchmark model on standard dataset
    ogb autobench --model yangheng/OmniGenome-186M --benchmark RGB

2. **Model Fine-tuning**::

    # Fine-tune on custom dataset
    ogb autotrain --dataset ./my_data --model yangheng/OmniGenome-186M --num-epochs 10

3. **Inference Pipeline**::

    # Inference on new sequences
    ogb autoinfer --model ./my_finetuned_model --input-file new_data.csv

❌ **Common Pitfalls**

- Not providing either ``--sequence`` or ``--input-file`` for autoinfer
- Using invalid benchmark names
- Forgetting to specify output directory when fine-tuning
- Using batch size too large for GPU memory

Detailed API Documentation
---------------------------

Below are the detailed API references for each CLI component.

OGB CLI (Unified Entry Point)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: omnigenbench.cli.ogb_cli
    :members:
    :undoc-members:
    :show-inheritance:
    :noindex:

The ``ogb`` command is the recommended unified entry point for all OmniGenBench CLI operations. It provides three main subcommands:

- ``autobench``: Automated benchmarking
- ``autotrain``: Automated training
- ``autoinfer``: Automated inference

**Usage**::

    ogb {autobench,autotrain,autoinfer} [options]

OmniGenome CLI (Alternative Entry Point)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: omnigenbench.cli.omnigenome_cli
    :members:
    :undoc-members:
    :show-inheritance:
    :noindex:

Alternative entry point via Python module. Supports:

- ``rna_design``: RNA sequence design
- ``autoinfer``: Model inference (also available via ogb)

**Usage**::

    python -m omnigenbench.cli.omnigenome_cli {rna_design,autoinfer} [options]

Base Commands
-------------

.. automodule:: omnigenbench.cli.commands.base
    :members:
    :undoc-members:
    :show-inheritance:
    :noindex:

The ``BaseCommand`` abstract class provides a common interface for all CLI commands. All command classes inherit from this base and implement the ``register_command`` method.

**Key Methods**:

- ``register_command(cls, subparsers)``: Register command with argparse
- ``add_common_arguments(cls, parser)``: Add common CLI arguments

Bench Commands
--------------

.. automodule:: omnigenbench.cli.commands.bench.bench_cli
    :members:
    :undoc-members:
    :show-inheritance:
    :noindex:

The ``BenchCommand`` class implements automated benchmarking functionality. It evaluates genomic foundation models on standard benchmark datasets.

**Supported Benchmarks**:

- RGB (RNA Genomic Benchmark)
- PGB (Protein Genomic Benchmark)
- GUE (Genomics Understanding Evaluation)
- GB (General Benchmark)
- BEACON (Comprehensive benchmark suite)

**Example**::

    # Basic benchmarking
    ogb autobench --model yangheng/OmniGenome-186M --benchmark RGB
    
    # With custom settings
    ogb autobench \\
        --model yangheng/OmniGenome-186M \\
        --benchmark RGB \\
        --trainer accelerate \\
        --bs_scale 2 \\
        --overwrite

RNA Commands
------------

.. automodule:: omnigenbench.cli.commands.rna.rna_design
    :members:
    :undoc-members:
    :show-inheritance:
    :noindex:

The ``RNADesignCommand`` class provides RNA sequence design functionality using genetic algorithms.

**Design Process**:

1. Load pre-trained RNA design model
2. Run genetic algorithm optimization
3. Generate sequences matching target structure
4. Save results (optional)

**Example**::

    # Basic design (recommended)
    ogb rna_design \
        --structure "(((...)))"
    
    # Advanced design
    ogb rna_design \
        --structure "(((...)))" \
        --model yangheng/OmniGenome-186M \
        --mutation-ratio 0.3 \
        --num-population 200 \
        --num-generation 150 \
        --output-file results.json
    
    # Legacy command (still supported)
    python -m omnigenbench.cli.omnigenome_cli rna_design --structure "(((...)))"

Advanced Usage Examples
-----------------------

Batch Processing Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~

Complete pipeline from training to inference::

    # Step 1: Fine-tune model
    ogb autotrain \\
        --dataset yangheng/tfb_promoters \\
        --model yangheng/OmniGenome-186M \\
        --num-epochs 10 \\
        --output-dir ./tfb_model
    
    # Step 2: Benchmark the fine-tuned model
    ogb autobench \\
        --model ./tfb_model \\
        --benchmark RGB
    
    # Step 3: Run inference on new data
    ogb autoinfer \\
        --model ./tfb_model \\
        --input-file new_sequences.csv \\
        --output-file predictions.json

Multi-GPU Training
~~~~~~~~~~~~~~~~~~

Using Accelerate for distributed training::

    # Configure Accelerate (one-time setup)
    accelerate config
    
    # Launch distributed training
    accelerate launch -m omnigenbench.cli.ogb_cli autotrain \\
        --dataset yangheng/tfb_promoters \\
        --model yangheng/OmniGenome-186M \\
        --trainer accelerate
    
    # Or use torchrun
    torchrun --nproc_per_node=4 -m omnigenbench.cli.ogb_cli autotrain \\
        --dataset data \\
        --model model

Inference with Different Input Formats
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**JSON Format**::

    # sequences.json
    {
      "sequences": [
        "ATCGATCGATCG",
        "CGTAGCTAGCTA"
      ]
    }
    
    # Run inference
    ogb autoinfer --model model --input-file sequences.json

**CSV Format with Metadata**::

    # data.csv
    # sequence,gene_id,organism
    # ATCGATCGATCG,GENE001,human
    # CGTAGCTAGCTA,GENE002,mouse
    
    # Run inference (preserves metadata)
    ogb autoinfer --model model --input-file data.csv

**Complex JSON with Metadata**::

    # complex_data.json
    {
      "data": [
        {"sequence": "ATCG", "id": 1, "type": "promoter"},
        {"sequence": "CGTA", "id": 2, "type": "enhancer"}
      ]
    }
    
    # Run inference
    ogb autoinfer --model model --input-file complex_data.json

Custom Benchmark Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    # Run benchmark with custom batch size and trainer
    ogb autobench \\
        --model yangheng/OmniGenome-186M \\
        --benchmark BEACON \\
        --trainer accelerate \\
        --bs_scale 4 \\
        --overwrite
    
    # Results will be saved to the benchmark directory
    # with model name and timestamp

RNA Design Workflow
~~~~~~~~~~~~~~~~~~~

Complete workflow for designing RNA sequences for multiple target structures::

    # Design multiple structures in batch (recommended: use ogb command)
    for structure in "(((...)))" "(((..)))" "((((....))))"; do
        ogb rna_design \
            --structure "$structure" \
            --num-population 200 \
            --num-generation 200 \
            --output-file "design_${structure//[().]/_}.json"
    done
    
    # Or using a structures file
    while IFS= read -r structure; do
        echo "Designing for: $structure"
        ogb rna_design \
            --structure "$structure" \
            --num-population 300 \
            --num-generation 150 \
            --output-file "design_$(echo $structure | md5sum | cut -d' ' -f1).json"
    done < structures.txt

**Advanced: Python Script for Batch Design**::

    from omnigenbench import OmniModelForRNADesign
    import json
    from pathlib import Path
    
    # Define target structures
    structures = [
        "(((...)))",           # Simple hairpin
        "(((..(((...)))..)))", # Stem-loop-stem
        "(((..(((...)))..(((...))).)))",  # Multi-loop
    ]
    
    # Initialize model once for efficiency
    model = OmniModelForRNADesign(model="yangheng/OmniGenome-186M")
    
    # Design sequences for each structure
    results = {}
    for structure in structures:
        print(f"Designing sequences for: {structure}")
        sequences = model.design(
            structure=structure,
            mutation_ratio=0.5,
            num_population=200,
            num_generation=150
        )
        results[structure] = sequences
        print(f"  Found {len(sequences) if isinstance(sequences, list) else 1} solutions")
    
    # Save all results
    Path("batch_design_results.json").write_text(
        json.dumps(results, indent=2)
    )
    print(f"\\nAll results saved to batch_design_results.json")

Troubleshooting
---------------

Common Issues and Solutions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Issue**: "Either --sequence or --input-file must be provided for inference"

**Solution**: Always provide input data for autoinfer::

    # Correct
    ogb autoinfer --model model --sequence "ATCG"
    ogb autoinfer --model model --input-file data.json
    
    # Wrong
    ogb autoinfer --model model  # Missing input!

**Issue**: CUDA out of memory during benchmarking

**Solution**: Reduce batch size or use gradient accumulation::

    # Reduce batch size
    ogb autobench --model model --benchmark RGB --bs_scale 1
    
    # Or use smaller model
    ogb autobench --model smaller_model --benchmark RGB

**Issue**: "Invalid benchmark name"

**Solution**: Use correct benchmark names::

    # Correct
    ogb autobench --model model --benchmark RGB
    ogb autobench --model model --benchmark GUE
    
    # Wrong
    ogb autobench --model model --benchmark rgb  # Case sensitive!

**Issue**: Model not found

**Solution**: Verify model path/name::

    # HuggingFace Hub model
    ogb autoinfer --model yangheng/OmniGenome-186M --sequence "ATCG"
    
    # Local model (use absolute path)
    ogb autoinfer --model /path/to/model --sequence "ATCG"

Performance Tips
~~~~~~~~~~~~~~~~

1. **Use appropriate batch size**::

    # For large GPU memory
    ogb autobench --model model --benchmark RGB --bs_scale 4
    
    # For limited memory
    ogb autobench --model model --benchmark RGB --bs_scale 1

2. **Enable mixed precision**::

    # Automatically enabled with accelerate trainer
    ogb autotrain --dataset data --model model --trainer accelerate

3. **Parallel inference**::

    # Increase batch size for faster inference
    ogb autoinfer --model model --input-file large_data.csv --batch-size 128

4. **GPU selection**::

    # Specify GPU device
    ogb autoinfer --model model --sequence "ATCG" --device cuda:0

Environment Variables
~~~~~~~~~~~~~~~~~~~~~

OmniGenBench respects the following environment variables::

    # HuggingFace cache directory
    export HF_HOME=/path/to/cache
    
    # CUDA device selection
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    
    # Disable warnings
    export PYTHONWARNINGS=ignore

Output Formats
--------------

AutoInfer Output
~~~~~~~~~~~~~~~~

The autoinfer command produces JSON output with the following structure::

    {
      "model": "model_name",
      "total_sequences": 100,
      "results": [
        {
          "sequence": "ATCGATCGATCG",
          "metadata": {"index": 0},
          "predictions": [0.8, 0.2],
          "probabilities": [0.8, 0.2],
          "logits": [1.5, -0.3]
        },
        ...
      ]
    }

**Fields**:

- ``model``: Model used for inference
- ``total_sequences``: Total number of sequences processed
- ``results``: List of per-sequence results
  
  - ``sequence``: Input sequence
  - ``metadata``: Original metadata from input
  - ``predictions``: Model predictions
  - ``probabilities``: Class probabilities (if available)
  - ``logits``: Raw logits (if available)
  - ``error``: Error message (if processing failed)

RNA Design Output
~~~~~~~~~~~~~~~~~

RNA design produces JSON with the following structure::

    {
      "structure": "(((...)))",
      "parameters": {
        "mutation_ratio": 0.5,
        "population": 100,
        "generations": 100
      },
      "best_sequences": [
        "GCGAAACGC",
        "GCUAAAGCC",
        ...
      ]
    }

See Also
--------

Related Documentation
~~~~~~~~~~~~~~~~~~~~~

- :doc:`../usage` - Basic usage guide
- :doc:`../cli` - CLI usage examples
- :doc:`trainers` - Trainer documentation
- :doc:`datasets` - Dataset documentation
- :doc:`models` - Model documentation

External Resources
~~~~~~~~~~~~~~~~~~

- `HuggingFace Hub <https://huggingface.co/yangheng>`_ - Pre-trained models
- `GitHub Repository <https://github.com/yangheng95/OmniGenBench>`_ - Source code
- `Paper <https://arxiv.org/abs/XXX>`_ - Research paper (if available)

Getting Help
~~~~~~~~~~~~

If you encounter issues or have questions:

1. Check the `documentation <https://omnigenbench.readthedocs.io>`_
2. Search `GitHub Issues <https://github.com/yangheng95/OmniGenBench/issues>`_
3. Open a new issue with:
   
   - Command used
   - Full error message
   - OmniGenBench version (``pip show omnigenbench``)
   - Python version
   - CUDA version (if using GPU)
