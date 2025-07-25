.. CLI Commands
.. ============

This section provides detailed documentation for the command-line interface (CLI) commands available in OmniGenBench. The CLI allows users to run benchmarks, train models, and manage resources directly from the terminal, making it easy to perform common tasks without writing Python scripts.

**Overview:**

- OmniGenBench's CLI is designed for codeless operation, enabling users to quickly execute benchmarking, training, and resource management workflows.
- The CLI covers base commands, main entry points, benchmarking commands, and specialized commands for RNA and other tasks.
- Each command is documented with its available options, arguments, and usage examples.

**Key CLI Components:**

- Base Commands: Core commands for general operations and configuration.
- Main CLI: The main entry point for OmniGenBench's command-line interface.
- Bench Commands: Commands for running automated benchmarks on genomic models and datasets.
- RNA Commands: Specialized commands for RNA-related tasks and workflows.

**Auto-Pipeline CLI Example:**

.. code-block:: bash

    omnigenbench bench --model model_name --dataset dataset_name
    omnigenbench train --model model_name --dataset dataset_name

Refer to the API documentation below for details on each CLI command, including available options and usage instructions.

Base Commands
-------------

.. automodule:: omnigenbench.cli.commands.base
    :members:
    :undoc-members:
    :show-inheritance:
    :noindex:

Main CLI
--------

.. automodule:: omnigenbench.cli.omnigenome_cli
    :members:
    :undoc-members:
    :show-inheritance:
    :noindex:

Bench Commands
--------------

.. automodule:: omnigenbench.cli.commands.bench.bench_cli
    :members:
    :undoc-members:
    :show-inheritance:
    :noindex:

RNA Commands
------------

.. automodule:: omnigenbench.cli.commands.rna.rna_design
    :members:
    :undoc-members:
    :show-inheritance:
    :noindex:
