.. _cli:

#######################
Command Line Interface
#######################

.. rst-class:: lead

   OmniGenBench comes with a powerful and intuitive command-line interface (CLI) that allows you to run complex workflows like benchmarking, training, and RNA design directly from your terminal, without writing any Python code.

This page provides an overview of the most common commands. For a full list of all available commands and arguments, you can always run:

.. code-block:: bash

   omnigenbench --help


**************************************
Automatic Benchmarking Command
**************************************

The ``autobench`` command is your primary tool for evaluating a model's performance on a standard benchmark dataset. It automates everything from data loading to metric calculation.

**Usage**

.. code-block:: bash

   omnigenbench autobench --model <model_name_or_path> --benchmark <benchmark_name>

**Example**

Here's how to evaluate the ``yangheng/OmniGenome-186M`` model on the ``RGB`` benchmark:

.. code-block:: bash

   omnigenbench autobench --model yangheng/OmniGenome-186M --benchmark RGB

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