Command Line Usage
==================
.. code-block:: text

       **  +----------- **           ___                     _
      @@                 @@         / _ \  _ __ ___   _ __  (_)
     @@* #============== *@@       | | | || '_ ` _ \ | '_ \ | |
     @@*                 *@@       | |_| || | | | | || | | || |
     *@@  +------------ *@@         \___/ |_| |_| |_||_| |_||_|
      *@*               @@*
       *@@  #========= @@*
        *@@*         *@@*
          *@@  +---@@@*              ____
            *@@*   **               / ___|  ___  _ __    ___   _ __ ___    ___
              **@**                | |  _  / _ \| '_ \  / _ \ | '_ ` _ \  / _ \
            *@@* *@@*              | |_| ||  __/| | | || (_) || | | | | ||  __/
          *@@ ---+  @@*             \____| \___||_| |_| \___/ |_| |_| |_| \___|
        *@@*         *@@*
       *@@ =========#  @@*
      *@@               @@*
     *@@ -------------+  @@*        ____                      _
     @@                   @@       | __ )   ___  _ __    ___ | |__
     @@ ===============#  @@       |  _ \  / _ \| '_ \  / __|| '_ \
      @@                 @@        | |_) ||  __/| | | || (__ | | | |
       ** -----------+  **         |____/  \___||_| |_| \___||_| |_|

OmniGenBench provides a rich set of command-line tools for batch evaluation, training, and RNA design.



Automatic Benchmarking Command
-----------------------------

Example:

.. code-block:: bash

   $ omnigenome autobench --model yangheng/OmniGenome-186M --benchmark RGB

Arguments:

--model, -m
    Required. Model to evaluate

--benchmark, -b
    Benchmark dataset: RGB, PGB, BEACON, etc.

--tokenizer, -t
    Specify tokenizer

--trainer
    Trainer backend: native, accelerate, hf_trainer

--overwrite
    Overwrite existing results if present

--bs_scale
    Batch size scaling factor

RNA Structure Design Command
---------------------------

Example:

.. code-block:: bash

   $ omnigenome rna_design --structure "(((...)))" --model yangheng/OmniGenome-186M

Arguments:

--structure
    Target RNA secondary structure (dot-bracket notation)

--model
    Model to use for design

--mutation-ratio
    Mutation rate for the genetic algorithm

--num-population
    Population size

--num-generation
    Number of evolutionary generations

--output-file
    Path to save the output results

For more commands and options:

.. code-block:: bash

   $ omnigenome --help
