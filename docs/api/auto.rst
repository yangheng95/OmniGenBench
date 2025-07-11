Auto-Pipelines
==============

This section documents the automated pipelines provided by OmniGenomeBench, which include AutoBench and AutoTrain. These tools are designed to streamline benchmarking and training workflows for genomic foundation models, making it easy for users to evaluate and train models with minimal manual setup.

OmniGenomeBench's auto-pipelines offer:
- Automated benchmarking of genomic models using standardized protocols and datasets.
- Automated training workflows with flexible configuration options.
- Integration with model, dataset, and pipeline hubs for easy access to resources.
- Command-line interfaces for running benchmarks and training directly from the terminal.

**Key Components:**
- `AutoBench`: Automates the benchmarking process for genomic models.
- `AutoTrain`: Automates the training process for genomic models.
- `BenchHub`, `ModelHub`, `PipelineHub`: Provide access to benchmarks, models, and pipelines.
- CLI commands: Allow users to run benchmarks and training via the command line.

**Example Usage:**
.. code-block:: python

    from omnigenome import AutoBench, AutoTrain

    # Run automated benchmarking
    bench = AutoBench("RGB", "model_name")
    bench.run()

    # Train a model
    trainer = AutoTrain("RGB", "model_name")
    trainer.run()

Refer to the API documentation below for details on each auto-pipeline component, including configuration, usage, and extension options.

Auto Bench
----------

.. automodule:: omnigenome.auto.auto_bench.auto_bench
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:
    :noindex:

.. automodule:: omnigenome.auto.auto_bench.auto_bench_cli
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:
    :noindex:

.. automodule:: omnigenome.auto.auto_bench.auto_bench_config
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:
    :noindex:

.. automodule:: omnigenome.auto.auto_bench.config_check
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:
    :noindex:

Auto Train
----------

.. automodule:: omnigenome.auto.auto_train.auto_train
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:
    :noindex:

.. automodule:: omnigenome.auto.auto_train.auto_train_cli
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:
    :noindex:

Bench Hub
---------

.. automodule:: omnigenome.auto.bench_hub.bench_hub
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:
    :noindex:
