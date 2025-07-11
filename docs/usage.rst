Basic Usage
===========

This page introduces common usage patterns of **OmniGenBench**, including automatic benchmarking, model training, inference, and resource downloads.

Automatic Benchmarking
-----------------------

.. code-block:: python

   from omnigenome import AutoBench
   bench = AutoBench("RGB", "yangheng/OmniGenome-186M")
   bench.run()

Model Training
--------------

.. code-block:: python

   from omnigenome import AutoTrain
   trainer = AutoTrain("Dataset", "yangheng/OmniGenome-186M")
   trainer.run()

Model Inference
---------------

.. code-block:: python

   from omnigenome import OmniModelForSequenceClassification
   model = OmniModelForSequenceClassification("yangheng/OmniGenome-186M")
   result = model.predict("ACGUAGGUAUCGUAGA")
   print(result)

Downloading Benchmarks (Datasets)
-------------------------------

.. code-block:: python

   from omnigenome.utility.hub_utils import download_model, download_benchmark
   download_model("OmniGenome-186M-SSP")
   download_benchmark("RGB")

For advanced usage, refer to the API Reference and Command Line Usage sections.
