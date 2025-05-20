# OmniGenBench Framework Module Appendix

> This appendix provides a detailed, markdown-formatted overview of each core and auxiliary module in **OmniGenBench**. For each module you will find:
>
> - **Functionality Description**  
> - **Inputs / Outputs Definition**  
> - **Interface Introduction**  
> - **Source Code Link**

## Interface Demos

Below are interactive HTML demos hosted on GitHub Pages. Click the links to explore each functionality in action.

- [🚀 AutoBench Tutorial Demo](https://colalab.ai/OmniGenBench/docs/interface_demos/autobench.html)
- [⚙️ Metric Customization Demo](https://colalab.ai/OmniGenBench/docs/interface_demos/metric_custmization.html)
- [🔄 Model Wrapper Demo](https://colalab.ai/OmniGenBench/docs/interface_demos/model_wrapper.html)
- [🧾 Task Definition Demo](https://colalab.ai/OmniGenBench/docs/interface_demos/task_definition.html)
- [🔬 Convert to Tensor Demo](https://colalab.ai/OmniGenBench/docs/interface_demos/convert_to_tensor.html)
- [🧬 RNA Augmentation Demo](https://colalab.ai/OmniGenBench/docs/interface_demos/rna_augmentation.html)
- [🧠 RNA Design Demo](https://colalab.ai/OmniGenBench/docs/interface_demos/rna_design.html)


## Interface Details

| Modules                                                | Inputs                                                               | Outputs                            | Interface                                                   |
|--------------------------------------------------------|----------------------------------------------------------------------|------------------------------------|-------------------------------------------------------------|
| [Data Parsing](#data-parsing)                          | Raw files (`.fasta`/`.fa`/`.json`); metadata (species, tags, splits) | `dict` of sequences + metadata     | `AbstractDataset.load_from_file(data_file)`                 |
| [Adaptive Truncation & Padding](#adaptive-truncation-padding) | Sequence strings; target length                                      | Truncated or padded sequence       | `AbstractDataset._pad_and_truncate(tensor_data)`           |
| [Instance Filtering (CD-HIT-EST)](#instance-filtering-cd-hit-est) | Parsed sequence list; similarity threshold                           | Filtered sequence list             | `cd-hit-est CI`                                             |
| [MLM Augmentation](#mlm-augmentation)                  | Sequence tensors; masking ratio                                      | Augmented sequence tensors         | `OmniGenomeModelForAugmentation.augment_sequence(sequence)` |
| [Data Hub](#data-hub)                                  | Dataset cards; upload packages                                       | Public dataset URLs; metadata pages | HF Space API                                              |
| [Base Model Template](#base-model-template)            | Token IDs; attention masks                                           | Hidden states; logits              | `BaseModel.forward(tensor_data)`                            |
| [Model Wrappers](#model-wrappers)                      | HF model instance                                                    | Wrapped `BaseModel` subclass       | `ClassificationModelWrapper`                                |
| [Tokenizer Wrappers](#tokenizer-wrappers)              | Raw sequence strings                                                 | Token ID tensors                   | `TokenizerWrapper.encode()` / `decode()`                    |
| [Trainer Backends](#trainer-backends)                  | Model; dataset; hyperparameters                                      | Checkpoints; logs                  | `Trainer.train()` / `AccelerateTrainer.train()` / `HFTrainer.train()` |
| [Evaluation & Inference API](#evaluation-inference-api) | Validation dataset; batch size                                       | Metric dict; prediction arrays     | `Trainer.evaluate(test_set)` / `model.inference(inference_set)` |
| [Model Hub Integration](#model-hub-integration)        | Model ID string                                                      | Downloaded weights; config         | `ModelHub.load(model_name_or_path)`                         |
| [Task Compiling](#task-compiling)                      | Task list; metadata file                                             | Benchmark suite object             | `Benchmark.compile()`                                       |
| [Automated Benchmarking](#automated-benchmarking)      | Model ID; suite object                                               | Raw metrics per task               | `autobench --benchmark <suite> --model <model_id>`          |
| [Metrics Configuration](#metrics-configuration)        | Config file path                                                     | Metric registry entries            | Internal JSON loader                                        |
| [Reporting & Leaderboard Submission](#reporting-leaderboard-submission) | Metrics CSV; HF Space credentials                  | Leaderboard entries                | `autobench --report`                                        |
| [Task Registration](#task-registration)                | `config.py`                                                          | `Task` object in registry          | `AutoBenchConfig(config_dict)`                              |
| [Dataset Processing](#dataset-processing)              | Selected Data Module pipeline                                        | `DataLoader`                       | `dataset_fn`                                                |
| [Model Composition](#model-composition)                | `BaseModel` class; head specs                                        | Composite model instance           | e.g. `model.set_loss_fn(torch.CrossEntropy())`              |
| [Task-Specific Configuration](#task-specific-configuration) | Loss name; metric list; optimizer settings                           | Initialized training components    | `loss_fn` / `metric_fns`                                    |
| [Pipeline Definitions](#pipeline-definitions)          | Training scripts or notebook paths                                   | Executable pipelines               | `Trainer.train().save_model()` / `ModelHub.load(...).inference()` |
| [Classification & Ranking Metrics](#classification-ranking-metrics) | `y_true`; `y_pred`                                       | Scalar or dict of scores           | `metric_registry["accuracy"]()`                             |
| [Regression Metrics](#regression-metrics)              | `y_true`; `y_pred`                                                   | Scalar or dict of scores           | `metric_registry["mse"]()`                                  |
| [Distance & Similarity Metrics](#distance-similarity-metrics) | Embedding pairs or sequence pairs                                    | Distance/similarity values         | `metric_registry["cosine"]()`                               |
| [Custom Genomic Metrics](#custom-genomic-metrics)      | Domain-specific outputs (e.g. structure alignments)                  | Metric values                      | Defined in task config                                      |
| [Sequence-Level Motif Analysis](#sequence-level-motif-analysis) | Attention maps; sequence motif analysis                              | heatmaps; Motif logos              | `MotifScanner.scan()`                                       |
| [Embedding Space Analysis](#embedding-space-analysis)  | Embedding arrays; labels                                             | Scatter plots; interactive HTML    | `EmbeddingProjector.project()`                              |
| [Attention Map Visualization](#attention-map-visualization) | Attention tensors; sequence tokens                                   | Attention heatmaps                 | `AttentionPlotter.plot()`                                   |

---

## Anchor Definitions

<a id="data-parsing"></a>
### Data parsing
Read multi-format genomic data (FASTA, JSON, CSV, Pandas, etc.), validate metadata tags, and convert to in-memory records.  
[Code](https://github.com/COLA-Laboratory/OmniGenBench/blob/9ecd8aa0b93d744a6894775046b8fcce3b7b5fbc/omnigenome/src/abc/abstract_dataset.py#L281)

<a id="adaptive-truncation-padding"></a>
### Adaptive truncation & padding
Dynamically shorten or pad sequences to model-compatible lengths while preserving key regions.  
[Code](https://github.com/COLA-Laboratory/OmniGenBench/blob/9ecd8aa0b93d744a6894775046b8fcce3b7b5fbc/omnigenome/src/abc/abstract_dataset.py#L182)

<a id="instance-filtering-cd-hit-est"></a>
### Instance filtering (CD-HIT-EST)
Remove near-duplicate sequences to prevent label leakage in structure tasks.  
[Code](https://bioinformatics.org/cd-hit/)

<a id="mlm-augmentation"></a>
### MLM augmentation
Mask and predict random nucleotides to generate synthetic variants for robustness.  
[Code](https://github.com/COLA-Laboratory/OmniGenBench/blob/9ecd8aa0b93d744a6894775046b8fcce3b7b5fbc/examples/tutorials/RNA_Augmentation_Tutorial.ipynb)

<a id="data-hub"></a>
### Data hub
Index and serve benchmark datasets via Hugging Face Spaces.  
[Code](https://huggingface.co/spaces/yangheng/OmniGenomeLeaderboard)

<a id="base-model-template"></a>
### Base model template
Define a minimal GFM interface (`forward`, `predict`, `inference`, `load`, `save`) for heterogeneous architectures.  
[Code](https://github.com/COLA-Laboratory/OmniGenBench/blob/9ecd8aa0b93d744a6894775046b8fcce3b7b5fbc/omnigenome/src/abc/abstract_model.py#L29)

<a id="model-wrappers"></a>
### Model wrappers
Adapt third-party models (e.g., Hugging Face) into a unified base model API.  
[Code](https://github.com/COLA-Laboratory/OmniGenBench/blob/9ecd8aa0b93d744a6894775046b8fcce3b7b5fbc/omnigenome/src/model/classiifcation/model.py#L16)

<a id="tokenizer-wrappers"></a>
### Tokenizer wrappers
Normalize tokenization methods across different schemes.  
[Code](https://huggingface.co/yangheng/MoEOmniGenomeV2/blob/main/omnigenome_wrapper.py)

<a id="trainer-backends"></a>
### Trainer backends
Provide native, Hugging Face, and Accelerate-compatible training loops.  
[Code](https://github.com/COLA-Laboratory/OmniGenBench/blob/9ecd8aa0b93d744a6894775046b8fcce3b7b5fbc/omnigenome/src/trainer/trainer.py#L77)

<a id="evaluation-inference-api"></a>
### Evaluation & inference API
Standardize model evaluation metrics and batch inference calls.  
[Code](https://github.com/COLA-Laboratory/OmniGenBench/blob/9ecd8aa0b93d744a6894775046b8fcce3b7b5fbc/omnigenome/src/trainer/trainer.py#L286)

<a id="model-hub-integration"></a>
### Model hub integration
Fetch and version pretrained GFMs from Hugging Face.  
[Code](https://github.com/COLA-Laboratory/OmniGenBench/blob/9ecd8aa0b93d744a6894775046b8fcce3b7b5fbc/omnigenome/utility/model_hub/model_hub.py#L20)

<a id="task-compiling"></a>
### Task compiling
Aggregate individual tasks into a coherent benchmark suite.  
[Code](https://github.com/COLA-Laboratory/OmniGenBench/blob/master/examples/RGB/metadata.py#L10)

<a id="automated-benchmarking"></a>
### Automated benchmarking
Orchestrate end-to-end evaluation of all tasks against a model.  
[Code](https://github.com/COLA-Laboratory/OmniGenBench/blob/9ecd8aa0b93d744a6894775046b8fcce3b7b5fbc/omnigenome/auto/auto_bench/auto_bench.py#L34)

<a id="metrics-configuration"></a>
### Metrics configuration
Load and validate metric definitions from JSON/YAML.  
[Code](https://github.com/COLA-Laboratory/OmniGenBench/blob/master/examples/RGB/RNA-mRNA/config.py#L162)

<a id="reporting-leaderboard-submission"></a>
### Reporting & leaderboard submission
Generate summary metrics and submit results to Hugging Face Space.  
[Code](https://huggingface.co/spaces/yangheng/OmniGenomeLeaderboard)

<a id="task-registration"></a>
### Task registration
Dynamically register new tasks into the platform via config files.  
[Code](https://github.com/COLA-Laboratory/OmniGenBench/blob/9ecd8aa0b93d744a6894775046b8fcce3b7b5fbc/examples/RGB/RNA-mRNA/config.py)

<a id="dataset-processing"></a>
### Dataset processing
Link a dataset parser and preprocessing pipeline to the task.  
[Code](https://github.com/COLA-Laboratory/OmniGenBench/blob/9ecd8aa0b93d744a6894775046b8fcce3b7b5fbc/examples/RGB/RNA-mRNA/config.py#L24)

<a id="model-composition"></a>
### Model composition
Attach task-specific heads to a base GFM.  
[Code](https://github.com/COLA-Laboratory/OmniGenBench/blob/9ecd8aa0b93d744a6894775046b8fcce3b7b5fbc/examples/RGB/RNA-mRNA/config.py#L72)

<a id="task-specific-configuration"></a>
### Task-specific configuration
Define loss functions, metrics, and optimizer settings for each task.  
[Code](https://github.com/COLA-Laboratory/OmniGenBench/blob/9ecd8aa0b93d744a6894775046b8fcce3b7b5fbc/examples/RGB/RNA-SNMD/config.py#L77)

<a id="pipeline-definitions"></a>
### Pipeline definitions
Specify batch training and simple deployment entrypoints.  
[Code](https://github.com/COLA-Laboratory/OmniGenBench/blob/9ecd8aa0b93d744a6894775046b8fcce3b7b5fbc/examples/tutorials/Secondary_Structure_Prediction_Tutorial.ipynb#L1008)

<a id="classification-ranking-metrics"></a>
### Classification & ranking metrics
Compute evaluation metrics such as accuracy, precision/recall, F1, ROC-AUC, and NDCG.  
[Code](https://github.com/COLA-Laboratory/OmniGenBench/tree/9ecd8aa0b93d744a6894775046b8fcce3b7b5fbc/examples/RGB/RNA-SSP-bpRNA/config.py#L37)

<a id="regression-metrics"></a>
### Regression metrics
Compute MAE, MSE, RMSE, R², and Spearman’s ρ.  
[Code](https://github.com/COLA-Laboratory/OmniGenBench/tree/9ecd8aa0b93d744a6894775046b8fcce3b7b5fbc/examples/RGB/RNA-SSP-bpRNA/config.py#L37)

<a id="distance-similarity-metrics"></a>
### Distance & similarity metrics
Evaluate vector-based distances like Euclidean, cosine similarity, and Hamming distance.  
[Code](https://github.com/COLA-Laboratory/OmniGenBench/tree/9ecd8aa0b93d744a6894775046b8fcce3b7b5fbc/examples/RGB/RNA-SSP-bpRNA/config.py#L37)

<a id="custom-genomic-metrics"></a>
### Custom genomic metrics
Structural alignment scores and domain-specific measures.  
[Code](#)

<a id="sequence-level-motif-analysis"></a>
### Sequence-level motif analysis
Identify and visualize important motifs via attention or gradients.  
[Code](#)

<a id="embedding-space-analysis"></a>
### Embedding space analysis
Project high-dimensional embeddings into 2D/3D for cluster inspection.  
[Code](#)

<a id="attention-map-visualization"></a>
### Attention map visualization
Render transformer attention weights over sequences.  
[Code](#)
