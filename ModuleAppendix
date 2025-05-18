
#  OmniGenBench Framework Module Appendix 

This appendix provides a detailed, markdown-formatted overview of each core and auxiliary module in **OmniGenBench**. For each module you will find:

- **Functionality Description**  
- **Inputs / Outputs Definition**  
- **Interface Introduction**  
- **Source Code Link**

Each module title links directly to a representative code example or resource.

---

## [1. Data Module](https://github.com/COLA-Laboratory/OmniGenBench/blob/9ecd8aa0b93d744a6894775046b8fcce3b7b5fbc/omnigenome/src/abc/abstract_dataset.py#L55)

### 1.1 Data Parsing  
- **Functionality Description**  
  Read multi-format genomic data (FASTA, JSON, CSV, Pandas, etc.), validate metadata tags, and convert to in-memory records ([dictionary-like data](https://github.com/COLA-Laboratory/OmniGenBench/blob/9ecd8aa0b93d744a6894775046b8fcce3b7b5fbc/omnigenome/src/abc/abstract_dataset.py#L44)).  
- **Inputs**  
  Raw files `.fasta` / `.fa` / `.json` / `.csv`; metadata (species, tags, splits)  
- **Outputs**  
  Python dictionary of sequences + metadata  
- **Interface**  
  `AbstractDataset.load_from_file(data_file)`  
- **Source Code Review**  
  [abstract_dataset.py#L281](https://github.com/COLA-Laboratory/OmniGenBench/blob/9ecd8aa0b93d744a6894775046b8fcce3b7b5fbc/omnigenome/src/abc/abstract_dataset.py#L281)

### 1.2 Adaptive Truncation & Padding  
- **Functionality Description**  
  Dynamically shorten or pad sequences to model-compatible lengths while preserving key regions.  
- **Inputs**  
  Sequence strings; target length  
- **Outputs**  
  Truncated or padded sequence strings  
- **Interface**  
  `AbstractDataset._pad_and_truncate(tensor_data)`  
- **Source Code**  
  [abstract_dataset.py#L182](https://github.com/COLA-Laboratory/OmniGenBench/blob/9ecd8aa0b93d744a6894775046b8fcce3b7b5fbc/omnigenome/src/abc/abstract_dataset.py#L182)

### 1.3 Instance Filtering (CD-HIT-EST)  
- **Functionality Description**  
  Remove near-duplicate sequences to prevent label leakage in structure tasks.  
- **Inputs**  
  Parsed sequence list; similarity threshold  
- **Outputs**  
  Filtered sequence list  
- **Interface**  
  `cd-hit-est CI`  
- **Source Code**  
  [CD-HIT-EST User Guide](https://bioinformatics.org/cd-hit/)

### 1.4 MLM Augmentation  
- **Functionality Description**  
  Mask and predict random nucleotides to generate synthetic variants for robustness.  
- **Inputs**  
  Sequence tensors; masking ratio  
- **Outputs**  
  Augmented sequence tensors  
- **Interface**  
  `OmniGenomeModelForAugmentation.augment_sequence(sequence)`  
- **Source Code**  
  [RNA_Augmentation_Tutorial.ipynb](https://github.com/COLA-Laboratory/OmniGenBench/blob/9ecd8aa0b93d744a6894775046b8fcce3b7b5fbc/examples/tutorials/RNA_Augmentation_Tutorial.ipynb)

### 1.5 Data Hub  
- **Functionality Description**  
  Index and serve benchmark datasets via HF Spaces.  
- **Inputs**  
  Dataset cards, upload packages  
- **Outputs**  
  Public dataset URLs; metadata pages  
- **Interface**  
  HF Space API  
- **Source Code**  
  [OmniGenomeLeaderboard Space](https://huggingface.co/spaces/yangheng/OmniGenomeLeaderboard)


---

## [2. Model Module](https://github.com/COLA-Laboratory/OmniGenBench/tree/9ecd8aa0b93d744a6894775046b8fcce3b7b5fbc/omnigenome/src/abc/abstract_model.py#L29)

### 2.1 Base Model Template  
- **Functionality Description**  
  Define a minimal GFM interface (`forward`, `predict`, `inference`, `load`, `save`) for heterogeneous architectures.  
- **Inputs**  
  Token IDs; attention masks  
- **Outputs**  
  Hidden states; logits  
- **Interface**  
  `BaseModel.forward(tensor_data)`  
  `BaseModel.predict(tensor_data)`
  `BaseModel.inference(sequence)`
  `BaseModel.load(load_path)`
  `BaseModel.save(save_path)`
- **Source Code**  
  [abstract_model.py#L29](https://github.com/COLA-Laboratory/OmniGenBench/blob/9ecd8aa0b93d744a6894775046b8fcce3b7b5fbc/omnigenome/src/abc/abstract_model.py#L29)

### 2.2 Model Wrappers  
- **Functionality Description**  
  Adapt third-party models (Hugging Face, custom) into the BaseModel API.  
- **Inputs**  
  HF model instance  
- **Outputs**  
  Wrapped `BaseModel` subclass  
- **Interface**  
  `ClassificationModelWrapper`  
- **Source Code**  
  [model/classiifcation/model.py#L16](https://github.com/COLA-Laboratory/OmniGenBench/blob/9ecd8aa0b93d744a6894775046b8fcce3b7b5fbc/omnigenome/src/model/classiifcation/model.py#L16)

### 2.3 Tokenizer Wrappers  
- **Functionality Description**  
  Normalize tokenization calls across k-mer, byte-pair, and custom schemes.  
- **Inputs**  
  Raw sequence strings  
- **Outputs**  
  Token ID tensors  
- **Interface**  
  `TokenizerWrapper.encode()` / `decode()`  
- **Source Code**  
  [omnigenome_wrapper.py](https://huggingface.co/yangheng/MoEOmniGenomeV2/blob/main/omnigenome_wrapper.py)

### 2.4 Trainer Backends  
- **Functionality Description**  
  Provide three interchangeable training engines: native loop, HF Trainer, Accelerate.  
- **Inputs**  
  Model, dataset, hyperparameters  
- **Outputs**  
  Checkpoints, logs  
- **Interface**  
  `Trainer.train()`
  `AccelerateTrainer.train()`
  `HFTraner.train()`
- **Source Code**  
  - Simple Native: [trainer.py#L77](https://github.com/COLA-Laboratory/OmniGenBench/blob/9ecd8aa0b93d744a6894775046b8fcce3b7b5fbc/omnigenome/src/trainer/trainer.py#L77)
  - HuggingFace: [hf_trainer.py#L17](https://github.com/COLA-Laboratory/OmniGenBench/blob/9ecd8aa0b93d744a6894775046b8fcce3b7b5fbc/omnigenome/src/trainer/hf_trainer.py#L17)  
  - Accelerate: [accelerate_trainer.py#L64](https://github.com/COLA-Laboratory/OmniGenBench/blob/9ecd8aa0b93d744a6894775046b8fcce3b7b5fbc/omnigenome/src/trainer/accelerate_trainer.py#L64)

### 2.5 Evaluation & Inference API  
- **Functionality Description**  
  Standardize model evaluation metrics and batch inference calls.  
- **Inputs**  
  Validation dataset, batch size  
- **Outputs**  
  Metric dict; prediction arrays  
- **Interfaces & Sources**  
  - `Trainer.evaluate(test_set)` ([trainer.py#L281](https://github.com/COLA-Laboratory/OmniGenBench/blob/9ecd8aa0b93d744a6894775046b8fcce3b7b5fbc/omnigenome/src/trainer/trainer.py#L286))  
  - `model.inference(inference_set)` ([model.py#L62](https://github.com/COLA-Laboratory/OmniGenBench/blob/9ecd8aa0b93d744a6894775046b8fcce3b7b5fbc/omnigenome/src/model/classiifcation/model.py#L62))

### 2.6 Model Hub Integration  
- **Functionality Description**  
  Fetch and version pretrained GFMs from Hugging Face.  
- **Inputs**  
  Model ID string  
- **Outputs**  
  Downloaded weights; config  
- **Interface**  
  `ModelHub.load(model_name_or_path)`  
- **Source Code**  
  [model_hub.py#L20](https://github.com/COLA-Laboratory/OmniGenBench/blob/9ecd8aa0b93d744a6894775046b8fcce3b7b5fbc/omnigenome/utility/model_hub/model_hub.py#L20)

---



## [3. Benchmark Module](https://github.com/COLA-Laboratory/OmniGenBench/tree/master/examples/RGB/metadata.py#L10)

### 3.1 Task Compiling  
- **Functionality Description**  
  Aggregate individual tasks into a coherent benchmark suite.  
- **Inputs**  
  Task list; metadata file  
- **Outputs**  
  Suite object with tasks  
- **Interface**  
  `Benchmark.compile()`  
- **Source Code**  
  [metadata.py#L10](https://github.com/COLA-Laboratory/OmniGenBench/blob/master/examples/RGB/metadata.py#L10)

### 3.2 Automated Benchmarking  
- **Functionality Description**  
  Orchestrate end-to-end evaluation of all tasks against a model.  
- **Inputs**  
  Model ID; suite object  
- **Outputs**  
  Raw metrics per task  
- **Interface**  
  `autobench --benchmark --model`  
- **Source Code**  
  [auto_bench.py#L34](https://github.com/COLA-Laboratory/OmniGenBench/blob/9ecd8aa0b93d744a6894775046b8fcce3b7b5fbc/omnigenome/auto/auto_bench/auto_bench.py#L34)

### 3.3 Metrics Configuration  
- **Functionality Description**  
  Load and validate metric definitions from JSON/YAML.  
- **Inputs**  
  Config file path  
- **Outputs**  
  Metric registry entries  
- **Interface**  
  Internal JSON loader  
- **Source Code**  
  [config.py#L162](https://github.com/COLA-Laboratory/OmniGenBench/blob/master/examples/RGB/RNA-mRNA/config.py#L162)

### 3.4 Reporting & Leaderboard Submission  
- **Functionality Description**  
  Generate summary tables and push to Hugging Face Space.  
- **Inputs**  
  Metrics CSV; HF Space credentials  
- **Outputs**  
  Leaderboard entries  
- **Interface**  
  `autobench --report`  
- **Source Code**  
  [OmniGenomeLeaderboard Space](https://huggingface.co/spaces/yangheng/OmniGenomeLeaderboard)

---


The following two layers of module are logically the sub module of the benchmark module, i.e., Task Module and Metric Module 


## [S1. Task Module](https://github.com/COLA-Laboratory/OmniGenBench/tree/9ecd8aa0b93d744a6894775046b8fcce3b7b5fbc/examples/RGB/RNA-mRNA/config.py)

### S1.1 Task Registration  
- **Functionality Description**  
  Dynamically register new tasks into the platform via config files.  
- **Inputs**  
  `config.py` 
- **Outputs**  
  `Task` object in registry  
- **Interface**  
  `AutoBenchConfig(config_dict)`  
- **Source Code**  
  [config.py#L24](https://github.com/COLA-Laboratory/OmniGenBench/blob/9ecd8aa0b93d744a6894775046b8fcce3b7b5fbc/examples/RGB/RNA-mRNA/config.py)

### S1.2 Dataset Processing  
- **Functionality Description**  
  Link a dataset parser and preprocessing pipeline to the task.  
- **Inputs**  
  Selected Data Module pipeline  
- **Outputs**  
  Ready-to-train DataLoader  
- **Interface**  
  `dataset_fn`  
- **Source Code**  
  [config.py#L24](https://github.com/COLA-Laboratory/OmniGenBench/blob/9ecd8aa0b93d744a6894775046b8fcce3b7b5fbc/examples/RGB/RNA-mRNA/config.py#L24)

### S1.3 Model Composition  
- **Functionality Description**  
  Attach task-specific heads to a base GFM.  
- **Inputs**  
  BaseModel class, head specs  
- **Outputs**  
  Composite model instance  
- **Interface**  
  e.g., `model.set_loss_fn(torch.CrossEntropy())`  
- **Source Code**  
  [config.py#L72](https://github.com/COLA-Laboratory/OmniGenBench/blob/9ecd8aa0b93d744a6894775046b8fcce3b7b5fbc/examples/RGB/RNA-mRNA/config.py#L72)

### S1.4 Task-Specific Configuration  
- **Functionality Description**  
  Define losses, metrics, and other hyperparameters per task.  
- **Inputs**  
  Loss name, metric list, optimizer settings  
- **Outputs**  
  Initialized training components  
- **Interface**  
  `loss_fn`, `metric_fns`  
- **Source Code**  
  - Loss: [config.py#L77](https://github.com/COLA-Laboratory/OmniGenBench/blob/9ecd8aa0b93d744a6894775046b8fcce3b7b5fbc/examples/RGB/RNA-SNMD/config.py#L77)  
  - Metrics: [config.py#L37](https://github.com/COLA-Laboratory/OmniGenBench/blob/9ecd8aa0b93d744a6894775046b8fcce3b7b5fbc/examples/RGB/RNA-SSP-bpRNA/config.py#L37)

### S1.5 Pipeline Definitions  
- **Functionality Description**  
  Specify batch training and simple deployment entrypoints.  
- **Inputs**  
  Training scripts or notebook paths  
- **Outputs**  
  Executable pipelines  
- **Interface**  
  `Trainer.train().save_model()` / `ModelHub.load(model_path).inference()`  
- **Source Code**  
  - Train: [batch_beacon_benchmark.py#L47](https://github.com/COLA-Laboratory/OmniGenBench/blob/9ecd8aa0b93d744a6894775046b8fcce3b7b5fbc/examples/autobench/batch_beacon_benchmark.py#L47)  
  - Deploy: [Secondary_Structure_Prediction_Tutorial.ipynb#L1008](https://github.com/COLA-Laboratory/OmniGenBench/blob/9ecd8aa0b93d744a6894775046b8fcce3b7b5fbc/examples/tutorials/Secondary_Structure_Prediction_Tutorial.ipynb#L1008)

---



## [S2. Metric Module](https://github.com/COLA-Laboratory/OmniGenBench/tree/9ecd8aa0b93d744a6894775046b8fcce3b7b5fbc/examples/RGB/RNA-SSP-bpRNA/config.py#L37)

### S2.1 Classification & Ranking Metrics  
- **Functionality Description**  
  Accuracy, Precision/Recall, F1, ROC-AUC, NDCG  
- **Interface**  
  `metric_registry["accuracy"]()`  

### S2.2 Regression Metrics  
- **Functionality Description**  
  MAE, MSE, RMSE, $R^2$, Spearman’s $\rho$  
- **Interface**  
  `metric_registry["mse"]()`  

### S2.3 Distance / Similarity Metrics  
- **Functionality Description**  
  Euclidean, Cosine, sequence Hamming distance  
- **Interface**  
  `metric_registry["cosine"]()`  

### S2.4 Custom Genomic Metrics  
- **Functionality Description**  
  Structural alignment scores, domain-specific measures  
- **Interface**  
  Defined in task config  

---


## [4. Interpretability Module](https://github.com/COLA-Laboratory/OmniGenBench/tree/master/examples/tutorials)

### 4.1 Sequence-Level Motif Analysis  
- **Functionality Description**  
  Identify and visualize high-importance motifs via gradient or attention attribution.  
- **Inputs**  
  Attention maps; sequence labels  
- **Outputs**  
  Motif logos; heatmaps  
- **Interface**  
  `MotifScanner.scan()`  
- **Source Code**  
  [seq_motif_preservation Section](#)

### 4.2 Embedding Space Analysis  
- **Functionality Description**  
  Project high-dim embeddings into 2D/3D for cluster inspection.  
- **Inputs**  
  Embedding arrays; labels  
- **Outputs**  
  Scatter plots; interactive HTML  
- **Interface**  
  `EmbeddingProjector.project()`  
- **Source Code**  
  [feature_embedding_analysis Section](#)

### 4.3 Attention Map Visualization  
- **Functionality Description**  
  Render transformer attention weights over sequences.  
- **Inputs**  
  Attention tensors; sequence tokens  
- **Outputs**  
  Attention heatmaps  
- **Interface**  
  `AttentionPlotter.plot()`  
- **Source Code**  
  [attention_inspection Section](#)

---
