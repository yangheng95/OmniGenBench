[//]: # (![favicon.png]&#40;asset/favicon.png&#41;)

[//]: # (<h3 align="center">OmniGenBench provides an all-in-one solution for genomic foundation model finetuning, inference, deployment and automated benchmarking, designed for research and applications in genomics.</h3>)

<div align="center">

  <a href="https://omnigenbenchdoc.readthedocs.io/en/latest/">
    <img src="https://img.shields.io/readthedocs/omnigenbench?logo=readthedocs&logoColor=white" alt="Documentation Status" />
  </a>

  <a href="https://pypi.org/project/omnigenome/">
    <img src="https://img.shields.io/pypi/v/omnigenome?color=blue&label=PyPI" alt="PyPI" />
  </a>

  <a href="https://pepy.tech/project/omnigenome">
    <img src="https://static.pepy.tech/badge/omnigenome" alt="PyPI Downloads" />
  </a>

  <a href="https://pypi.org/project/omnigenbench/">
    <img src="https://img.shields.io/pypi/pyversions/omnigenbench" alt="Python Versions (omnigenbench)" />
  </a>

  <a href="https://github.com/yangheng95/omnigenome/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/yangheng95/omnigenome" alt="License" />
  </a>

</div>
<h3 align="center">
  <a href="#installation">📦 Installation</a>
  <span> · </span>
  <a href="#quick-start">🚀 Getting Started</a>
  <span> · </span>
  <a href="#supported-models">🧬 Model Support</a>
  <span> · </span>
  <a href="#benchmarks">📊 Benchmarks </a>
  <span> · </span>
  <a href="#tutorials">🧪 Application Tutorials</a>
  <span> · </span>
  <a href="https://arxiv.org/pdf/2505.14402">📚 Paper</a>
</h3>


## 🔍 What You Can Do with OmniGenBench?

- 🧬 **Benchmark effortlessly** — Run automated and reproducible evaluations for genomic foundation models  
- 🧠 **Understand your models** — Explore interpretability across diverse tasks and species  
- ⚙️ **Run tutorials instantly** — Use click-to-run guides for genomic sequence modeling  
- 🚀 **Fine-tune and infer efficiently** — Accelerated workflows for fine-tuning and inference on GFMs on downstream tasks

## Installation

### Requirements
Before installing OmniGenBench, ensure you have the following:
- **Python**: 3.10 or higher (3.12 recommended for best compatibility)
- **PyTorch**: 2.6.0 or higher (with CUDA support for GPU acceleration)
- **Transformers**: 4.46.0 or higher (HuggingFace library)

### PyPI Installation (Recommended)
Install the latest stable release from PyPI:
```bash
# Create dedicated conda environment (recommended)
conda create -n omnigen_env python=3.12
conda activate omnigen_env

# Install OmniGenBench
pip install omnigenbench -U
```

### Source Installation (For Development)
Clone the repository and install in editable mode for development:
```bash
git clone https://github.com/yangheng95/OmniGenBench.git
cd OmniGenBench
pip install -e .
```

**Note**: For RNA structure prediction and design features, ViennaRNA is required. Install via conda: `conda install -c bioconda viennarna`

## Quick Start
*OmniGenBench provides unified interfaces for model inference, automated benchmarking, and fine-tuning across 30+ genomic foundation models and 80+ standardized tasks.*

### Auto-inference via CLI
Run inference with fine-tuned models on genomic sequences:
```bash
# Single sequence inference (TF binding prediction)
ogb autoinfer \
    --model yangheng/ogb_tfb_finetuned \
    --sequence "ATCGATCGATCGATCG" \
    --output-file predictions.json

# Batch inference from file (translation efficiency prediction)
ogb autoinfer \
    --model yangheng/ogb_te_finetuned \
    --input-file sequences.json \
    --batch-size 64 \
    --output-file results.json
```

### Auto-inference via Python API
Programmatic inference with three-line workflow:
```python
from omnigenbench import ModelHub

# Load fine-tuned model from HuggingFace Hub
model = ModelHub.load("yangheng/ogb_tfb_finetuned")

# Predict transcription factor binding (919 TFs, multi-label classification)
outputs = model.inference("ATCGATCGATCGATCGATCGATCGATCGATCG" * 10)
print(outputs)
# {'predictions': array([1, 0, 1, ...]), 
#  'probabilities': array([0.92, 0.15, 0.87, ...])}

# Interpret results
import numpy as np
binding_sites = np.where(outputs['predictions'] == 1)[0]
print(f"Predicted binding: {len(binding_sites)}/919 transcription factors")
```
**More Examples**: See [Getting Started Guide](docs/GETTING_STARTED.md) and [AutoInfer Examples](examples/autoinfer_examples/) for advanced usage patterns.

### Auto-benchmark via CLI
Automated benchmarking with statistical rigor (multi-seed evaluation):
```bash
# Evaluate model on RGB benchmark (12 RNA tasks) with 3 random seeds
ogb autobench \
    --model yangheng/OmniGenome-186M \
    --benchmark RGB \
    --seeds 0 1 2 \
    --trainer accelerate

# Legacy command (still supported for backward compatibility)
# autobench --model_name_or_path "yangheng/OmniGenome-186M" --benchmark "RGB"
```
**Output**: Results include mean ± standard deviation for each metric (e.g., MCC: 0.742 ± 0.015, F1: 0.863 ± 0.009)

**Visualization**: See [AutoBench GIF](asset/AutoBench.gif) for workflow demonstration.

### Auto-benchmark via Python API
Programmatic benchmarking with flexible configuration:
```python
from omnigenbench import AutoBench

# Initialize benchmark
gfm = 'LongSafari/hyenadna-medium-160k-seqlen-hf'
benchmark = "RGB"  # Options: RGB, BEACON, PGB, GUE, GB
bench_size = 8
seeds = [0, 1, 2, 3, 4]  # Multi-seed for statistical rigor

# Run automated evaluation
bench = AutoBench(
    benchmark=benchmark,
    model_name_or_path=gfm,
    overwrite=False  # Skip completed tasks
)
bench.run(autocast=False, batch_size=bench_size, seeds=seeds)
```
**Advanced Usage**: See [Benchmarking with LoRA](examples/autobench_gfm_evaluation/benchmarking_with_lora.ipynb) for parameter-efficient fine-tuning during evaluation.


## Supported Models

OmniGenBench provides plug-and-play evaluation for **30+ genomic foundation models**, covering both **RNA** and **DNA** modalities across multiple species. All models integrate seamlessly with the framework's automated benchmarking and fine-tuning workflows.

### Representative Models

| Model          | Params | Pre-training Corpus                        | Key Features                                          |
|----------------|--------|--------------------------------------------|-------------------------------------------------------|
| **OmniGenome** | 186M   | 54B plant RNA+DNA tokens                   | Multi-modal encoder, structure-aware, plant-specialized |
| **Agro-NT-1B** | 985M   | 48 edible-plant genomes                    | Billion-scale DNA LM with NT-V2 k-mer vocabulary     |
| **RiNALMo**    | 651M   | 36M ncRNA sequences                        | Largest public RNA LM with FlashAttention-2          |
| **DNABERT-2**  | 117M   | 32B DNA tokens, 136 species (BPE)          | Second-generation DNA BERT with byte-pair encoding   |
| **RNA-FM**     | 96M    | 23M ncRNA sequences                        | High performance on RNA structure prediction tasks   |
| **RNA-MSM**    | 96M    | Multi-sequence alignments                  | MSA-based evolutionary modeling for RNA              |
| **NT-V2**      | 96M    | 300B DNA tokens (850 species)              | Hybrid k-mer vocabulary, cross-species               |
| **HyenaDNA**   | 47M    | Human reference genome                     | Long-context (160k-1M tokens) autoregressive model   |
| **SpliceBERT** | 19M    | 2M pre-mRNA sequences                      | Fine-grained splice-site recognition                 |
| **Caduceus**   | 1.9M   | Human chromosomes                          | Ultra-compact reverse-complement equivariant DNA LM  |
| **RNA-BERT**   | 0.5M   | 4,000+ ncRNA families (Rfam)               | Compact RNA BERT with nucleotide-level masking       |

**Complete Model List**: See Appendix E of the [paper](https://arxiv.org/pdf/2505.14402) for all 30+ supported models, including PlantRNA-FM, UTR-LM, MP-RNA, CALM, and more.

**Model Access**: All models are available on HuggingFace Hub and can be loaded with `ModelHub.load("model-name")`.

## Benchmarks

OmniGenBench supports **five curated benchmark suites** covering both **sequence-level** and **structure-level** genomics tasks across species. All benchmarks are automatically downloaded from HuggingFace Hub on first use.

| Suite        | Focus                       | #Tasks / Datasets        | Representative Tasks                                   |
|--------------|-----------------------------|--------------------------|--------------------------------------------------------|
| **RGB**      | RNA structure + function    | 12 tasks (SN-level)      | Secondary structure, solvent accessibility, degradation |
| **BEACON**   | RNA (multi-domain)          | 13 tasks                 | Base pairing, mRNA design, RNA contact prediction      |
| **PGB**      | Plant long-range DNA        | 7 categories             | PolyA signal, enhancer, chromatin, splice site (up to 50kb context) |
| **GUE**      | DNA general understanding   | 36 datasets (9 tasks)    | TF binding, core promoter, enhancer, epigenetics       |
| **GB**       | Classic DNA classification  | 9 datasets               | Human/mouse enhancers, promoter variant classification |

**Evaluation Protocol**: All benchmarks follow standardized protocols with multi-seed evaluation (typically 3-5 runs) for statistical rigor. Results report mean ± standard deviation for each metric.

**Accessing Benchmarks**: Use `AutoBench(benchmark="RGB")` or `ogb autobench --benchmark RGB` to automatically download and evaluate on any suite.


## Tutorials

### RNA Design

RNA design is the inverse problem of RNA structure prediction: given a target secondary structure (in dot-bracket notation), design RNA sequences that fold into that structure. OmniGenBench provides both CLI and Python API for RNA sequence design using genetic algorithms enhanced with masked language modeling.

#### CLI Usage
```bash
# Basic RNA design for a simple hairpin structure
ogb rna_design --structure "(((...)))"

# Design with custom parameters for better results
ogb rna_design \
    --structure "(((...)))" \
    --model yangheng/OmniGenome-186M \
    --mutation-ratio 0.3 \
    --num-population 200 \
    --num-generation 150 \
    --output-file results.json

# Design complex structure (stem-loop-stem)
ogb rna_design \
    --structure "(((..(((...)))..)))" \
    --num-population 300 \
    --num-generation 200 \
    --output-file complex_design.json
```

**Note**: RNA design is now available through the unified `ogb` command interface.

#### Python API Usage
```python
from omnigenbench import OmniModelForRNADesign

# Initialize model
model = OmniModelForRNADesign(model="yangheng/OmniGenome-186M")

# Design sequences for target structure
sequences = model.design(
    structure="(((...)))",      # Target structure in dot-bracket notation
    mutation_ratio=0.5,          # Mutation rate for genetic algorithm
    num_population=100,          # Population size
    num_generation=100           # Number of generations
)

print(f"Designed {len(sequences)} sequences:")
for seq in sequences[:5]:
    print(f"  {seq}")
```

**Key Features:**
- 🧬 Multi-objective genetic algorithm with MLM-guided mutations
- ⚡ Automatic GPU acceleration for large populations
- 📊 Real-time progress tracking with early termination
- 🎯 Returns multiple optimal solutions (up to 25 sequences)
- 💾 JSON output format for downstream analysis

**Common Structure Patterns:**
- Simple hairpin: `"(((...)))"`
- Stem-loop-stem: `"(((..(((...)))..)))"`
- Multi-loop: `"(((...(((...)))..(((...))).)))"`
- Long stem: `"((((((((....))))))))"`

The comprehensive tutorials of RNA Design can be found in:
- [RNA Design Examples](examples/rna_sequence_design/rna_design_examples.py) - Comprehensive examples
- [RNA Design README](examples/rna_sequence_design/README.md) - Detailed documentation
- [RNA Design Tutorial](examples/rna_sequence_design/RNA_Design_Tutorial.ipynb) - Interactive notebook

You can find a visual demo of RNA Design [here](asset/RNADesign-Demo.gif).

### RNA Secondary Structure Prediction

RNA secondary structure prediction is a fundamental problem in computational biology,
where the goal is to predict the secondary structure of an RNA sequence.
In this demo, we show how to use OmniGenBench to predict the secondary structure of RNA sequences using a pre-trained model.
The tutorials of RNA Secondary Structure Prediction can be found in
[Secondary_Structure_Prediction_Tutorial.ipynb](examples/rna_secondary_structure_prediction/Secondary_Structure_Prediction_Tutorial.ipynb).

You can find a visual example of RNA Secondary Structure Prediction [here](asset/RNASSP-Demo.gif).

### More Tutorials
Please find more usage tutorials in [examples](examples).

## Citation
```bibtex
@article{yang2024omnigenbench,
      title={OmniGenBench: A Modular Platform for Reproducible Genomic Foundation Models Benchmarking}, 
      author={Heng Yang and Jack Cole, Yuan Li, Renzhi Chen, Geyong Min and Ke Li},
      year={2024},
      eprint={https://arxiv.org/abs/2505.14402},
      archivePrefix={arXiv},
      primaryClass={q-bio.GN},
      url={https://arxiv.org/abs/2505.14402}, 
}
```
## License
OmniGenBench is licensed under the Apache License 2.0. See the LICENSE file for more information.


## Contribution
We welcome contributions to OmniGenBench! If you have any ideas, suggestions, or bug reports, please open an issue or submit a pull request on GitHub.
