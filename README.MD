# OmniGenome: A Comprehensive Toolkit of Genomic Modeling and Benchmarking 

# Introduction
OmniGenome is a comprehensive toolkit for genomic modeling and benchmarking.
It provides a unified interface for various genomic modeling tasks, including genome sequence classification, regression and so on.
OmniGenome is designed to be easy to use and flexible, allowing users to customize their workflows and experiment with different algorithms and parameters. 
It also includes a set of benchmarking tools to evaluate the performance of different genomic foundation models and compare their results. 
OmniGenome is written in Python and is available as an open-source project on GitHub.

# What you can do with OmniGenome
- Click-to-run tutorials of Genomic sequence modeling
- Automated benchmarking of genomic foundation models
- Instant inference using pre-trained checkpoints
- Customizable pipeline for genomic modeling tasks

# Installation
before installing OmniGenome, you need to install the following dependencies:
- Python 3.9+
- PyTorch 2.0+
- Transformers 4.37.0+

- To install OmniGenome, you can use pip:
```bash
pip install omnigenome -U
```
or you can clone the repository and install it from source:
```bash
git clone https://github.com/yangheng95/OmniGenome.git
cd OmniGenome
pip install -e .
```

# Quick Start
## Auto-benchmark for genomic foundation models (a.k.a., pretrained models)
```
from omnigenome import AutoBench
gfm = 'LongSafari/hyenadna-medium-160k-seqlen-hf'
# bench_root could be "RGB", "GB", "PGB", "GUE", which will be downloaded from the Hugging Face model hub
bench_root = "RGB"
bench = AutoBench(bench_root=bench_root, model_name_or_path=gfm, overwrite=False)
bench.run(autocast=False, batch_size=bench_size, seeds=seeds)
```

# License
OmniGenome is licensed under the MIT License. See the LICENSE file for more information.

# Citation
TBC
