#%% raw
# # 🧬 Variant Effect Prediction with OmniGenBench
# 
# Welcome to this comprehensive tutorial on predicting the functional effects of genetic variants using **OmniGenBench** and **PlantRNA-FM** (Plant RNA Foundation Model, 35M parameters, published in *Nature Machine Intelligence*). This guide demonstrates a modern approach to variant effect prediction using specialized plant RNA foundation models.
# 
# > 📚 **Prerequisite**: It is highly recommended to first study the **[Fundamental Concepts Tutorial](https://github.com/yangheng95/OmniGenBench/blob/master/examples/00_fundamental_concepts.ipynb)**, which covers language model concepts, machine learning task classification, and foundation model principles—particularly how PlantRNA-FM works.
# 
# ### 1. The Biological Challenge: Understanding Genetic Variants
# 
# **Genetic variants** are differences in DNA sequences between individuals, ranging from single nucleotide polymorphisms (SNPs) to larger structural variations. While most variants are benign, some can significantly impact:
# 
# - **Gene expression** - affecting protein production levels
# - **Protein function** - altering structure or activity
# - **Regulatory networks** - disrupting transcription factor binding
# - **Disease susceptibility** - increasing risk for genetic disorders
# 
# Experimentally validating millions of variants is impractical, making computational prediction essential for genomic medicine.
# 
# ### 2. The Data: Variant Effect Prediction Dataset
# 
# Our dataset contains:
# - **Genomic coordinates** - Chromosomal positions of variants
# - **Reference and alternative alleles** - DNA sequence changes
# - **Functional annotations** - Known biological impacts
# 
# **Dataset Structure:**
# 
# | chr | pos | ref | alt | functional_impact |
# |-----|-----|-----|-----|------------------|
# | chr1 | 12345 | A | G | Benign |
# | chr2 | 67890 | C | T | Pathogenic |
# 
# ### 3. Quick Start: Variant Effect Prediction Workflow
# 
# This tutorial demonstrates the practical application of **[Fundamental Concepts](https://github.com/yangheng95/OmniGenBench/blob/master/examples/00_fundamental_concepts.ipynb)** using a streamlined 4-step workflow:
# 
# 
#%%
# from IPython.display import Image, display
# display(Image(filename="4-step workflow.png"))

#%% md
# **Variant Effect Prediction** uses embedding comparison to assess functional impacts without training. We leverage `OmniModelForSequenceClassification` with **PlantRNA-FM** to extract plant-optimized embeddings and compare reference vs. alternative sequences.
# 
# ### 4. Tutorial Structure
# 
# 1. **[Data Preparation](01_vep_data_preparation.ipynb)**: Load variant datasets and reference genome
# 2. **[Model Setup](02_vep_model_setup.ipynb)**: Initialize PlantRNA-FM for plant genomic analysis
# 3. **[Embedding Extraction](03_embedding_and_scoring.ipynb)**: Compare reference and alternative sequences using PlantRNA-FM
# 4. **[Visualization](04_visualization_and_export.ipynb)**: Analyze and export results
# 
# Let's begin!
# 
#%% md
# ## 🚀 Step 1: Environment Setup and Configuration
# 
#%%
# %pip install omnigenbench -U

#%%
from omnigenbench import (
    OmniTokenizer,
    OmniModelForSequenceClassification,
    OmniDatasetForSequenceClassification
)

#%% md
# ### Configuration
# 
# Define analysis parameters with sensible defaults:
# 
#%%
# Configuration parameters
dataset_name = "yangheng/variant_effect_prediction"
model_name = "yangheng/PlantRNA-FM"  # Using PlantRNA-FM for plant variant analysis
max_length = 512
batch_size = 16
context_length = 200
max_variants = 100  # Use subset for quick testing
cache_dir = "vep_data"
output_dir = "vep_results"

#%% md
# ## 📊 Step 2: Data Loading
# 
# Load the variant dataset using OmniGenBench's enhanced data loading:
# 
#%%
# Load tokenizer
tokenizer = OmniTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Load dataset with automatic caching
datasets = OmniDatasetForSequenceClassification.from_hub(
    dataset_name_or_path=dataset_name,
    tokenizer=tokenizer,
    max_length=max_length,
    cache_dir=cache_dir
)

#%% md
# ## 🔧 Step 3: Model Initialization
# 
#%%
model = OmniModelForSequenceClassification(
    model_name,
    tokenizer=tokenizer,
    num_labels=2,
    trust_remote_code=True
)
model.eval()

#%% md
# ## 🧬 Step 4: Variant Effect Scoring
# 
# Extract embeddings and calculate effect scores:
# 
#%%
import torch
from tqdm import tqdm

test_dataset = datasets['test']
dataloader = test_dataset.get_dataloader(batch_size=batch_size, shuffle=False)

results = []
with torch.no_grad():
    for batch in tqdm(dataloader, desc="Processing variants"):
        outputs = model(**batch)
        embeddings = outputs.hidden_states[-1][:, 0, :]  # CLS token
        results.append(embeddings)

all_embeddings = torch.cat(results, dim=0)

#%% md
# ## 📈 Results and Visualization
# 
# For detailed analysis and visualization, see the complete tutorial notebooks:
# - **[01_vep_data_preparation.ipynb](01_vep_data_preparation.ipynb)**
# - **[02_vep_model_setup.ipynb](02_vep_model_setup.ipynb)**
# - **[03_embedding_and_scoring.ipynb](03_embedding_and_scoring.ipynb)**
# - **[04_visualization_and_export.ipynb](04_visualization_and_export.ipynb)**
# 