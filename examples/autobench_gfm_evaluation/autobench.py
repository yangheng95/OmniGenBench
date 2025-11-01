# %% [markdown]
# # 🧬 Automated Genomic Foundation Model Benchmarking with OmniGenBench
# 
# Welcome to this comprehensive tutorial where we'll explore how to **systematically evaluate genomic foundation models** using **OmniGenBench's AutoBench framework**. This guide will walk you through automated benchmarking, parameter-efficient fine-tuning with LoRA, and comprehensive performance analysis across multiple genomic tasks.
# 
# ### 1. The Evaluation Challenge: Why Benchmark Genomic Foundation Models?
# 
# **Genomic Foundation Model (GFM) benchmarking** is crucial for understanding model capabilities and selecting the best models for specific applications. Systematic evaluation addresses several critical needs:
# 
# - **Model Selection**: Comparing different foundation models to choose the optimal one for your task
# - **Performance Validation**: Ensuring models perform reliably across diverse genomic tasks
# - **Efficiency Analysis**: Understanding computational trade-offs between model size and performance
# - **Research Advancement**: Contributing to the scientific understanding of genomic AI capabilities
# 
# The challenge lies in the diversity of genomic tasks and the computational cost of comprehensive evaluation. This is where automated benchmarking becomes essential.
# 
# ### 2. The Solution: AutoBench Framework
# 
# **AutoBench** is OmniGenBench's automated evaluation system that provides:
# 
# - **Standardized Benchmarks**: Curated datasets covering diverse genomic tasks (GUE, RGB suites)
# - **Parameter-Efficient Fine-tuning**: LoRA integration for efficient model adaptation
# - **Automated Pipeline**: End-to-end evaluation with minimal manual intervention
# - **Comprehensive Metrics**: Multiple evaluation metrics and statistical analysis
# - **Scalable Evaluation**: Support for evaluating multiple models across multiple tasks
# 
# **Benchmark Suites:**
# 
# | Benchmark Suite | Tasks | Focus Area |
# |----------------|--------|------------|
# | **GUE** (Genomic Understanding Evaluation) | DNA/Protein tasks | Basic genomic understanding |
# | **RGB** (RNA Genome Benchmark) | RNA-specific tasks | RNA biology and function |
# 
# ### 3. The Tool: LoRA + AutoBench Integration
# 
# #### Parameter-Efficient Fine-Tuning with LoRA
# **Low-Rank Adaptation (LoRA)** enables efficient fine-tuning of large genomic foundation models by:
# 
# 1. **Freezing Base Model**: Original model weights remain unchanged
# 2. **Low-Rank Decomposition**: Adding small trainable matrices to transformer layers
# 3. **Reduced Parameters**: Dramatically fewer trainable parameters (~0.1-1% of original)
# 4. **Task Switching**: Easy switching between different task-specific adapters
# 
# #### AutoBench Integration
# The integration provides:
# - **Automated LoRA Configuration**: Optimal LoRA parameters for genomic tasks
# - **Batch Processing**: Evaluate multiple models and tasks simultaneously
# - **Resource Management**: Efficient GPU memory usage and training optimization
# - **Result Aggregation**: Comprehensive performance reporting and analysis
# 
# ### 4. The Workflow: A 4-Step Guide to Automated Benchmarking
# 
# ```mermaid
# flowchart TD
#     subgraph "4-Step Workflow for Automated GFM Benchmarking"
#         A["📥 Step 1: Setup and Configuration<br/>Configure benchmarks, models, and LoRA parameters"] --> B["🔧 Step 2: Model Loading and Preparation<br/>Load foundation models and initialize AutoBench"]
#         B --> C["🎓 Step 3: Automated Benchmarking<br/>Run systematic evaluation across tasks and models"]
#         C --> D["🔮 Step 4: Analysis and Interpretation<br/>Analyze results and derive insights"]
#     end
# 
#     style A fill:#e1f5fe,stroke:#333,stroke-width:2px
#     style B fill:#f3e5f5,stroke:#333,stroke-width:2px
#     style C fill:#e8f5e8,stroke:#333,stroke-width:2px
#     style D fill:#fff3e0,stroke:#333,stroke-width:2px
# ```
# 
# Let's start systematically evaluating genomic foundation models!

# %% [markdown]
# ## 🚀 Step 1: Setup and Configuration
# 
# This first step focuses on setting up our automated benchmarking environment and configuring the evaluation parameters.
# 
# ### 1.1: Environment Setup
# 
# First, let's install the required packages for automated benchmarking with LoRA fine-tuning.

# %% [markdown]
# ## 1. Setup & Installation
# 
# First, let's ensure all the required packages are installed. If you have already installed them, you can skip this cell.

# %%

# %% [markdown]
# ## 2. Import Libraries
# 
# Import all the necessary libraries for the benchmark.

# %%
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from omnigenbench import AutoBench

print("Libraries imported successfully.")

# %% [markdown]
# ## 3. Configuration
# 
# This section contains all the settings for the LoRA fine-tuning experiment. You can easily modify these parameters to test different models, benchmarks, or LoRA settings.

# %%
# --- General Settings ---
BENCHMARK = "RGB"  # Benchmark suite to use, e.g., "GUE", "RGB"
BATCH_SIZE = 8
PATIENCE = 3
EPOCHS = 1
MAX_EXAMPLES = 1000  # Use a smaller number for quick testing, set to None for all data
SEED = random.randint(0, 1000)

# --- Model Selection ---
# Choose the Genomic Foundation Model (GFM) to fine-tune
GFM_TO_TUNE = 'yangheng/OmniGenome-52M'

# List of available GFMs for testing
AVAILABLE_GFMS = [
    'yangheng/OmniGenome-52M',
    # 'yangheng/OmniGenome-186M',
    # 'yangheng/OmniGenome-v1.5',
    # 'zhihan1996/DNABERT-2-117M',
    # 'LongSafari/hyenadna-large-1m-seqlen-hf',
    # 'InstaDeepAI/nucleotide-transformer-v2-100m-multi-species',
    # 'kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16',
    # 'multimolecule/rnafm',

    # Evo models, you need to install the `evo` package according to the official documentation
    # 'arcinstitute/evo-1-131k-base',
    # 'SpliceBERT-510nt',
]
GFM = GFM_TO_TUNE.split('/')[-1]
# --- LoRA Configuration ---
# This dictionary contains LoRA settings for different models.
# `r`: The rank of the update matrices.
# `lora_alpha`: The scaling factor.
# `lora_dropout`: The dropout probability for LoRA layers.
# `target_modules`: The modules (e.g., attention blocks) to apply LoRA to.
LORA_CONFIGS = {
    "OmniGenome-52M": {
        "r": 8, "lora_alpha": 32, "lora_dropout": 0.1,
        "target_modules": ["key", "value", "dense"], "bias": "none"
    },
    "OmniGenome-186M": {
        "r": 8, "lora_alpha": 32, "lora_dropout": 0.1,
        "target_modules": ["key", "value", "dense"], "bias": "none"
    },
    "caduceus-ph_seqlen-131k_d_model-256_n_layer-16": {
        "r": 8, "lora_alpha": 32, "lora_dropout": 0.1,
        "target_modules": ["in_proj", "x_proj", "out_proj"], "bias": "none"
    },
    "rnamsm": {
        "r": 8, "lora_alpha": 32, "lora_dropout": 0.1,
        "target_modules": ["q_proj", "v_proj", "out_proj"], "bias": "none"
    },
    "rnafm": {
        "r": 8, "lora_alpha": 32, "lora_dropout": 0.1,
        "target_modules": ["key", "value", "dense"], "bias": "none"
    },
    "rnabert": {
        "r": 8, "lora_alpha": 32, "lora_dropout": 0.1,
        "target_modules": ["key", "value", "dense"], "bias": "none"
    },
    "agro-nucleotide-transformer-1b": {
        "r": 8, "lora_alpha": 32, "lora_dropout": 0.1,
        "target_modules": ["key", "value", "dense"], "bias": "none"
    },
    "SpliceBERT-510nt": {
        "r": 8, "lora_alpha": 32, "lora_dropout": 0.1,
        "target_modules": ["key", "value", "dense"], "bias": "none"
    },
    "DNABERT-2-117M": {
        "r": 8, "lora_alpha": 32, "lora_dropout": 0.1,
        "target_modules": ["Wqkv", "dense"], "bias": "none"
    },
    "3utrbert": {
        "r": 8, "lora_alpha": 32, "lora_dropout": 0.1,
        "target_modules": ["key", "value", "dense"], "bias": "none"
    },
    "hyenadna-large-1m-seqlen-hf": {
        "r": 8, "lora_alpha": 32, "lora_dropout": 0.1,
        "target_modules": ["in_proj", "out_proj"], "bias": "none"
    },
    "nucleotide-transformer-v2-100m-multi-species": {
        "r": 8, "lora_alpha": 32, "lora_dropout": 0.1,
        "target_modules": ["key", "value", "dense"], "bias": "none"
    },
    "evo-1-131k-base": {
        "r": 8, "lora_alpha": 32, "lora_dropout": 0.1,
        "target_modules": [
            "Wqkv", "out_proj",
            "mlp",
            "projections",
            "out_filter_dense"
        ],
        "bias": "none"
    },
    "evo-1.5-8k-base": {
        "r": 8, "lora_alpha": 32, "lora_dropout": 0.1,
        "target_modules": [
            "Wqkv", "out_proj",
            "l1", "l2", "l3",
            "projections",
            "out_filter_dense"
        ],
        "bias": "none"
    },
    "evo-1-8k-base": {
        "r": 8, "lora_alpha": 32, "lora_dropout": 0.1,
        "target_modules": [
            "Wqkv", "out_proj",
            "l1", "l2", "l3",
            "projections",
            "out_filter_dense"
        ],
        "bias": "none"
    },
    "evo2_7b": {
        "r": 8, "lora_alpha": 32, "lora_dropout": 0.1,
        "target_modules": [
            "Wqkv", "out_proj",
            "l1", "l2", "l3",
            # "projections",
            "out_filter_dense"
        ],
        "bias": "none"
    },
}

print(f"Configuration loaded:")
print(f"  - GFM to Tune: {GFM_TO_TUNE}")
print(f"  - Benchmark: {BENCHMARK}")
print(f"  - Epochs: {EPOCHS}")
print(f"  - LoRA Config: {LORA_CONFIGS.get(GFM, LORA_CONFIGS[GFM])}")

# %% [markdown]
# ## 4. Model-Specific Loading
# 
# Different GFMs may require specific loading procedures. This function handles these special cases, particularly for models like `multimolecule` or `evo` which might have custom tokenizers or model classes.

# %%
def load_gfm_and_tokenizer(gfm_name):
    """Loads a GFM and its tokenizer, handling special cases."""
    print(f"\nLoading model and tokenizer for: {gfm_name}")
    
    if 'multimolecule' in gfm_name:
        from multimolecule import RnaTokenizer, AutoModelForTokenPrediction
        tokenizer = RnaTokenizer.from_pretrained(gfm_name)
        model = AutoModelForTokenPrediction.from_pretrained(gfm_name, trust_remote_code=True).base_model
        print("Loaded multimolecule model with custom classes.")
        
    elif 'evo-1' in gfm_name:
        # Using transformers to load Evo models
        config = AutoConfig.from_pretrained(gfm_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(gfm_name, config=config, trust_remote_code=True).backbone
        tokenizer = AutoTokenizer.from_pretrained(gfm_name, trust_remote_code=True)
        tokenizer.pad_token_id = tokenizer.pad_token_type_id
        
        # Patch for the unembedding layer
        model.config = config
        model.config.pad_token_id = tokenizer.pad_token_id
        model.unembed.unembed = lambda x: x
        print("Loaded Evo model with custom patching.")
        
    else:
        # Default loading for most Hugging Face models
        tokenizer = None  # Let AutoBench handle it
        model = gfm_name
        print("Using standard model name for AutoBench.")
        
    return model, tokenizer

print("Model loading function defined.")

# %% [markdown]
# ## 5. Running LoRA Fine-tuning
# 
# Now, we'll execute the LoRA fine-tuning process for the selected model. `AutoBench` handles the entire workflow, from data loading and preprocessing to training and evaluation.

# %%
# Load the selected model and tokenizer
model, tokenizer = load_gfm_and_tokenizer(GFM_TO_TUNE)

# Initialize AutoBench
print(f"\nInitializing AutoBench for benchmark: {BENCHMARK}")
bench = AutoBench(
    benchmark=BENCHMARK,
    config_or_model=GFM_TO_TUNE,
    tokenizer=tokenizer,
    overwrite=True,
    trainer='native',  # 'native' or 'accelerate'
    autocast='fp16',  # 'fp16', 'bf16', or 'fp32'
    device='cuda',
)

# Get the appropriate LoRA config
lora_config = LORA_CONFIGS.get(GFM, LORA_CONFIGS[GFM])

# Run the benchmark with LoRA fine-tuning
print(f"\nStarting LoRA fine-tuning for {GFM_TO_TUNE}...")
bench.run(
    batch_size=BATCH_SIZE,
    gradient_accumulation_steps=1,
    patience=PATIENCE,
    max_examples=MAX_EXAMPLES,
    seeds=SEED,
    epochs=EPOCHS,
    lora_config=lora_config, # Pass the LoRA config here
)

print("\n🎉 LoRA fine-tuning complete!")
print("Check the 'autobench_logs' and 'autobench_evaluations' directories for results.")

# %% [markdown]
# ## 6. Multi-Model LoRA Fine-tuning (Optional)
# 
# The following section demonstrates how to automate the LoRA fine-tuning process for a list of GFMs. Uncomment and run this cell to compare the performance of multiple models with LoRA.

# %%
# # Uncomment this cell to run LoRA fine-tuning for multiple models

# print("Starting multi-model LoRA fine-tuning...")
# print("="*50)

# for gfm in AVAILABLE_GFMS:
#     try:
#         # Load model and tokenizer
#         model, tokenizer = load_gfm_and_tokenizer(gfm)

#         # Initialize AutoBench
#         print(f"\nInitializing AutoBench for {gfm} on {BENCHMARK}")
#         bench = AutoBench(
#             benchmark=BENCHMARK,
#             config_or_model=model,
#             tokenizer=tokenizer,
#             overwrite=True,
#             trainer='native',
#             autocast='fp16',
#             device='cuda',
#         )

#         # Get the appropriate LoRA config
#         lora_config = LORA_CONFIGS.get(gfm.split('/')[-1], LORA_CONFIGS['default'])

#         # Run the benchmark
#         print(f"\nStarting LoRA fine-tuning for {gfm}...")
#         bench.run(
#             batch_size=BATCH_SIZE,
#             patience=PATIENCE,
#             max_examples=MAX_EXAMPLES,
#             seeds=SEED,
#             epochs=EPOCHS,
#             lora_config=lora_config,
#         )
#         print(f"\n✅ Finished fine-tuning for {gfm}.")
#         print("="*50)

#     except Exception as e:
#         print(f"\n❌ An error occurred while processing {gfm}: {e}")
#         print("="*50)
#         continue

# print("\n🎉 All models have been processed!")


