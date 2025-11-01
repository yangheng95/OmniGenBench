# %% [markdown]
# # 🧬 RNA Secondary Structure Prediction with OmniGenBench
# 
# Welcome to this comprehensive tutorial where we'll explore how to predict **RNA secondary structures** from primary sequences using **OmniGenBench**. This guide will walk you through a complete genomic deep learning project, from understanding the fundamental biological concepts to deploying a trained model for real-world applications.
# 
# ### 1. The Biological Challenge: What is RNA Secondary Structure?
# 
# **RNA secondary structure** refers to the pattern of base pairing within a single RNA molecule, forming loops, stems, and other structural elements that are crucial for RNA function. Understanding these structures is fundamental because:
# 
# - **Functional Importance**: RNA structure directly determines function in processes like catalysis (ribozymes), regulation (miRNAs), and protein synthesis (rRNA, tRNA)
# - **Disease Relevance**: Structural mutations can disrupt RNA function, leading to genetic disorders
# - **Drug Design**: RNA structures serve as targets for therapeutic interventions
# - **Synthetic Biology**: Designing functional RNA molecules requires precise structural control
# 
# The challenge lies in predicting these complex three-dimensional folding patterns from linear sequence information - a problem that has puzzled scientists for decades.
# 
# ### 2. The Data: RNA Secondary Structure Dataset
# 
# To train our predictive model, we utilize the **bpRNA dataset**, a comprehensive collection of RNA sequences with experimentally determined secondary structures.
# 
# - **What it contains**: RNA sequences with annotated base-pairing patterns
# - **What it labels**: Each nucleotide position is classified based on its structural role:
#   - `.` (unpaired/loop regions)
#   - `(` and `)` (base-paired positions forming stems)
# - **Our Goal**: Train a model that can accurately predict the structural state of each nucleotide position
# 
# **Dataset Structure:**
# 
# | sequence | structure |
# |---------|-----------|
# | AUGCCGUGC... | .(((...))).|
# | GCCAUGCUA... | (((....))).| 
# | ... | ... |
# 
# ### 3. The Tool: From Language Models to Genomic Foundation Models
# 
# #### The Rise of Language Models
# **Language Models (LMs)** like BERT have revolutionized Natural Language Processing by learning deep patterns in text. Similarly, **Genomic Foundation Models (GFMs)** like **OmniGenome** learn the "grammar" of genetic sequences.
# 
# #### RNA Structure Prediction as Token Classification
# In this tutorial, we treat RNA secondary structure prediction as a **token classification** problem, where each nucleotide (token) in the sequence is assigned a structural label (unpaired, left bracket, or right bracket).
# 
# ### 4. The Workflow: A 4-Step Guide to Fine-Tuning
# 
# ```mermaid
# flowchart TD
#     subgraph "4-Step Workflow for RNA Structure Prediction"
#         A["📥 Step 1: Data Preparation<br/>Download and process the bpRNA dataset"] --> B["🔧 Step 2: Model Initialization<br/>Load the pre-trained OmniGenome model"]
#         B --> C["🎓 Step 3: Model Training<br/>Fine-tune the model on structure data"]
#         C --> D["🔮 Step 4: Model Inference<br/>Predict structures on new sequences"]
#     end
# 
#     style A fill:#e1f5fe,stroke:#333,stroke-width:2px
#     style B fill:#f3e5f5,stroke:#333,stroke-width:2px
#     style C fill:#e8f5e8,stroke:#333,stroke-width:2px
#     style D fill:#fff3e0,stroke:#333,stroke-width:2px
# ```
# 
# Let's get started!

# %% [markdown]
# ## 🚀 Step 1: Data Preparation
# 
# This first step focuses on preparing our data for RNA secondary structure prediction. It involves:
# 1. **Environment Setup**: Installing and importing necessary libraries
# 2. **Configuration**: Centralizing important parameters  
# 3. **Data Acquisition**: Loading the bpRNA dataset
# 4. **Data Pipeline**: Creating efficient data loading for token classification
# 
# ### 1.1: Environment Setup
# 
# First, let's install the required packages. `omnigenbench` provides state-of-the-art genomic foundation models optimized for RNA structure prediction.

# %% [markdown]
# ### 1.2: Import Required Libraries
# 
# Next, we import the essential libraries for RNA secondary structure prediction. This includes tools for data processing, deep learning, and the specialized OmniGenBench components for token classification tasks.

# %% [markdown]
# !pip install omnigenbench -U  # Install the latest version of omnigenbench

# %% [markdown]
# ### 1.3: Global Configuration
# 
# Let's centralize all important parameters for easy experimentation and reproducibility.
# 
# #### Key Parameters
# - **Dataset**: We'll use the bpRNA dataset for RNA secondary structure prediction
# - **Model**: We select the `OmniGenome-52M` model for efficient learning and prototyping
# - **Labels**: Structure prediction uses 3 classes: unpaired (.), left bracket ((), and right bracket ())

# %% [markdown]
# ### 1.4: Data Acquisition and Loading

# %%
import os
import torch
import numpy as np

from omnigenbench import (
    ClassificationMetric,
    AccelerateTrainer,
    ModelHub,
    OmniTokenizer,
    OmniDatasetForTokenClassification,
    OmniModelForTokenClassification,
)

# %%
model_name_or_path = "yangheng/OmniGenome-52M"
dataset_name = "rna_secondary_structure"

# Define label mapping for RNA secondary structure
label2id = {"(": 0, ")": 1, ".": 2}  # Left bracket, Right bracket, Unpaired

# %%
# Load tokenizer and datasets using enhanced OmniDataset framework
tokenizer = OmniTokenizer.from_pretrained(model_name_or_path)
print(f"✅ Tokenizer loaded: {model_name_or_path}")

# Load datasets for token classification
print("🏗️ Loading datasets with automatic download...")
datasets = OmniDatasetForTokenClassification.from_hub(
    dataset_name_or_path="RNA-SSP-Archive2",
    tokenizer=tokenizer,
    max_length=512,
    label2id=label2id,
    max_examples=1000,  # For quick testing; set to None for full dataset
)
print(f"📊 Datasets loaded: {list(datasets.keys())}")

# %% [markdown]
# ### 1.5: Dataset Loading with OmniGenBench
# 
# With OmniGenBench, data loading for token classification is significantly simplified! The framework automatically handles:
# 
# #### A. Automatic Data Processing
# The `OmniDatasetForTokenClassification` class automatically:
# 1. **Downloads and processes** the bpRNA dataset from our curated collection
# 2. **Handles sequence preprocessing** including tokenization and label alignment
# 3. **Manages token-level classification formatting** for structure prediction
# 4. **Creates train/validation/test splits** ready for training
# 
# #### B. Built-in Optimizations
# The framework includes several optimizations:
# 1. **Efficient batching** for variable-length sequences
# 2. **Memory management** for large RNA datasets
# 3. **Automatic label alignment** with tokenized sequences
# 4. **Proper masking** for padded positions
# 
# This streamlined approach eliminates the need for complex custom dataset classes.

# %%
print("📝 Data loading completed! Using modern OmniDataset framework.")
print(f"📊 Loaded datasets: {list(datasets.keys())}")
for split, dataset in datasets.items():
    print(f"  - {split}: {len(dataset)} samples")

# Inspect a sample to understand the data structure
if len(datasets["train"]) > 0:
    sample = datasets["train"][0]
    print(f"\n🔍 Sample data structure:")
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape {value.shape}, dtype {value.dtype}")
        else:
            print(f"  {key}: {type(value)}")

# %% [markdown]
# ## 🚀 Step 2: Model Initialization
# 
# With our data pipeline in place, it's time to set up the model for RNA secondary structure prediction. We'll leverage the power of Genomic Foundation Models by adapting a pre-trained **OmniGenome** model for token classification.
# 
# This involves:
# 1. **The Tokenizer**: Converting RNA sequences into numerical format
# 2. **The Base Model**: The pre-trained OmniGenome model with genomic understanding
# 3. **The Classification Head**: A layer that maps sequence representations to structure labels
# 
# The `OmniModelForTokenClassification` class handles this seamlessly for our 3-class structure prediction task.

# %%
# === Model Initialization ===
# We support all genomic foundation models from Hugging Face Hub.

model = OmniModelForTokenClassification(
    model_name_or_path,
    tokenizer,
    label2id=label2id,  # 3 classes: (, ), .
)

print(f"✅ Model loaded: {model_name_or_path}")
print(f"📊 Model configuration:")
print(f"  - Architecture: Token-level classification")
print(f"  - Number of classes: {len(label2id)} (brackets and unpaired)")
print(f"  - Max sequence length: 512")
print(f"  - Model parameters: ~52M")

# %% [markdown]
# ## 🚀 Step 3: Model Training
# 
# Now comes the exciting part! We'll fine-tune our model to predict RNA secondary structures. During training, the model learns to associate sequence patterns with structural elements.
# 
# ### Our Training Strategy
# 
# We use a sophisticated approach optimized for token classification:
# 
# 1. **Evaluation Metrics**: For RNA structure prediction, we use:
#    - **F1-Score**: Balances precision and recall for structure prediction
#    - **Accuracy**: Overall correctness of token-level predictions
#    - **OmniGenBench supports 60+ ML metrics** for comprehensive evaluation
# 
# 2. **Advanced Training Features**:
#    - **Automatic mixed precision** for faster training
#    - **Gradient accumulation** for effective batch processing
#    - **Learning rate scheduling** with warmup
#    - **Early stopping** to prevent overfitting
# 
# The `AccelerateTrainer` provides a modern, efficient training pipeline with these optimizations built-in.

# %%
# Define evaluation metrics for token classification
metric_functions = [
    ClassificationMetric(ignore_y=-100).f1_score,
    ClassificationMetric(ignore_y=-100).accuracy,
]

# Initialize the modern AccelerateTrainer
trainer = AccelerateTrainer(
    model=model,
    train_dataset=datasets["train"],
    eval_dataset=datasets["valid"],
    test_dataset=datasets["test"],
    compute_metrics=metric_functions,
)

print("🎓 Starting training...")
print("⚡ Using AccelerateTrainer with automatic optimizations:")
print("  - Mixed precision training for speed and memory efficiency")
print("  - Automatic gradient accumulation")
print("  - Learning rate scheduling with warmup")
print("  - Early stopping based on validation metrics")

# Train the model
metrics = trainer.train()
trainer.save_model("ogb_rna_structure_finetuned")

print("✅ Training completed!")
print("📊 Final metrics:")
for metric_name, metric_value in metrics.items():
    if isinstance(metric_value, dict):
        print(f"  {metric_name}:")
        for k, v in metric_value.items():
            print(f"    {k}: {v:.4f}")
    else:
        print(f"  {metric_name}: {metric_value:.4f}")

# %% [markdown]
# ## 🔮 Step 4: Model Inference and Interpretation
# 
# Now that we have a trained model, let's use it for RNA secondary structure prediction on new sequences. This process demonstrates how the model can identify structural patterns in RNA sequences.
# 
# ### The Inference Pipeline
# 
# Our inference pipeline includes several key steps:
# 1. **Load the Model**: We load the best-performing model saved during training using ModelHub
# 2. **Process Input**: We take new RNA sequences and apply the same preprocessing steps
# 3. **Run Prediction**: We feed the processed sequence to the model and get structure predictions for each nucleotide
# 4. **Interpret Results**: We convert token predictions back to dot-bracket notation and visualize the predicted structure
# 
# Let's test our model on sample RNA sequences with different structural characteristics.

# %%
# Load the fine-tuned model for inference
inference_model = ModelHub.load("yangheng/ogb_rna_structure_finetuned")

# Test sequences with different structural characteristics
sample_sequences = {
    "Hairpin structure": "GCUGGGAUGUUGGCUUAGAAGCAGCCAUCAUUUAAAGAGUGCGUAACAGCUCACCAGC",
    "Complex structure": "AUCUGUACUAGUUAGCUAACUAGAUCUGUAUCUGGCGGUUCCGUGGAAGAACUGACGU",
    "Simple stem-loop": "GGGAAACCCUUUGGGAAACCC",
    "Linear sequence": "AUCGAUCGAUCGAUCGAUCGAUC",
}

# Define label mapping for interpretation
id2label = {v: k for k, v in label2id.items()}

with torch.no_grad():
    print("🔮 Running structure prediction on sample sequences...\n")
    
    for seq_name, sequence in sample_sequences.items():
        print(f"📊 Analysis for {seq_name}:")
        print(f"  📏 Sequence: {sequence}")
        print(f"  📏 Length: {len(sequence)} nucleotides")
        
        # Get predictions
        outputs = inference_model.inference(sequence)
        predictions = outputs.get('predictions', None)
        
        if predictions is not None:
            # Convert predictions to dot-bracket notation
            predicted_structure = ""
            for pred in predictions:
                if pred in id2label:
                    predicted_structure += id2label[pred]
                else:
                    predicted_structure += "."
            
            print(f"  🎯 Predicted structure: {predicted_structure}")
            
            # Analyze structural features
            left_brackets = predicted_structure.count("(")
            right_brackets = predicted_structure.count(")")
            unpaired = predicted_structure.count(".")
            
            print(f"  📈 Structural analysis:")
            print(f"    Paired bases: {min(left_brackets, right_brackets) * 2}")
            print(f"    Unpaired bases: {unpaired}")
            print(f"    Pairing efficiency: {min(left_brackets, right_brackets) * 2 / len(sequence):.2%}")
            
            # Structure quality assessment
            if abs(left_brackets - right_brackets) == 0:
                balance = "🟢 Balanced (valid structure)"
            elif abs(left_brackets - right_brackets) <= 2:
                balance = "🟡 Nearly balanced"
            else:
                balance = "🔴 Unbalanced (check prediction)"
            
            print(f"    Bracket balance: {balance}")
            
        print("─" * 60)

# %% [markdown]
# ### Advanced Analysis: Structure Visualization and Validation
# 
# Let's perform more detailed analysis including structure validation and potential applications of our predictions.

# %%
# Advanced structure analysis with detailed validation
def validate_structure(sequence, structure):
    """Validate RNA secondary structure prediction"""
    
    # Check for balanced brackets
    stack = []
    pairs = []
    
    for i, char in enumerate(structure):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                left_pos = stack.pop()
                pairs.append((left_pos, i))
            else:
                return False, "Unmatched closing bracket", pairs
    
    if stack:
        return False, "Unmatched opening bracket", pairs
        
    return True, "Valid structure", pairs

# Analyze a complex example
test_sequence = "GCUGGGAUGUUGGCUUAGAAGCAGCCAUCAUUUAAAGAGUGCGUAACAGCUCACCAGC"
print("🔬 Advanced Structure Analysis")
print("=" * 60)
print(f"Sequence: {test_sequence}")
print(f"Length: {len(test_sequence)} nucleotides")

# Get prediction
outputs = inference_model.inference(test_sequence)
predictions = outputs.get('predictions', None)

if predictions is not None:
    predicted_structure = "".join([id2label.get(pred, ".") for pred in predictions])
    
    print(f"Predicted: {predicted_structure}")
    
    # Validate structure
    is_valid, message, pairs = validate_structure(test_sequence, predicted_structure)
    
    print(f"\n🎯 Validation Results:")
    print(f"  Status: {'✅ Valid' if is_valid else '❌ Invalid'}")
    print(f"  Message: {message}")
    
    if pairs:
        print(f"  Base pairs found: {len(pairs)}")
        print(f"  Pairing details:")
        for i, (left, right) in enumerate(pairs[:5]):  # Show first 5 pairs
            left_base = test_sequence[left] if left < len(test_sequence) else "?"
            right_base = test_sequence[right] if right < len(test_sequence) else "?"
            print(f"    Pair {i+1}: {left_base}{left+1}-{right_base}{right+1}")
        
        if len(pairs) > 5:
            print(f"    ... and {len(pairs)-5} more pairs")
    
    print(f"\n🧬 Biological Insights:")
    print(f"  • This model captures local and global structural patterns")
    print(f"  • Predictions can guide RNA design and engineering")
    print(f"  • Structure affects RNA stability and function")
    print(f"  • Applications include drug discovery and synthetic biology")

print(f"\n🎉 Structure prediction completed successfully!")
print("🚀 Your model is ready for:")
print("  - RNA drug target identification")
print("  - Synthetic RNA design and optimization") 
print("  - Understanding RNA-protein interactions")
print("  - Advancing structural biology research")

# %% [markdown]
# ## 🎉 Tutorial Summary and Next Steps
# 
# Congratulations! You have successfully completed this comprehensive tutorial on RNA secondary structure prediction with OmniGenBench.
# 
# ### What You've Learned
# 
# You've walked through a complete, end-to-end MLOps workflow for token classification in structural biology. Specifically, you have:
# 
# 1. **Understood the "Why"**: Gained appreciation for the biological importance of RNA secondary structure and how Genomic Foundation Models provide powerful solutions for structural prediction.
# 
# 2. **Mastered the 4-Step Workflow**:
#    - **Step 1: Data Preparation**: You learned how to acquire, process, and efficiently load RNA structure datasets using the enhanced OmniDataset framework.
#    - **Step 2: Model Initialization**: You saw how to leverage pre-trained models and adapt them for token-level classification tasks.
#    - **Step 3: Model Training**: You implemented robust training strategies using AccelerateTrainer with proper evaluation metrics and modern optimizations.
#    - **Step 4: Model Inference**: You used your fine-tuned model to predict secondary structures and validated the biological meaningfulness of predictions.
# 
# 3. **Advanced Capabilities**: You explored:
#    - Token-level classification for position-specific structure prediction
#    - Structure validation and bracket balancing
#    - Biological interpretation of predicted structures
#    - Real-world applications in RNA engineering and drug discovery
# 
# ### Next Steps and Applications
# 
# Your trained model can now be applied to:
# - **Drug Discovery**: Identify RNA structural targets for therapeutic intervention
# - **Synthetic Biology**: Design RNA molecules with desired structural properties
# - **Biotechnology**: Engineer riboswitches and regulatory RNA elements
# - **Research**: Study structure-function relationships in RNA biology
# 
# ### Further Learning
# 
# Explore our other tutorials to expand your genomic AI toolkit:
# - **[mRNA Degradation Prediction](../mRNA_degrad_rate_regression/)**: Predict RNA stability and degradation rates
# - **[Translation Efficiency Prediction](../translation_efficiency_prediction/)**: Model protein production rates
# - **[Transcription Factor Binding](../tfb_prediction/)**: Understand gene regulation mechanisms
# 
# Thank you for following along. We hope this tutorial has provided you with the knowledge and confidence to apply deep learning to your own structural biology research. The future of computational biology is in your hands!
# 
# **Happy coding and discovering! 🧬✨**

# %% [markdown]
# ### Step 5: Define the Metrics
# We have implemented a diverse set of genomic metrics in OmniGenome, please refer to the documentation for more details.
# Users can also define their own metrics by inheriting the `OmniGenomeMetric` class. 
# The `compute_metrics` can be a metric function list and each metric function should return a dictionary of metrics.

# %%
compute_metrics = [
    ClassificationMetric(ignore_y=-100).accuracy_score,
    ClassificationMetric(ignore_y=-100, average="macro").f1_score,
    ClassificationMetric(ignore_y=-100).matthews_corrcoef,
]


# %% [markdown]
# ## Step 6: Define and Initialize the Trainer

# %%
# Initialize the MetricVisualizer for logging the metrics
mv = MetricVisualizer(name="OmniGenome-186M-SSP")

for seed in seeds:
    optimizer = torch.optim.AdamW(
        ssp_model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    trainer = Trainer(
        model=ssp_model,
        train_loader=train_loader,
        eval_loader=valid_loader,
        test_loader=test_loader,
        batch_size=batch_size,
        epochs=epochs,
        optimizer=optimizer,
        compute_metrics=compute_metrics,
        seeds=seed,
        device=autocuda.auto_cuda(),
    )

    metrics = trainer.train()
    test_metrics = metrics["test"][-1]
    mv.log(config_or_model.split("/")[-1], "F1", test_metrics["f1_score"])
    mv.log(
        config_or_model.split("/")[-1],
        "Accuracy",
        test_metrics["accuracy_score"],
    )
    print(metrics)
    mv.summary()

# %% [markdown]
# ### Step 7. Experimental Results Visualization
# The experimental results are visualized in the following plots. The plots show the F1 score and accuracy of the model on the test set for each run. The average F1 score and accuracy are also shown.

# %% [markdown]
# |### Step 8. Model Checkpoint for Sharing
# The model checkpoint can be saved and shared with others for further use. The model checkpoint can be loaded using the following code:

# %% [markdown]
# **Regular checkpointing and resuming are good practices to save the model at different stages of training.**

# %%
path_to_save = "OmniGenome-186M-SSP"
ssp_model.save(path_to_save, overwrite=True)

# Load the model checkpoint
ssp_model = ssp_model.load(path_to_save)
results = ssp_model.inference("CAGUGCCGAGGCCACGCGGAGAACGAUCGAGGGUACAGCACUA")
print(results["predictions"])
print("logits:", results["logits"])

# %% [markdown]
# 
# # What if someone doesn't know how to initialize the model?

# %%
# We can load the model checkpoint using the ModelHub
from omnigenbench import ModelHub

ssp_model = ModelHub.load("OmniGenome-186M-SSP")
results = ssp_model.inference("CAGUGCCGAGGCCACGCGGAGAACGAUCGAGGGUACAGCACUA")
print(results["predictions"])
print("logits:", results["logits"])

# %% [markdown]
# ## Step 8. Model Inference

# %%
examples = [
    "GCUGGGAUGUUGGCUUAGAAGCAGCCAUCAUUUAAAGAGUGCGUAACAGCUCACCAGC",
    "AUCUGUACUAGUUAGCUAACUAGAUCUGUAUCUGGCGGUUCCGUGGAAGAACUGACGUGUUCAUAUUCCCGACCGCAGCCCUGGGAGACGUCUCAGAGGC",
]

results = ssp_model.inference(examples)
structures = ["".join(prediction) for prediction in results["predictions"]]
print(results)
print(structures)

# %% [markdown]
# ### Step 9. Pipeline Creation
# The OmniGenome package provides pipelines for genomic FM development. The pipeline can be used to train, fine-tune, and evaluate genomic FMs. The pipeline can be used with a single command to train a genomic FM on a dataset. The pipeline can also be used to fine-tune a pre-trained genomic FM on a new dataset. The pipeline can be used to evaluate the performance of a genomic FM on a dataset. The pipeline can be used to generate predictions using a genomic FM.

# %%
# from omnigenbench import Pipeline, PipelineHub
# 
# pipeline = Pipeline(
#     name="OmniGenome-186M-SSP-Pipeline",
#     # config_or_model="OmniGenome-186M-SSP",  # The model name or path can be specified
#     # tokenizer="OmniGenome-186M-SSP",  # The tokenizer can be specified
#     config_or_model=ssp_model,
#     tokenizer=ssp_model.tokenizer,
#     datasets={
#         "train": "toy_datasets/train.json",
#         "test": "toy_datasets/test.json",
#         "valid": "toy_datasets/valid.json",
#     },
#     trainer=trainer,
#     device=ssp_model.model.device,
# )

# %% [markdown]
# ### Using the Pipeline

# %%
# results = pipeline(examples[0])
# print(results)
# 
# pipeline.train()
# 
# pipeline.save("OmniGenome-186M-SSP-Pipeline", overwrite=True)
# 
# pipeline = PipelineHub.load("OmniGenome-186M-SSP-Pipeline")
# results = pipeline(examples)
# print(results)

# %% [markdown]
# ## Web Demo for RNA Secondary Structure Prediction

# %%
import os
import time
import base64
import tempfile
from pathlib import Path
import json
import numpy as np
import gradio as gr
import RNA
from omnigenbench import ModelHub

# 加载模型
ssp_model = ModelHub.load("OmniGenome-186M-SSP")

# 临时 SVG 存储目录
TEMP_DIR = Path(tempfile.mkdtemp())
print(f"Using temporary directory: {TEMP_DIR}")


def ss_validity_loss(rna_strct: str) -> float:
    left = right = 0
    dots = rna_strct.count('.')
    for c in rna_strct:
        if c == '(':
            left += 1
        elif c == ')':
            if left:
                left -= 1
            else:
                right += 1
        elif c != '.':
            raise ValueError(f"Invalid char {c}")
    return (left + right) / (len(rna_strct) - dots + 1e-8)


def find_invalid_positions(struct: str) -> list:
    stack, invalid = [], []
    for i, c in enumerate(struct):
        if c == '(': stack.append(i)
        elif c == ')':
            if stack:
                stack.pop()
            else:
                invalid.append(i)
    invalid.extend(stack)
    return invalid


def generate_svg_datauri(rna_seq: str, struct: str) -> str:
    """生成 SVG 并返回 Base64 URI"""
    try:
        path = TEMP_DIR / f"{hash(rna_seq+struct)}.svg"
        RNA.svg_rna_plot(rna_seq, struct, str(path))
        time.sleep(0.1)
        svg_bytes = path.read_bytes()
        b64 = base64.b64encode(svg_bytes).decode('utf-8')
    except Exception as e:
        err = ('<svg xmlns="http://www.w3.org/2000/svg" width="400" height="200">'
               f'<text x="50" y="100" fill="red">Error: {e}</text></svg>')
        b64 = base64.b64encode(err.encode()).decode('utf-8')
    return f"data:image/svg+xml;base64,{b64}"


def fold(rna_seq: str, gt_struct: str):
    """展示 Ground Truth、ViennaRNA 与模型预测的结构对比"""
    if not rna_seq.strip():
        return "", "", "", ""
    # Ground Truth: 用户输入优先
    ground = gt_struct.strip() if gt_struct and gt_struct.strip() else ""
    gt_uri = generate_svg_datauri(rna_seq, ground) if ground else ""

    # ViennaRNA 预测
    vienna_struct, vienna_energy = RNA.fold(rna_seq)
    vienna_uri = generate_svg_datauri(rna_seq, vienna_struct)

    # 模型预测
    result = ssp_model.inference(rna_seq)
    pred = "".join(result.get('predictions', []))
    if ss_validity_loss(pred):
        for i in find_invalid_positions(pred):
            pred = pred[:i] + '.' + pred[i+1:]
    pred_uri = generate_svg_datauri(rna_seq, pred)

    # 统计信息
    match_gt = (sum(a==b for a,b in zip(ground, pred)) / len(ground)) if ground else 0
    match_vienna = sum(a==b for a,b in zip(vienna_struct, pred)) / len(vienna_struct)
    stats = (
        f"GT↔Pred Match: {match_gt:.2%}" + (" | " if ground else "") +
        f"Vienna↔Pred Match: {match_vienna:.2%}"
    )

    # 合并 HTML：三图水平排列
    combined = (
        '<div style="display:flex;justify-content:space-around;">'
        f'{f"<div><h4>Ground Truth</h4><img src=\"{gt_uri}\" style=\"max-width:100%;height:auto;\"/></div>" if ground else ""}'
        f'<div><h4>ViennaRNA</h4><img src=\"{vienna_uri}\" style=\"max-width:100%;height:auto;\"/></div>'
        f'<div><h4>Prediction</h4><img src=\"{pred_uri}\" style=\"max-width:100%;height:auto;\"/></div>'
        '</div>'
    )
    return ground, vienna_struct, pred, stats, combined


def sample_rna_sequence():
    try:
        exs = [json.loads(l) for l in open('toy_datasets/Archive2/test.json')]
        ex = exs[np.random.randint(len(exs))]
        return ex['seq'], ex.get('label','')
    except Exception as e:
        return f"Load Error: {e}", ""

# Gradio UI
with gr.Blocks(css="""
.heading {text-align:center;color:#2a4365;}
.controls {display:flex;gap:10px;margin:20px 0;}
.status {padding:10px;background:#f0f4f8;border-radius:4px;white-space:pre;}
""") as demo:
    gr.Markdown("# RNA Structure Pecdiction", elem_classes="heading")
    with gr.Row():
        rna_input = gr.Textbox(label="RNA Seqeuence", lines=3)
        structure_input = gr.Textbox(label="Ground Truth Structure (Optional)", lines=3)
    with gr.Row(elem_classes="controls"):
        sample_btn = gr.Button("Load Example")
        run_btn = gr.Button("Predict", variant="primary")
    stats_out    = gr.Textbox(label="Stats", interactive=False, elem_classes="status")
    gt_out       = gr.Textbox(label="Ground Truth", interactive=False)
    vienna_out   = gr.Textbox(label="ViennaRNA Structure", interactive=False)
    pred_out     = gr.Textbox(label="Prediction Structure", interactive=False)
    combined_view= gr.HTML(label="Visualization")

    run_btn.click(
        fold,
        inputs=[rna_input, structure_input],
        outputs=[gt_out, vienna_out, pred_out, stats_out, combined_view]
    )
    sample_btn.click(
        sample_rna_sequence,
        outputs=[rna_input, structure_input]
    )

    demo.launch(share=True)


# %% [markdown]
# ### Conclusion
# In this demonstration, we have shown how to fine-tune a genomic foundation model for RNA secondary structure prediction using the OmniGenome package. We have also shown how to use the trained model for inference and how to create a web demo for RNA secondary structure prediction. We hope this demonstration will help you get started with genomic foundation model development using OmniGenome.

# %%


