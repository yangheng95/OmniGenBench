#%% md
# # 🧬 Translation Efficiency Prediction with Foundation Model
# 
# Welcome to this comprehensive tutorial where we'll explore how to predict **Translation Efficiency (TE)** from mRNA sequences using **OmniGenBench** and **PlantRNA-FM** (Plant RNA Foundation Model). This guide demonstrates the practical application of plant-specialized genomic deep learning for rice translation efficiency prediction.
# 
# > 📚 **Prerequisite**: If you are new to OmniGenBench, it is highly recommended to first study the **[Fundamental Concepts Tutorial](https://github.com/yangheng95/OmniGenBench/blob/master/examples/00_fundamental_concepts.ipynb)**, which covers general knowledge such as language model concepts, machine learning task classification, and foundation model principles—particularly how PlantRNA-FM works.
# 
# ### 1. The Biological Challenge: Translation Efficiency in Plants
# 
# **Translation** is one of the most fundamental processes in molecular biology - it's the mechanism by which cells read mRNA sequences and synthesize proteins. **Translation Efficiency (TE)** in plants quantifies how effectively this process occurs for a given plant mRNA molecule, directly impacting protein production levels in plant cells.
# 
# Understanding and predicting TE in plants has profound implications across multiple domains:
# - **Synthetic Biology & Crop Engineering**: Designing optimized gene circuits for precise protein expression in transgenic plants
# - **Agricultural Biotechnology**: Engineering crop plants for enhanced nutritional content (e.g., Golden Rice, protein-enriched grains)
# - **Stress Tolerance**: Understanding how plants regulate translation under stress conditions (drought, heat, salinity)
# - **Yield Improvement**: Optimizing expression of yield-related genes through codon optimization and UTR engineering
# - **Plant Defense**: Enhancing production of defense proteins and secondary metabolites
# 
# However, experimentally measuring TE across thousands of plant mRNA sequences is time-consuming and costly. This is where **PlantRNA-FM** (published in *Nature Machine Intelligence*, 35M parameters), a plant-specialized foundation model, can provide transformative solutions by learning plant-specific translation patterns with remarkable efficiency.
# 
# ### 2. The Data: Rice Translation Efficiency Dataset
# 
# To train our predictive model, we utilize a carefully curated dataset from *Oryza sativa* (rice), a model organism in plant biology.
# 
# - **What it contains**: mRNA sequences with experimentally determined translation efficiency measurements
# - **What it labels**: Each sequence is classified as either "High TE" (1) or "Low TE" (0) based on ribosome profiling data
# - **Our Goal**: Train a model that can accurately classify any rice mRNA sequence by its translation efficiency potential
# 
# **Dataset Structure:**
# 
# | sequence | label | 
# |---------|-------|
# | AUGGCCAUUGUAAUUGGCCGACUUGA... | 1 (High TE) | 
# | AUGGCUACUAGCUAGCUAGCUAGC...    | 0 (Low TE) | 
# | ...                                | ...  | 
# 
# Find the dataset template in **[Dataset Template](https://github.com/yangheng95/OmniGenBench/blob/master/examples/translation_efficiency_prediction/05_advanced_dataset_creation.ipynb)** and customize it as needed for your experiments.
# 
# ### 3. Quick Start: Translation Efficiency Prediction Workflow
# 
# This tutorial demonstrates the practical application of the **[Fundamental Concepts](https://github.com/yangheng95/OmniGenBench/blob/master/examples/00_fundamental_concepts.ipynb)** to a specific biological problem. We'll use the standard 4-step OmniGenBench workflow:
#%%
# from IPython.display import Image, display
# display(Image(filename="4-step workflow.png"))
#%% md
# 
# **Translation Efficiency Prediction** is a **sequence classification** task where we predict binary labels (High TE vs Low TE) for plant mRNA sequences. We'll use `OmniModelForSequenceClassification` with **PlantRNA-FM** to leverage plant-specific codon usage patterns and RNA structural features.
# 
# ### 4. Tutorial Structure
# 
# 1. **[Data Preparation](https://github.com/yangheng95/OmniGenBench/blob/master/examples/translation_efficiency_prediction/01_data_preparation.ipynb)**: Download and preprocess the rice translation efficiency dataset
# 2. **[Model Initialization](https://github.com/yangheng95/OmniGenBench/blob/master/examples/translation_efficiency_prediction/02_model_initialization.ipynb)**: Load PlantRNA-FM and set it up for binary classification of plant mRNA sequences
# 3. **[Training Implementation](https://github.com/yangheng95/OmniGenBench/blob/master/examples/translation_efficiency_prediction/03_model_training.ipynb)**: Fine-tune PlantRNA-FM using rice TE data and validate its performance
# 4. **[Inference: Make Predictions](https://github.com/yangheng95/OmniGenBench/blob/master/examples/translation_efficiency_prediction/04_model_inference.ipynb)**: Use the trained model to predict translation efficiency on new plant mRNA sequences
# 
# Let's get started!
#%% md
# ## 🚀 Step 1: Data Preparation
# 
# This first step is all about getting our data ready for in-silico analysis. It involves four key parts:
# 1.  **Environment Setup**: Installing and importing the necessary libraries.
# 2.  **Configuration**: Defining all our important parameters in one place.
# 3.  **Data Acquisition**: Downloading and preparing the raw dataset.
# 4.  **Data Loading**: Creating a pipeline to efficiently feed data to the model.
# 
# ### 1.1: Environment Setup
# 
# First, let's install the required Python packages. `omnigenbench` is our core library, `transformers` provides the underlying model architecture, and the other packages are utilities for our workflow.
#%%
# !pip install omnigenbench -U
#%% md
# Next, we import the libraries we just installed. This gives us the tools for data processing, deep learning, and interacting with the operating system.
# 
# A key part of this setup is determining the best available hardware for training. Our script will automatically prioritize a **CUDA-enabled GPU** if one is available, as this can accelerate training by 10-100x compared to a CPU. This makes a huge difference when working with large models and datasets.
#%%
from omnigenbench import (
    ClassificationMetric,
    AccelerateTrainer,
    ModelHub,
    OmniTokenizer,
    OmniDatasetForSequenceClassification,
    OmniModelForSequenceClassification,
)

#%% md
# ### 1.2: Global Configuration
# 
# To make our tutorial easy to modify and understand, we'll centralize all important parameters in this section. This is a best practice in software development that makes experiments more reproducible.
# 
# #### Key Parameters
# -   **Dataset**: We define the local path and download URL for our dataset.
# -   **Model**: We select which pre-trained foundation model to use. For this tutorial, we'll start with `PlantRNA-FM` (Hugging Face: `yangheng/PlantRNA-FM`) because it's fast and efficient, making it perfect for learning and prototyping.
# 
# This centralized approach allows you to easily experiment with different settings (e.g., a larger model, a different learning rate) without having to hunt through the code.
# 
# #### Note
# Almost all the parameters here are standard in machine learning workflows and have a default value that works well if you don't set them. Don't worry if some of these terms are unfamiliar right now. We'll explain each one as we go along.
#%%
model_name_or_path = "yangheng/PlantRNA-FM"  # Plant RNA Foundation Model
dataset_name = "translation_efficiency_prediction"
#%% md
# ### 1.3: Data Acquisition
# 
# With our environment configured, it's time to download the DeepSEA dataset. The function below automates this process by:
# 1.  Checking if the data already exists.
# 2.  Downloading the dataset from the specified URL if needed.
# 3.  Extracting the files.
# 4.  Cleaning up the temporary zip file.
# 
# This ensures we have the `train.jsonl`, `valid.jsonl`, and `test.jsonl` files ready for the next stage. These files represent the standard splits for training, validating, and testing our model.
#%%
# Model and Tokenizer

# We define the label mapping in the training
label2id = {"0": 0, "1": 1}  # 0: Low TE, 1: High TE

# Initialize tokenizer
tokenizer = OmniTokenizer.from_pretrained(model_name_or_path)

datasets = OmniDatasetForSequenceClassification.from_hub(
    dataset_name_or_path="translation_efficiency_prediction",
    tokenizer=tokenizer,
    max_length=512,
    label2id=label2id,
)
#%% md
# ### 1.4: Dataset Loading with OmniGenBench
# 
# With OmniGenBench, data loading is significantly simplified! The framework automatically handles:
# 
# #### Automatic Data Processing
# The `OmniDatasetForSequenceClassification` class automatically:
# 1. **Downloads and processes** the dataset from our curated collection
# 2. **Handles sequence preprocessing** including truncation, padding, and tokenization
# 3. **Manages binary classification formatting** for translation efficiency prediction
# 4. **Creates train/validation/test splits** ready for training
# 
# This streamlined approach eliminates the need for custom dataset classes while maintaining full flexibility and performance.
#%%
print(f"📊 Loaded datasets: {list(datasets.keys())}")
for split, dataset in datasets.items():
    print(f"  - {split}: {len(dataset)} samples")
#%% md
# ## 🚀 Step 2: Model Initialization with PlantRNA-FM
# 
# With our data pipeline in place, it's time to set up **PlantRNA-FM** (Plant RNA Foundation Model). Instead of building a model from scratch, we'll load the pre-trained PlantRNA-FM and adapt it for our rice translation efficiency task.
# 
# This process involves three key components:
# 1.  **The Tokenizer**: Converts plant RNA sequences into a numerical format PlantRNA-FM can process. It's crucial that we use the tokenizer specifically designed for PlantRNA-FM.
# 2.  **PlantRNA-FM Base Model**: This plant-specialized model has learned fundamental patterns of plant RNA sequences, codon usage, and regulatory elements from extensive plant transcriptome data.
# 3.  **The Classification Head**: We add a classification head on top of PlantRNA-FM that maps plant RNA representations to our binary labels (High TE vs Low TE).
# 
# The `OmniModelForSequenceClassification` class seamlessly combines PlantRNA-FM with the task-specific classification head for rice translation efficiency prediction.
#%%
# === Model Initialization with PlantRNA-FM ===
# Using PlantRNA-FM for plant-specific translation efficiency prediction

model = OmniModelForSequenceClassification(
    model_name_or_path,  # PlantRNA-FM
    tokenizer,
    num_labels=len(list(label2id.keys())),  # Binary classification: Low TE vs High TE
)

print(f"✅ Loaded PlantRNA-FM for rice translation efficiency prediction")

#%% md
# ## 🚀 Step 3: Fine-tuning PlantRNA-FM
# 
# This is the most exciting part! With our data and PlantRNA-FM ready, we can now begin the **fine-tuning** process. During training, PlantRNA-FM will adapt its plant-specific knowledge to learn the relationship between rice mRNA sequence features and translation efficiency.
# 
# The `AccelerateTrainer` from `omnigenbench` handles the training process efficiently, allowing us to fine-tune PlantRNA-FM with just a few lines of code.
#%%
metric_functions = [ClassificationMetric().f1_score]

trainer = AccelerateTrainer(
    model=model,
    train_dataset=datasets["train"],
    eval_dataset=datasets["valid"],
    test_dataset=datasets["test"],
    compute_metrics=metric_functions,
)
print("🎓 Starting training...")

metrics = trainer.train()
trainer.save_model("ogb_te_finetuned")

print('Metrics:', metrics)
#%% md
# ## 🔮 Step 4: Model Inference and Interpretation
# 
# Now that we have a trained model, let's use it for its intended purpose: predicting translation efficiency on new mRNA sequences. This process is called **inference**.
# 
# ### The Inference Pipeline
# 
# Our inference pipeline consists of a few key steps:
# 1. **Load the Model**: We load the best-performing model that was saved during training.
# 2. **Process the Input**: We take new mRNA sequences and apply the same preprocessing steps we used for our training data (truncating/padding and tokenizing).
# 3. **Run Prediction**: We feed the processed sequence to the model and get its predictions. We use `torch.no_grad()` to disable gradient calculations, which makes inference faster and uses less memory.
# 4. **Interpret the Results**: The model's raw output is a probability score. We'll interpret these to make them more understandable, identifying whether sequences have high or low translation efficiency and with what level of confidence.
# 
# To demonstrate, we'll test our model on a few sample sequences and print out a user-friendly summary of the results. This shows how the model can be used in a real-world application to analyze sequences of interest.
#%%

inference_model = ModelHub.load("yangheng/ogb_te_finetuned")

sample_sequences = {
    "Optimized sequence": "AAACCAACAAAATGCAGTAGAAGTACTCTCGAGCTATAGTCGCGACGTGCTGCCCCGCAGGAGTACAGTAGTAGTACAACGTAAGCGGGAGCAACAGACTCCCCCCCTGCAACCCACTGTGCCTGTGCCCTCGACGCGTCTCCGTCGCTTTGGCAAATGTCACGTACATATTACCGTCTCAGGCTCTCAGCCATGCTCCCTACCACCCCTGCAGCGAAGCAAAAGCCACGCACGCGGCGCCTGACATGTAACAGGACTAGACCATCTTGTTCATTTCCCGCACCCCCTCCTCTCCTCTTCCTCCATCTGCCTCTTTAAAACAGTAAAAATAACCGTGCATCCCCTGGGCAAAATCTCTCCCATACATACACTACAGCGGCGAACCTTTCCTTATTCTCGCAACGCCTCGGTAACGGGCAGCGCCTGCTCCGCGCCGCGGTTGCGAGTTCGGGAAGGCGGCCGGAGTCGCGGGGAGGAGAGGGAGGATTCGATCGGCCAGA",
    "Suboptimal sequence": "TGGAGATGGGCAGATGGCACACAAAACATGAATAGAAAACCCAAAAGGAAGGATGAAAAAAACACACACACACACACACACAAAACACAGAGAGAGAGAGAGAGAGAGCGAGAAAAGAAAAGAAAAAACCAATTCTTTTGGTCTCTTCCCTCTCCGTTTGTCGTGTCGAAGCCTTTGCCCCCACCACCTCCTCCTCTCCTCTCCCTTCCTCCCCTCCTCCCCATCTCGCTCTCCTCCCTCCTCTCTCCTCTCCTCGTCTCCTCTTCCTCTCCATTCCATTGGCCATTCCATTCCATTCCACCCCCCATGAAACCCCAAACCCTCGTCGGCCTCGCCGCGCTCGCGTAGCGCACCCGCCCTTCTCCTCTCGCCGGTGGTCCGCCGCCAGCCTCCCCCCACCCGATCCCGCCGCCCCCCCCGCCTTCACCCCGCCCACGCGGACGCATCCGATCCCGCCGCATCGCCGCGCGGGGGGGGGGGGGGGGGGGGGGGGGAGGGCACG",
    "Random sequence": "AUGC" * (128 // 4),
}
for seq_name, sequence in sample_sequences.items():
    outputs = inference_model.inference(sequence)

    # —— Result Interpretation ——
    prediction = outputs['predictions']
    confidence = outputs['confidence']
    print(f"  - Predicted Translation Efficiency: {prediction} (Confidence: {confidence:.2f})")

#%% md
# ## 🎉 Tutorial Summary and Next Steps
# 
# Congratulations! You have successfully completed this comprehensive tutorial on translation efficiency prediction with OmniGenBench.
# 
# ### What You've Learned
# 
# You've walked through a complete, end-to-end application of genomic deep learning, demonstrating how the concepts from the **[Fundamental Concepts Tutorial](../00_fundamental_concepts.ipynb)** apply to a real biological problem. Specifically, you have:
# 
# 1. **Applied Task Formulation**: Successfully framed translation efficiency prediction as a sequence classification problem
# 2. **Mastered the 4-Step Workflow**:
#    - **Step 1: Data Preparation**: Acquired, processed, and loaded the rice translation efficiency dataset
#    - **Step 2: Model Initialization**: Set up OmniModelForSequenceClassification for binary classification
#    - **Step 3: Model Training**: Fine-tuned the model using best practices and appropriate evaluation metrics
#    - **Step 4: Model Inference**: Generated predictions on new mRNA sequences and interpreted results
# 3. **Understood Practical Application**: Gained hands-on experience with a biologically relevant prediction task
# 
# ### 🚀 Next Steps
# 
# Now that you've mastered translation efficiency prediction, you can:
# 
# #### 🧬 **Explore Other Sequence Classification Tasks**
# - **Promoter Recognition**: Identify regulatory sequences
# - **Subcellular Localization**: Predict protein cellular destinations  
# - **Functional Annotation**: Classify protein or RNA functions
# 
# #### 📊 **Try Different Task Types**
# - **Sequence Regression**: Gene expression level prediction
# - **Token Classification**: Binding site identification
# - **Multi-label Classification**: Multi-functional sequence prediction
# 
# #### 🔬 **Advanced Techniques**
# - **Custom Dataset Creation**: Use the [Advanced Dataset Creation Tutorial](./05_advanced_dataset_creation.ipynb)
# - **Model Comparison**: Benchmark different foundation models
# - **Hyperparameter Optimization**: Fine-tune model performance
# - **Biological Validation**: Compare predictions with experimental data
# 
# ### 📚 Resources
# 
# - **[Fundamental Concepts Tutorial](../00_fundamental_concepts.ipynb)**: Review core concepts anytime
# - **[OmniGenBench Documentation](https://omnigenbench.readthedocs.io/)**: Complete API reference
# - **[GitHub Repository](https://github.com/yangheng95/OmniGenBench)**: Source code and community discussions
# 
# Thank you for following along. We hope this tutorial has provided you with the knowledge and confidence to apply deep learning to your own genomics research. Happy coding!
#%% md
# 