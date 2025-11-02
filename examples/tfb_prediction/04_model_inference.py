#%% md
# ##  TFB Prediction Tutorial 4/4: Model Inference - From Trained Model to Predictions
# 
# Welcome to the final tutorial in our series on fine-tuning a Genomic Foundation Model. We have come a long way:
# 
# 1.  We prepared the DeepSEA dataset ([01_data_preparation.ipynb](https://github.com/yangheng95/OmniGenBench/blob/master/examples/tfb_prediction/01_data_preparation.ipynb)).
# 2.  We initialized a model architecture suitable for our task ([02_model_initialization.ipynb](https://github.com/yangheng95/OmniGenBench/blob/master/examples/tfb_prediction/02_model_initialization.ipynb)).
# 3.  We trained the model and saved the best-performing checkpoint ([03_model_training.ipynb](https://github.com/yangheng95/OmniGenBench/blob/master/examples/tfb_prediction/03_model_training.ipynb)).
# 
# Now, we have a powerful, fine-tuned model stored as `best_model.pth`. But a model is only useful if we can use it to make predictions on new, unseen data. This process is called **inference**.
# 
# In this tutorial, we will cover:
# 1.  **The Inference Pipeline**: Understanding the essential steps for getting predictions.
# 2.  **Inference with `ModelHub`**: The easy, one-line way to load and predict with `OmniGenBench`.
# 3.  **Manual Inference**: A more hands-on approach for custom data processing workflows.
# 4.  **Final Evaluation**: Assessing our model's performance on the held-out test set.
# 5.  **Deployment Concepts**: A brief look at how to serve your model as an API.
# 
# By the end, you will be able to use your trained model to predict transcription factor binding sites and understand how to integrate it into larger applications.
#%% md
# ### 1. The Inference Pipeline
# 
# Making a prediction with a trained model involves a clear, logical sequence of steps. It's crucial that the data processing during inference **exactly matches** the processing used during training.
# 
# The pipeline looks like this:
# 
# 1.  **Load Model and Tokenizer**: You must load the exact model checkpoint (`best_model.pth`) that you saved during training. You also need the same tokenizer that was used to prepare the training data.
# 2.  **Prepare Input**: Take a new, raw DNA sequence.
# 3.  **Tokenize**: Use the loaded tokenizer to convert the DNA sequence into `input_ids` and an `attention_mask`, just as we did for the training data. This includes applying the same `max_length`, padding, and truncation strategies.
# 4.  **Predict**: Pass the tokenized input through the model to get the raw output scores (logits).
# 5.  **Post-process**: Convert the logits into a more interpretable format, such as probabilities (by applying a Sigmoid function) or binary predictions (by applying a threshold like 0.5).
# 
# **`OmniGenBench` simplify all the processes and provides a unified interface for inference.**
#%% md
# ### 2. Inference with `ModelHub`: The Easy Way
# 
# For many standard use cases, `OmniGenBench` offers a high-level `ModelHub` API that encapsulates the entire inference pipeline. It allows you to load a fine-tuned model and get predictions with a single line of code.
# 
# `ModelHub` automatically handles loading the correct model architecture, the tokenizer, and the saved weights from your checkpoint directory.
# 
# Let's see it in action.
#%%
# Load the trained model using ModelHub - matches complete tutorial
from omnigenbench import ModelHub

# Load the trained model
inference_model = ModelHub.load("yangheng/ogb_tfb_finetuned")

# Define sample sequences for testing - matches complete tutorial style
sample_sequences = {
    "Random sequence": "AGCT" * (128 // 4),
    "AT-rich sequence": "AATT" * (128 // 4),
    "GC-rich sequence": "GCGC" * (128 // 4),
}

print("🧬 Testing model on sample DNA sequences:")
print("=" * 50)
print(inference_model.inference(sample_sequences["Random sequence"]))
print("=" * 50)
print(inference_model.inference(sample_sequences["AT-rich sequence"]))
print("=" * 50)
print(inference_model.inference(sample_sequences["GC-rich sequence"]))

#%% md
# This approach is incredibly convenient for standard models and tasks. You simply point `ModelHub` to your checkpoint directory, and it takes care of the rest.
#%% md
# ### 3. Manual Inference: For Custom Workflows
# 
# While `ModelHub` is convenient, sometimes you need more control. For example, you might have a custom data loading pipeline or want to perform inference on a large dataset more efficiently. In these cases, you can perform the inference steps manually.
# 
# This involves loading the model and tokenizer yourself and then explicitly tokenizing the input before passing it to the model. This is essentially what `ModelHub` does under the hood.
# 
# #### 3.1. Setup: Re-initializing Model, Tokenizer, and Data
# 
# First, let's set up our environment again. We need to load the necessary libraries and define our configuration. We also need our test dataloader to evaluate the model on the full test set.
#%%
# Import libraries for manual inference
import torch
import numpy as np
import os
from omnigenbench import (
    OmniTokenizer,
    OmniModelForMultiLabelSequenceClassification,
    OmniDatasetForMultiLabelClassification
)

# Configuration for inference - matches complete tutorial
model_name_or_path = "yangheng/OmniGenome-52M"
dataset_name = "deepsea_tfb_prediction"
num_labels = 919
max_length = 512
device = "cuda" if torch.cuda.is_available() else "cpu"

print("✅ Libraries imported and configuration set!")
print(f"🎯 Device: {device}")
print(f"🧬 Model: {model_name_or_path}")
print(f"📊 Labels: {num_labels} TF binding sites")
#%%
# Load tokenizer and test dataset for evaluation
print("🔄 Loading tokenizer...")
tokenizer = OmniTokenizer.from_pretrained(model_name_or_path)
print(f"✅ Tokenizer loaded")

print("📊 Loading test dataset...")
datasets = OmniDatasetForMultiLabelClassification.from_hub(
    dataset_name_or_path=dataset_name,
    tokenizer=tokenizer,
    max_length=max_length,
    max_examples=100,  # Small subset for demonstration
    force_padding=False
)

print(f"🧪 Test dataset: {len(datasets['test'])} samples")

# For manual inference, we can load the model architecture and weights
# In practice, you would load your trained checkpoint here
print("🔄 Loading model for manual inference...")
model = OmniModelForMultiLabelSequenceClassification(
    model_name_or_path,
    tokenizer,
    num_labels=num_labels,
)
model.to(device)
model.eval()
print("✅ Model loaded and ready for manual inference!")
print(f"📊 Model has {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")
#%% md
# #### 3.2. Loading the Fine-Tuned Model
# 
# Now, we load the model. We first initialize the model architecture using `OmniModelForMultiLabelSequenceClassification` and then load our fine-tuned weights from the `best_model.pth` file.
#%%
# Quick demonstration with ModelHub - matches complete tutorial approach
print("🔮 Quick Inference with ModelHub:")
print("=" * 40)

# Use ModelHub for easy inference
model_hub_inference = ModelHub.load("yangheng/ogb_tfb_finetuned")

# Test with a sample sequence
test_sequence = "GATTACAGATTACAGATTACA" * 10  # Create a longer test sequence

print(f"🧬 Test sequence length: {len(test_sequence)} bp")
print("🔄 Running inference...")

# Make prediction
result = model_hub_inference.inference(test_sequence)
print("✅ Inference completed!")

if hasattr(result, 'shape'):
    print(f"📊 Output shape: {result.shape}")
    print(f"🎯 Predicted {result.shape[-1]} TF binding probabilities")
else:
    print(f"📊 Result type: {type(result)}")

print("🚀 ModelHub provides easy one-line inference!")
#%% md
# #### 3.3. The Inference Loop
# 
# With the model and dataloader ready, we can now loop through the test set, make predictions, and store the results. This is the core of manual inference.
#%%
# Manual inference on test dataset
print("🔍 Manual Inference on Test Dataset:")
print("=" * 45)

# Get a few test samples for demonstration
test_samples = [datasets['test'][i] for i in range(min(3, len(datasets['test'])))]

all_predictions = []
all_labels = []

print("🔄 Processing test samples...")

with torch.no_grad():  # Disable gradient calculations for efficiency
    for i, sample in enumerate(test_samples):
        # Prepare input
        input_ids = sample['input_ids'].unsqueeze(0).to(device)  # Add batch dimension
        attention_mask = sample['attention_mask'].unsqueeze(0).to(device)
        true_labels = sample['labels']
        
        # Run inference
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.squeeze(0)  # Remove batch dimension
        
        # Convert to probabilities
        probabilities = torch.sigmoid(logits)
        
        # Apply threshold for binary predictions
        predictions = (probabilities > 0.5).float()
        
        print(f"\n📋 Sample {i+1}:")
        print(f"   🧬 Sequence length: {input_ids.shape[1]} tokens")
        print(f"   🎯 Predicted binding sites: {predictions.sum().item():.0f}/{len(predictions)}")
        print(f"   🏷️ True binding sites: {true_labels.sum().item():.0f}/{len(true_labels)}")
        print(f"   📈 Max probability: {probabilities.max().item():.3f}")
        print(f"   📉 Min probability: {probabilities.min().item():.3f}")
        
        all_predictions.append(predictions.cpu().numpy())
        all_labels.append(true_labels.numpy())

print(f"\n✅ Manual inference completed on {len(test_samples)} samples!")
print("🎯 This demonstrates the full inference pipeline")
#%% md
# This final score gives you a reliable estimate of how well your model will perform in a real-world scenario on new genomic data.
#%% md
# ### 4. Deployment Concepts: Serving Your Model
# 
# A trained model in a notebook is great for research, but for real-world applications, you'll want to **deploy** it as a service. This typically means wrapping it in a web API. A popular choice for this is **FastAPI**.
# 
# The concept is simple:
# 1.  **Create a FastAPI App**: A simple Python script that defines API endpoints.
# 2.  **Load the Model**: In the app's startup logic, you would load your fine-tuned model and tokenizer once (e.g., using the `ModelHub` or the manual method).
# 3.  **Define a Prediction Endpoint**: Create an endpoint (e.g., `/predict`) that accepts a DNA sequence as input.
# 4.  **Process and Predict**: Inside the endpoint function, you would call the model's inference method with the input sequence and return the prediction as a JSON response.
# 
# While a full deployment tutorial is beyond our current scope, the `ModelHub` API is designed to make this transition as smooth as possible.
# 
# ### Summary and Conclusion
# 
# This tutorial marks the end of our journey from a biological question to a fully trained and evaluated model. We have covered the complete lifecycle: data preparation, model initialization, training, and now, inference.
# 
# You have learned how to:
# -   Understand the end-to-end inference pipeline.
# -   Use the high-level `ModelHub` for quick and easy predictions.
# -   Perform manual inference for greater control and batch processing.
# -   Evaluate the final model on a test set to get a definitive performance metric.
# -   Conceptualize how to deploy your model as a service.
# 
# You are now equipped with the fundamental skills to tackle your own genomic prediction tasks using `OmniGenBench`. You can return to the main tutorial or explore the other examples in the repository to learn about different tasks and models. Happy modeling!
#%% md
# 