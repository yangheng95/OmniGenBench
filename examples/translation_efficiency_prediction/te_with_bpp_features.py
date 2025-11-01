# -*- coding: utf-8 -*-
"""
Translation Efficiency Prediction with BPP Features
====================================================

This example demonstrates how to predict translation efficiency (TE) by:
1. Using a custom dataset with prepare_input to compute BPP (Base Pairing Probability) matrices
2. Padding sequences to 512 length
3. Creating a custom model with linear layers to fuse BPP features with sequence embeddings

Author: YANG, HENG
"""

import os
import torch
import torch.nn as nn
import numpy as np
import ViennaRNA as RNA
from omnigenbench import (
    OmniTokenizer,
    OmniDatasetForSequenceClassification,
    OmniModelForSequenceClassification,
    AccelerateTrainer,
    ClassificationMetric,
)

# ============================================================================
# Step 1: Custom Dataset with BPP Calculation
# ============================================================================

class TEDatasetWithBPP(OmniDatasetForSequenceClassification):
    """
    Custom dataset that computes BPP (Base Pairing Probability) matrix 
    for each RNA sequence and includes it in the prepared input.
    """
    
    def __init__(self, dataset_name_or_path, tokenizer, max_length=512, **kwargs):
        """
        Initialize the dataset.
        
        Args:
            dataset_name_or_path: Path to dataset or dataset name
            tokenizer: Tokenizer for sequence encoding
            max_length: Maximum sequence length (default: 512)
        """
        self.max_length = max_length
        super().__init__(dataset_name_or_path, tokenizer, max_length, **kwargs)
        
    def compute_bpp_matrix(self, sequence):
        """
        Compute Base Pairing Probability (BPP) matrix using ViennaRNA.
        
        Args:
            sequence: RNA sequence string
            
        Returns:
            BPP matrix as numpy array of shape (seq_len, seq_len)
        """
        try:
            # Create fold compound
            fc = RNA.fold_compound(sequence)
            
            # Compute partition function (required for BPP)
            fc.pf()
            
            # Get base pairing probability matrix
            bpp_matrix = np.zeros((len(sequence), len(sequence)))
            
            # Extract base pairing probabilities
            for i in range(1, len(sequence) + 1):
                for j in range(i + 1, len(sequence) + 1):
                    prob = fc.pr_structure("(" * (j - i) + ")" * (j - i))
                    if prob > 0:
                        bpp_matrix[i-1, j-1] = prob
                        bpp_matrix[j-1, i-1] = prob
                        
            return bpp_matrix
            
        except Exception as e:
            print(f"Warning: Failed to compute BPP for sequence: {e}")
            # Return zero matrix on failure
            return np.zeros((len(sequence), len(sequence)))
    
    def compute_bpp_matrix_efficient(self, sequence):
        """
        More efficient BPP computation using ViennaRNA's bpp method.
        
        Args:
            sequence: RNA sequence string
            
        Returns:
            BPP matrix as numpy array
        """
        try:
            # Create fold compound and compute partition function
            fc = RNA.fold_compound(sequence)
            fc.pf()
            
            # Get base pair probability matrix directly
            bpp = fc.bpp()
            
            # Convert to numpy array
            seq_len = len(sequence)
            bpp_matrix = np.zeros((seq_len, seq_len), dtype=np.float32)
            
            # ViennaRNA bpp is 1-indexed
            for i in range(1, seq_len + 1):
                for j in range(i + 1, seq_len + 1):
                    if bpp[i][j] > 0:
                        bpp_matrix[i-1, j-1] = bpp[i][j]
                        bpp_matrix[j-1, i-1] = bpp[i][j]
            
            return bpp_matrix
            
        except Exception as e:
            print(f"Warning: Failed to compute BPP: {e}")
            return np.zeros((len(sequence), len(sequence)), dtype=np.float32)
    
    def prepare_input(self, instance, **kwargs):
        """
        Prepare input with BPP features.
        
        This method:
        1. Extracts sequence and label from instance
        2. Pads/truncates sequence to max_length (512)
        3. Computes BPP matrix
        4. Tokenizes sequence
        5. Returns all features including BPP matrix
        
        Args:
            instance: Dictionary with 'sequence' and 'label' keys
            
        Returns:
            Dictionary with tokenized inputs and BPP matrix
        """
        # Extract sequence and label
        if isinstance(instance, str):
            sequence = instance
            labels = -100
        elif isinstance(instance, dict):
            sequence = instance.get("sequence", instance.get("seq"))
            label = instance.get("label", instance.get("labels"))
            labels = label if label is not None else -100
        else:
            raise ValueError("Unknown instance format")
        
        # Convert T to U for RNA
        sequence = sequence.replace("T", "U").replace("t", "u")
        
        # Pad or truncate sequence to max_length
        if len(sequence) > self.max_length:
            sequence = sequence[:self.max_length]
        elif len(sequence) < self.max_length:
            # Pad with 'N' (or you can use a specific padding nucleotide)
            sequence = sequence + "N" * (self.max_length - len(sequence))
        
        # Compute BPP matrix for the sequence
        bpp_matrix = self.compute_bpp_matrix_efficient(sequence)
        
        # Pad BPP matrix to (max_length, max_length)
        if bpp_matrix.shape[0] < self.max_length:
            padded_bpp = np.zeros((self.max_length, self.max_length), dtype=np.float32)
            padded_bpp[:bpp_matrix.shape[0], :bpp_matrix.shape[1]] = bpp_matrix
            bpp_matrix = padded_bpp
        
        # Tokenize sequence
        tokenized_inputs = self.tokenizer(
            sequence,
            padding=kwargs.get("padding", "max_length"),
            truncation=kwargs.get("truncation", True),
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        # Squeeze to remove batch dimension
        for col in tokenized_inputs:
            tokenized_inputs[col] = tokenized_inputs[col].squeeze()
        
        # Add BPP matrix to inputs
        tokenized_inputs["bpp_matrix"] = torch.tensor(bpp_matrix, dtype=torch.float32)
        
        # Add labels
        if labels != -100:
            if isinstance(labels, str):
                labels = self.label2id.get(str(labels), -100)
            if not isinstance(labels, int):
                raise ValueError("Label must be an integer for sequence classification")
        
        tokenized_inputs["labels"] = torch.tensor(labels)
        
        return tokenized_inputs


# ============================================================================
# Step 2: Custom Model with BPP Feature Fusion
# ============================================================================

class TEModelWithBPPFusion(OmniModelForSequenceClassification):
    """
    Custom model that fuses BPP features with sequence embeddings.
    
    Architecture:
    1. Transformer encoder processes sequence
    2. CNN processes BPP matrix
    3. Linear layers fuse both features
    4. Final classification layer
    """
    
    def __init__(self, config_or_model, tokenizer, num_labels=2, **kwargs):
        """
        Initialize model with BPP fusion capability.
        
        Args:
            config_or_model: Path to pre-trained model
            tokenizer: Tokenizer for sequence encoding
            num_labels: Number of classification labels
        """
        super().__init__(config_or_model, tokenizer, num_labels=num_labels, **kwargs)
        
        # Get hidden size from the base model
        hidden_size = self.model.config.hidden_size
        
        # BPP matrix processor (2D CNN)
        self.bpp_processor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 512 -> 256
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 256 -> 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
        )
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size + 128, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # Replace the original classifier
        self.classifier = nn.Linear(256, num_labels)
        
        # Initialize new layers
        self._init_fusion_layers()
    
    def _init_fusion_layers(self):
        """Initialize weights for new layers."""
        for module in [self.bpp_processor, self.fusion, self.classifier]:
            for layer in module.modules():
                if isinstance(layer, (nn.Linear, nn.Conv2d)):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
    
    def forward(self, input_ids=None, attention_mask=None, bpp_matrix=None, labels=None, **kwargs):
        """
        Forward pass with BPP feature fusion.
        
        Args:
            input_ids: Token IDs from tokenizer
            attention_mask: Attention mask
            bpp_matrix: Base pairing probability matrix
            labels: Classification labels
            
        Returns:
            Dictionary with loss and logits
        """
        # Process sequence with transformer
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        
        # Get sequence representation (use [CLS] token or mean pooling)
        sequence_features = outputs.last_hidden_state[:, 0, :]  # [batch, hidden_size]
        
        # Process BPP matrix if provided
        if bpp_matrix is not None:
            # Add channel dimension: [batch, seq_len, seq_len] -> [batch, 1, seq_len, seq_len]
            bpp_input = bpp_matrix.unsqueeze(1)
            
            # Process with CNN
            bpp_features = self.bpp_processor(bpp_input)  # [batch, 128, 1, 1]
            bpp_features = bpp_features.view(bpp_features.size(0), -1)  # [batch, 128]
            
            # Concatenate sequence and BPP features
            fused_features = torch.cat([sequence_features, bpp_features], dim=1)
        else:
            # If no BPP matrix, pad with zeros
            batch_size = sequence_features.size(0)
            zero_bpp = torch.zeros(batch_size, 128, device=sequence_features.device)
            fused_features = torch.cat([sequence_features, zero_bpp], dim=1)
        
        # Apply fusion layers
        fused_features = self.fusion(fused_features)
        
        # Classification
        logits = self.classifier(fused_features)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return {
            "loss": loss,
            "logits": logits,
            "last_hidden_state": outputs.last_hidden_state,
        }


# ============================================================================
# Step 3: Training Script
# ============================================================================

def main():
    """Main training function."""
    
    print("=" * 80)
    print("Translation Efficiency Prediction with BPP Features")
    print("=" * 80)
    
    # Configuration
    config_or_model = "yangheng/PlantRNA-FM"
    dataset_name = "translation_efficiency_prediction"
    max_length = 512
    batch_size = 4
    epochs = 10
    learning_rate = 2e-5
    
    print(f"\n📋 Configuration:")
    print(f"  Model: {config_or_model}")
    print(f"  Max Length: {max_length}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning Rate: {learning_rate}")
    
    # Step 1: Initialize tokenizer
    print(f"\n🔧 Step 1: Loading tokenizer...")
    tokenizer = OmniTokenizer(config_or_model)
    print(f"  ✅ Tokenizer loaded")
    
    # Step 2: Load datasets with BPP computation
    print(f"\n📊 Step 2: Loading datasets with BPP features...")
    
    # Define label mapping
    label2id = {"0": 0, "1": 1}
    
    try:
        # Try to load from local directory
        train_dataset = TEDatasetWithBPP(
            dataset_name_or_path=f"{dataset_name}/train.json",
            tokenizer=tokenizer,
            max_length=max_length,
            label2id=label2id,
        )
        
        valid_dataset = TEDatasetWithBPP(
            dataset_name_or_path=f"{dataset_name}/valid.json",
            tokenizer=tokenizer,
            max_length=max_length,
            label2id=label2id,
        )
        
        test_dataset = TEDatasetWithBPP(
            dataset_name_or_path=f"{dataset_name}/test.json",
            tokenizer=tokenizer,
            max_length=max_length,
            label2id=label2id,
        )
        
        print(f"  ✅ Loaded datasets:")
        print(f"     Train: {len(train_dataset)} samples")
        print(f"     Valid: {len(valid_dataset)} samples")
        print(f"     Test: {len(test_dataset)} samples")
        
    except Exception as e:
        print(f"  ⚠️  Warning: Could not load dataset: {e}")
        print(f"  💡 Tip: Make sure dataset files exist in '{dataset_name}/' directory")
        print(f"  Creating demo with synthetic data for illustration...")
        
        # Create synthetic demo data
        demo_data = []
        for i in range(10):
            seq = "AUGCAUGCAUGC" * 40  # ~480 nucleotides
            label = i % 2
            demo_data.append({"sequence": seq, "label": label})
        
        # Save to temp files
        import json
        os.makedirs(dataset_name, exist_ok=True)
        for split, data in [("train", demo_data[:6]), ("valid", demo_data[6:8]), ("test", demo_data[8:])]:
            with open(f"{dataset_name}/{split}.json", "w") as f:
                for item in data:
                    f.write(json.dumps(item) + "\n")
        
        # Reload datasets
        train_dataset = TEDatasetWithBPP(
            dataset_name_or_path=f"{dataset_name}/train.json",
            tokenizer=tokenizer,
            max_length=max_length,
            label2id=label2id,
        )
        valid_dataset = TEDatasetWithBPP(
            dataset_name_or_path=f"{dataset_name}/valid.json",
            tokenizer=tokenizer,
            max_length=max_length,
            label2id=label2id,
        )
        test_dataset = TEDatasetWithBPP(
            dataset_name_or_path=f"{dataset_name}/test.json",
            tokenizer=tokenizer,
            max_length=max_length,
            label2id=label2id,
        )
        print(f"  ✅ Demo datasets created")
    
    # Step 3: Initialize model with BPP fusion
    print(f"\n🤖 Step 3: Initializing model with BPP fusion...")
    model = TEModelWithBPPFusion(
        config_or_model=config_or_model,
        tokenizer=tokenizer,
        num_labels=2,
        label2id=label2id,
        dataset_class=TEDatasetWithBPP,  # Pass dataset class for inference
    )
    print(f"  ✅ Model initialized with BPP fusion layers")
    print(f"     Architecture: Transformer + BPP CNN + Fusion Layers")
    
    # Step 4: Setup training
    print(f"\n🎯 Step 4: Setting up trainer...")
    
    metric_functions = [
        ClassificationMetric().f1_score,
        ClassificationMetric().accuracy,
        ClassificationMetric().precision,
        ClassificationMetric().recall,
    ]
    
    trainer = AccelerateTrainer(
        model=model,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        test_dataset=test_dataset,
        metric_functions=metric_functions,
        model_save_path="te_bpp_finetuned",
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        patience=3,
    )
    
    print(f"  ✅ Trainer configured")
    
    # Step 5: Train model
    print(f"\n🚀 Step 5: Training model...")
    print(f"  This may take a while depending on your hardware...")
    
    trainer.train()
    
    print(f"\n✅ Training completed!")
    print(f"   Model saved to: te_bpp_finetuned/")
    
    # Step 6: Evaluate on test set
    print(f"\n📈 Step 6: Evaluating on test set...")
    test_results = trainer.evaluate(test_dataset)
    
    print(f"\n📊 Test Results:")
    for metric_name, value in test_results.items():
        print(f"   {metric_name}: {value:.4f}")
    
    # Step 7: Demo inference
    print(f"\n🔮 Step 7: Demo inference with BPP features...")
    
    model.eval()
    sample_seq = "AUGCAUGCAUGCAUGCAUGC" * 20  # Sample sequence
    
    with torch.no_grad():
        # The model will use dataset_class to prepare input with BPP
        outputs = model.inference({"sequence": sample_seq})
        
        prediction = outputs.get("predictions", [0])[0]
        confidence = outputs.get("confidence", 0.5)
        
        print(f"\n   Sample sequence: {sample_seq[:50]}...")
        print(f"   Prediction: {'High TE' if prediction == 1 else 'Low TE'}")
        print(f"   Confidence: {confidence:.4f}")
    
    print(f"\n{'='*80}")
    print(f"✨ Example completed successfully!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
