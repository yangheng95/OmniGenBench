# -*- coding: utf-8 -*-
"""
Attention Score Extraction Example

This example demonstrates how to use the attention extraction functionality
that is now available in ALL OmniModel classes through EmbeddingMixin.

Key Feature: ALL OmniModel types support embedding and attention extraction!
   - OmniModelForEmbedding
   - OmniModelForSequenceClassification  
   - OmniModelForSequenceRegression
   - OmniModelForTokenClassification
   - And all other OmniModel variants!
"""

import torch
import os
import sys

# Add the omnigenbench to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omnigenbench import (
    OmniModelForEmbedding,
    OmniModelForSequenceClassification,
    OmniModelForSequenceRegression,
)


def main():
    """Main function demonstrating attention extraction capabilities."""

    print("OmniGenome Attention Extraction Demo")
    print("=" * 50)
    print("Note: ALL OmniModel types support embedding and attention extraction!")
    print("=" * 50)

    # Initialize model (using a smaller model for demo)
    print("\n[INFO] Loading model...")
    model_name = "yangheng/OmniGenome-186M"  # Standard OmniGenBench model
    
    print("[INFO] You can use ANY OmniModel type for embedding and attention extraction:")
    print("  - OmniModelForEmbedding (dedicated)")
    print("  - OmniModelForSequenceClassification (classification + embeddings)")
    print("  - OmniModelForSequenceRegression (regression + embeddings)")
    print("  - OmniModelForTokenClassification (token classification + embeddings)")
    print("  - And all other OmniModel variants!")
    
    try:
        # Option 1: Use dedicated embedding model (Recommended for embedding-focused tasks)
        model = OmniModelForEmbedding(model_name, trust_remote_code=True)
        
        # Option 2: Use classification model (also supports attention extraction!)
        # Note: Use the constructor directly, not .from_pretrained()
        # model = OmniModelForSequenceClassification(
        #     model=model_name,
        #     num_labels=2,  # Specify task parameters
        #     trust_remote_code=True
        # )
        
        # Option 3: Use regression model (also supports attention extraction!)
        # model = OmniModelForSequenceRegression(
        #     model=model_name,
        #     trust_remote_code=True
        # )
        
        print(f"✅ Model loaded successfully: {model_name}")
        print(f"✅ Model type: {type(model).__name__}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("💡 Make sure you have internet connection and the model exists")
        return

    # Example genomic sequences
    sequences = [
        "ATCGATCGATCGTAGCTAGCTAGCT",
        "GGCCTTAACCGGTTAACCGGTTAA",
        "TTTTAAAACCCCGGGGTTTTAAAA"
    ]

    print(f"\n🧬 Test sequences:")
    for i, seq in enumerate(sequences):
        print(f"  Sequence {i+1}: {seq}")

    # 1. Extract attention scores from a single sequence
    print("\n" + "="*50)
    print("1️⃣  Single Sequence Attention Extraction")
    print("="*50)

    sequence = sequences[0]
    print(f"🔍 Analyzing sequence: {sequence}")

    try:
        # Extract attention scores
        attention_result = model.extract_attention_scores(
            sequence=sequence,
            max_length=128,
            layer_indices=None,  # Extract all layers
            head_indices=None,   # Extract all heads
            return_on_cpu=True
        )

        print(f"✅ Attention extraction successful!")
        print(f"📊 Attention tensor shape: {attention_result['attentions'].shape}")
        print(f"🔤 Number of tokens: {len(attention_result['tokens'])}")
        print(f"🎯 First 10 tokens: {attention_result['tokens'][:10]}")

        # Show attention statistics
        stats = model.get_attention_statistics(
            attention_result['attentions'],
            attention_result['attention_mask']
        )

        print(f"📈 Attention Statistics:")
        print(f"  - Attention matrix shape: {stats['attention_matrix'].shape}")
        print(f"  - Average attention entropy: {stats['attention_entropy'].mean():.4f}")
        print(f"  - Max attention concentration: {stats['attention_concentration'].max():.4f}")

    except Exception as e:
        print(f"❌ Error in attention extraction: {e}")

    # 2. Batch attention extraction
    print("\n" + "="*50)
    print("2️⃣  Batch Attention Extraction")
    print("="*50)

    try:
        batch_results = model.batch_extract_attention_scores(
            sequences=sequences,
            batch_size=2,
            max_length=128,
            layer_indices=[0, -1],  # First and last layer only
            head_indices=[0, 1, 2], # First 3 heads only
            return_on_cpu=True
        )

        print(f"✅ Batch attention extraction successful!")
        print(f"📊 Number of sequences processed: {len(batch_results)}")

        for i, result in enumerate(batch_results):
            print(f"  Sequence {i+1} attention shape: {result['attentions'].shape}")

    except Exception as e:
        print(f"❌ Error in batch attention extraction: {e}")

    # 3. Attention visualization (if matplotlib is available)
    print("\n" + "="*50)
    print("3️⃣  Attention Visualization")
    print("="*50)

    try:
        # Try to visualize attention pattern
        fig = model.visualize_attention_pattern(
            attention_result=attention_result,
            layer_idx=0,
            head_idx=0,
            save_path="attention_heatmap.png",
            figsize=(10, 8)
        )

        if fig is not None:
            print("✅ Attention heatmap generated and saved as 'attention_heatmap.png'")
        else:
            print("⚠️  Visualization skipped (matplotlib not available)")

    except Exception as e:
        print(f"❌ Error in visualization: {e}")

    # 4. Compare attention patterns between sequences
    print("\n" + "="*50)
    print("4️⃣  Attention Pattern Comparison")
    print("="*50)

    try:
        # Extract attention from multiple sequences for comparison
        comparison_results = []
        for i, seq in enumerate(sequences[:2]):  # Compare first two sequences
            result = model.extract_attention_scores(
                sequence=seq,
                max_length=128,
                layer_indices=[0],  # Just first layer
                head_indices=[0],   # Just first head
                return_on_cpu=True
            )
            comparison_results.append(result)

            # Get statistics
            stats = model.get_attention_statistics(
                result['attentions'],
                result['attention_mask']
            )

            print(f"🧬 Sequence {i+1}: {seq[:20]}...")
            print(f"  - Attention entropy: {stats['attention_entropy'].mean():.4f}")
            print(f"  - Max self-attention: {stats['self_attention_scores'].max():.4f}")
            print(f"  - Attention concentration: {stats['attention_concentration'].mean():.4f}")

        print("✅ Attention comparison completed!")

    except Exception as e:
        print(f"❌ Error in attention comparison: {e}")

    # 5. Advanced usage: Extract specific layers and heads
    print("\n" + "="*50)
    print("5️⃣  Advanced: Specific Layer/Head Extraction")
    print("="*50)

    try:
        # Extract only middle layers and specific heads
        advanced_result = model.extract_attention_scores(
            sequence=sequences[0],
            max_length=128,
            layer_indices=[2, 3, 4],  # Middle layers (adjust based on model size)
            head_indices=[0, 4, 8],   # Specific heads (adjust based on model)
            return_on_cpu=True
        )

        print(f"✅ Advanced extraction successful!")
        print(f"📊 Filtered attention shape: {advanced_result['attentions'].shape}")
        print(f"📋 Shape explanation: (layers={advanced_result['attentions'].shape[0]}, "
              f"heads={advanced_result['attentions'].shape[1]}, "
              f"seq_len={advanced_result['attentions'].shape[2]})")

    except Exception as e:
        print(f"❌ Error in advanced extraction: {e}")
        print("💡 This might happen if the specified layer/head indices don't exist in the model")

    print("\n" + "="*50)
    print("🎉 Demo completed!")
    print("="*50)
    print("\n📝 Summary of embedding and attention features:")
    print("  1. extract_attention_scores() - Extract attention from single sequence")
    print("  2. batch_extract_attention_scores() - Batch processing")
    print("  3. get_attention_statistics() - Compute attention statistics")
    print("  4. visualize_attention_pattern() - Create attention heatmaps")
    print("  5. encode() - Generate sequence embeddings")
    print("  6. batch_encode() - Batch embedding generation")
    print("  7. encode_tokens() - Token-level embeddings")
    print("  8. compute_similarity() - Compute embedding similarity")
    print("\n💡 These functions are available in ALL OmniModel types:")
    print("  ✅ OmniModelForEmbedding")
    print("  ✅ OmniModelForSequenceClassification")
    print("  ✅ OmniModelForSequenceRegression")
    print("  ✅ OmniModelForTokenClassification")
    print("  ✅ OmniModelForMLM")
    print("  ✅ All other OmniModel variants!")
    print("\n🎯 This means you can:")
    print("  - Do classification AND extract embeddings from the same model")
    print("  - Do regression AND analyze attention patterns from the same model")
    print("  - Use any fine-tuned model for embedding extraction")
    print("  - Supports:")
    print("    - Filtering specific layers and attention heads")
    print("    - Batch processing for efficiency")
    print("    - GPU/CPU memory management")
    print("    - Statistical analysis of attention patterns")
    print("    - Visualization capabilities")


if __name__ == "__main__":
    main()
