#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RNA Sequence Design Example
============================

This script demonstrates various use cases of the RNA design functionality
in OmniGenBench, from simple hairpin structures to complex multi-loop designs.

Author: OmniGenBench Team
Version: 0.3.22alpha
"""

import json
from pathlib import Path
from omnigenbench import OmniModelForRNADesign


def example_1_simple_hairpin():
    """
    Example 1: Design a simple hairpin structure
    
    A hairpin is one of the most basic RNA secondary structures,
    consisting of a stem (base-paired region) and a loop (unpaired region).
    """
    print("=" * 70)
    print("Example 1: Simple Hairpin Design")
    print("=" * 70)
    
    # Define target structure: 3-base-pair stem with 3-nucleotide loop
    structure = "(((...)))"
    print(f"\nTarget Structure: {structure}")
    print(f"Description: 3bp stem with 3nt loop")
    print(f"Total length: {len(structure)} nucleotides\n")
    
    # Initialize the model
    print("Loading model...")
    model = OmniModelForRNADesign(model="yangheng/OmniGenome-186M")
    
    # Design sequences with default parameters
    print("Designing sequences...\n")
    sequences = model.design(
        structure=structure,
        mutation_ratio=0.5,
        num_population=100,
        num_generation=50
    )
    
    # Display results
    print(f"\n[SUCCESS] Found {len(sequences)} sequence(s):\n")
    for i, seq in enumerate(sequences[:10], 1):  # Show top 10
        print(f"  {i:2d}. {seq}")
    
    return sequences


def example_2_custom_parameters():
    """
    Example 2: Design with custom evolutionary parameters
    
    This example shows how to tune the genetic algorithm parameters
    for better exploration vs exploitation trade-offs.
    """
    print("\n" + "=" * 70)
    print("Example 2: Custom Parameters for Complex Structure")
    print("=" * 70)
    
    # More complex structure: stem-loop-stem
    structure = "(((..(((...)))..)))"
    print(f"\nTarget Structure: {structure}")
    print(f"Description: Nested stem-loops")
    print(f"Total length: {len(structure)} nucleotides\n")
    
    model = OmniModelForRNADesign(model="yangheng/OmniGenome-186M")
    
    # Use more aggressive exploration for complex structures
    print("Designing with custom parameters:")
    print("  - Population: 200 (larger for more diversity)")
    print("  - Generations: 100 (more iterations)")
    print("  - Mutation ratio: 0.3 (conservative mutations)")
    print()
    
    sequences = model.design(
        structure=structure,
        mutation_ratio=0.3,      # Lower mutation for stability
        num_population=200,      # Larger population for diversity
        num_generation=100       # More generations for convergence
    )
    
    print(f"\n[SUCCESS] Found {len(sequences)} sequence(s):\n")
    for i, seq in enumerate(sequences[:5], 1):  # Show top 5
        print(f"  {i}. {seq}")
    
    return sequences


def example_3_batch_design():
    """
    Example 3: Batch design multiple structures
    
    Demonstrates efficient batch processing of multiple target structures,
    useful for large-scale RNA design projects.
    """
    print("\n" + "=" * 70)
    print("Example 3: Batch Design Multiple Structures")
    print("=" * 70)
    
    # Define a set of target structures
    structures = {
        "hairpin": "(((...)))",
        "stem_loop": "(((..)))",
        "multi_loop": "(((..(((...)))..(((...))).)))",
        "long_stem": "((((....))))",
    }
    
    print(f"\nDesigning {len(structures)} different structures...\n")
    
    # Initialize model once for efficiency
    model = OmniModelForRNADesign(model="yangheng/OmniGenome-186M")
    
    # Design each structure
    results = {}
    for name, structure in structures.items():
        print(f"[INFO] Designing '{name}': {structure}")
        sequences = model.design(
            structure=structure,
            mutation_ratio=0.5,
            num_population=100,
            num_generation=50
        )
        results[name] = {
            "structure": structure,
            "sequences": sequences[:5],  # Store top 5
            "count": len(sequences)
        }
        print(f"   [SUCCESS] Found {len(sequences)} solutions\n")
    
    # Save results to file
    output_file = Path("batch_design_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"[INFO] All results saved to: {output_file}\n")
    
    return results


def example_4_validate_designs():
    """
    Example 4: Validate designed sequences
    
    Shows how to verify that designed sequences actually fold into
    the target structure using ViennaRNA.
    """
    print("\n" + "=" * 70)
    print("Example 4: Validate Designed Sequences")
    print("=" * 70)
    
    try:
        import ViennaRNA
    except ImportError:
        print("\n⚠️  ViennaRNA not available for validation")
        print("Install with: pip install ViennaRNA")
        return
    
    structure = "(((...)))"
    print(f"\nTarget Structure: {structure}\n")
    
    model = OmniModelForRNADesign(model="yangheng/OmniGenome-186M")
    sequences = model.design(
        structure=structure,
        num_population=50,
        num_generation=30
    )
    
    print(f"Validating {min(5, len(sequences))} designed sequences...\n")
    
    valid_count = 0
    for i, seq in enumerate(sequences[:5], 1):
        # Predict structure using ViennaRNA
        predicted_structure, mfe = ViennaRNA.fold(seq)
        
        # Check if it matches target
        is_valid = predicted_structure == structure
        valid_count += is_valid
        
        status = "[OK]" if is_valid else "[FAIL]"
        print(f"{status} Sequence {i}: {seq}")
        print(f"   Predicted: {predicted_structure}")
        print(f"   Target:    {structure}")
        print(f"   MFE: {mfe:.2f} kcal/mol")
        print()
    
    accuracy = (valid_count / min(5, len(sequences))) * 100
    print(f"Validation Accuracy: {accuracy:.1f}% ({valid_count}/{min(5, len(sequences))})")


def example_5_save_and_load():
    """
    Example 5: Save designs and reload for analysis
    
    Demonstrates best practices for saving and organizing design results.
    """
    print("\n" + "=" * 70)
    print("Example 5: Save and Load Design Results")
    print("=" * 70)
    
    structure = "((((....))))"
    print(f"\nTarget Structure: {structure}\n")
    
    model = OmniModelForRNADesign(model="yangheng/OmniGenome-186M")
    sequences = model.design(
        structure=structure,
        num_population=100,
        num_generation=50
    )
    
    # Prepare comprehensive results
    results = {
        "metadata": {
            "structure": structure,
            "structure_length": len(structure),
            "model": "yangheng/OmniGenome-186M",
            "description": "Long stem hairpin design"
        },
        "parameters": {
            "mutation_ratio": 0.5,
            "num_population": 100,
            "num_generation": 50
        },
        "results": {
            "num_sequences": len(sequences),
            "sequences": sequences,
            "top_5": sequences[:5]
        }
    }
    
    # Save with descriptive filename
    output_file = Path(f"rna_design_{structure.replace('(', 'L').replace(')', 'R').replace('.', 'U')}.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"[SUCCESS] Saved {len(sequences)} sequences to: {output_file}")
    
    # Reload and display
    with open(output_file, "r") as f:
        loaded = json.load(f)
    
    print(f"\n[INFO] Loaded results:")
    print(f"   Structure: {loaded['metadata']['structure']}")
    print(f"   Sequences found: {loaded['results']['num_sequences']}")
    print(f"   Top sequence: {loaded['results']['top_5'][0]}")
    
    return output_file


def main():
    """
    Run all examples demonstrating RNA design functionality.
    """
    print("\n" + "=" * 70)
    print("RNA Sequence Design Examples - OmniGenBench")
    print("=" * 70)
    print("\nThis script demonstrates various RNA design use cases:")
    print("  1. Simple hairpin design")
    print("  2. Custom parameters for complex structures")
    print("  3. Batch design multiple structures")
    print("  4. Validate designed sequences")
    print("  5. Save and load design results")
    print("\n" + "=" * 70 + "\n")
    
    try:
        # Run examples
        example_1_simple_hairpin()
        example_2_custom_parameters()
        example_3_batch_design()
        example_4_validate_designs()
        example_5_save_and_load()
        
        print("\n" + "=" * 70)
        print("[SUCCESS] All examples completed successfully!")
        print("=" * 70 + "\n")
        
    except Exception as e:
        print(f"\n[ERROR] Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
