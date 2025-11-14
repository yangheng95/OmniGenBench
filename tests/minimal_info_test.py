#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Minimal test for dataset_info.json display functionality
"""

import json
import os

# Test reading dataset_info.json directly
dataset_info_path = "__OMNIGENBENCH_DATA__/benchmarks/GB/human_ensembl_regulatory/dataset_info.json"

if os.path.exists(dataset_info_path):
    print("=" * 80)
    print("Dataset Info File Content Test".center(80))
    print("=" * 80)
    
    with open(dataset_info_path, 'r', encoding='utf-8') as f:
        info = json.load(f)
    
    print(f"\n[SUCCESS] Loaded dataset_info.json from:")
    print(f"  {dataset_info_path}\n")
    
    # Display using tabulate
    try:
        from tabulate import tabulate
        
        # Basic Information
        print("\n" + "=" * 80)
        print("BASIC INFORMATION")
        print("=" * 80)
        basic_table = [
            ["Dataset Name", info.get("dataset_name", "N/A")],
            ["Version", info.get("version", "N/A")],
            ["Task Type", info.get("task_type", "N/A")],
            ["Species", info.get("species", "N/A")],
            ["Genome Type", info.get("genome_type", "N/A")],
        ]
        print(tabulate(basic_table, headers=["Field", "Value"], tablefmt="grid"))
        
        # Statistics
        if "statistics" in info:
            print("\n" + "=" * 80)
            print("STATISTICS")
            print("=" * 80)
            stats = info["statistics"]
            
            if "num_samples" in stats:
                samples_table = []
                for split, count in stats["num_samples"].items():
                    samples_table.append([split.capitalize(), count])
                print("\nSample Counts:")
                print(tabulate(samples_table, headers=["Split", "Count"], tablefmt="grid"))
            
            if "num_classes" in stats:
                print(f"\nNumber of Classes: {stats['num_classes']}")
            
            if "sequence_length" in stats:
                seq_len = stats["sequence_length"]
                print(f"\nSequence Length:")
                if isinstance(seq_len, dict):
                    for k, v in seq_len.items():
                        print(f"  {k}: {v}")
                else:
                    print(f"  {seq_len}")
        
        # Biological Significance
        if "biological_significance" in info:
            print("\n" + "=" * 80)
            print("BIOLOGICAL SIGNIFICANCE")
            print("=" * 80)
            print(info["biological_significance"])
        
        # Applications
        if "applications" in info:
            print("\n" + "=" * 80)
            print("APPLICATIONS")
            print("=" * 80)
            for i, app in enumerate(info["applications"], 1):
                print(f"{i}. {app}")
        
        # Citation
        if "citation" in info:
            print("\n" + "=" * 80)
            print("CITATION")
            print("=" * 80)
            print(info["citation"])
        
        print("\n" + "=" * 80)
        print("Test Completed Successfully!".center(80))
        print("=" * 80)
        
    except ImportError:
        print("\n[WARNING] tabulate not installed, showing raw JSON:")
        print(json.dumps(info, indent=2, ensure_ascii=False))

else:
    print(f"[ERROR] File not found: {dataset_info_path}")
