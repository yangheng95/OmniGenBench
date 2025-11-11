# -*- coding: utf-8 -*-
# file: test_example_notebooks.py
# time: 00:00 01/11/2025
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# Copyright (C) 2019-2025. All Rights Reserved.

"""
Test suite for converting and executing example notebooks.
Converts .ipynb files to .py scripts and validates they run correctly.
Excludes tfb_prediction and variant_effect_prediction as requested.
"""

import os
import sys
import subprocess
import pytest
import json
import re
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent
EXAMPLES_DIR = PROJECT_ROOT / "examples"

# List of notebooks to test (excluding tfb and vep)
NOTEBOOKS_TO_TEST = [
    "00_fundamental_concepts.ipynb",
    "attention_score_extraction/Attention_Analysis_Tutorial.ipynb",
    "autobench_gfm_evaluation/benchmarking_with_lora.ipynb",
    "genomic_data_augmentation/RNA_Augmentation_Tutorial.ipynb",
    "genomic_embeddings/RNA_Embedding_Tutorial.ipynb",
    "mRNA_degrad_rate_regression/mRNA_degrade_regression.ipynb",
    "rna_secondary_structure_prediction/Secondary_Structure_Prediction_Tutorial.ipynb",
    "rna_secondary_structure_prediction/ZeroShot_Structure_Prediction_Tutorial.ipynb",
    "rna_sequence_design/RNA_Design_Tutorial.ipynb",
    "translation_efficiency_prediction/01_data_preparation.ipynb",
    "translation_efficiency_prediction/02_model_initialization.ipynb",
    "translation_efficiency_prediction/03_model_training.ipynb",
    "translation_efficiency_prediction/04_model_inference.ipynb",
    "translation_efficiency_prediction/05_advanced_dataset_creation.ipynb",
    "translation_efficiency_prediction/quickstart_te.ipynb",
]

# Notebooks that require significant compute time or resources
# These will only have syntax validation, not full execution
SLOW_NOTEBOOKS = [
    "autobench_gfm_evaluation/benchmarking_with_lora.ipynb",  # Runs actual training
    "translation_efficiency_prediction/03_model_training.ipynb",  # Runs training
    "mRNA_degrad_rate_regression/mRNA_degrade_regression.ipynb",  # Runs training
    "translation_efficiency_prediction/01_data_preparation.ipynb",  # Downloads real data
    "translation_efficiency_prediction/quickstart_te.ipynb",  # Full workflow with training
    "rna_secondary_structure_prediction/Secondary_Structure_Prediction_Tutorial.ipynb",  # May download data
]


def is_vscode_notebook(notebook_path: Path) -> bool:
    """
    Check if a notebook is in VSCode's XML format.
    
    Args:
        notebook_path: Path to the notebook file
        
    Returns:
        True if VSCode format, False otherwise
    """
    with open(notebook_path, 'r', encoding='utf-8') as f:
        first_line = f.readline()
        return '<VSCode.Cell' in first_line or '<VSCode.Cell' in f.read(1000)


def extract_python_from_vscode_notebook(notebook_path: Path) -> str:
    """
    Extract Python code from VSCode notebook format.
    
    Args:
        notebook_path: Path to the .ipynb file in VSCode format
        
    Returns:
        Python code as string
    """
    with open(notebook_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract all Python cells using regex
    pattern = r'<VSCode\.Cell[^>]*language="python"[^>]*>(.*?)</VSCode\.Cell>'
    python_cells = re.findall(pattern, content, re.DOTALL)
    
    # Join all Python code
    python_code = '\n\n'.join(cell.strip() for cell in python_cells)
    
    return python_code


def convert_notebook_to_script(notebook_path: Path) -> Path:
    """
    Convert a Jupyter notebook to a Python script using nbconvert.
    
    Args:
        notebook_path: Path to the .ipynb file
        
    Returns:
        Path to the generated .py file
    """
    script_path = notebook_path.with_suffix('.py')
    
    # Check if it's a VSCode notebook format
    if is_vscode_notebook(notebook_path):
        # Extract Python code from VSCode format
        try:
            python_code = extract_python_from_vscode_notebook(notebook_path)
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(python_code)
            return script_path
        except Exception as e:
            pytest.skip(f"Could not extract Python from VSCode notebook {notebook_path}: {e}")
    
    # Use jupyter nbconvert to convert notebook to script
    cmd = [
        'jupyter', 'nbconvert',
        '--to', 'python',
        '--output', script_path.stem,
        str(notebook_path)
    ]
    
    try:
        result = subprocess.run(
            cmd,
            cwd=notebook_path.parent,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Failed to convert notebook: {result.stderr}")
            
        return script_path
        
    except Exception as e:
        pytest.skip(f"Could not convert notebook {notebook_path}: {e}")


def clean_script_for_testing(script_path: Path) -> None:
    """
    Clean up the generated Python script to make it testable.
    Removes interactive elements and notebook magic commands.
    
    Args:
        script_path: Path to the .py file
    """
    with open(script_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    cleaned_lines = []
    skip_next = False
    
    for line in lines:
        # Skip IPython magic commands
        if line.strip().startswith('%') or line.strip().startswith('!'):
            continue
        # Skip get_ipython() calls
        if 'get_ipython()' in line:
            continue
        # Skip display() and Image() calls (from IPython.display)
        if 'display(' in line or line.strip().startswith('display('):
            continue
        # Skip interactive input
        if 'input()' in line:
            continue
        # Comment out IPython.display imports but keep other imports
        if 'from IPython.display import' in line or 'import IPython.display' in line:
            line = '# ' + line  # Comment it out instead of removing
        
        cleaned_lines.append(line)
    
    with open(script_path, 'w', encoding='utf-8') as f:
        f.writelines(cleaned_lines)


@pytest.mark.parametrize("notebook_rel_path", NOTEBOOKS_TO_TEST)
def test_notebook_conversion_and_execution(notebook_rel_path: str):
    """
    Test that notebooks can be converted to Python scripts and executed without errors.
    For slow notebooks (training-heavy), only syntax validation is performed.
    
    Args:
        notebook_rel_path: Relative path to notebook from examples/ directory
    """
    notebook_path = EXAMPLES_DIR / notebook_rel_path
    
    # Check notebook exists
    if not notebook_path.exists():
        pytest.skip(f"Notebook not found: {notebook_path}")
    
    print(f"\n[INFO] Testing notebook: {notebook_rel_path}")
    
    # Convert notebook to script
    script_path = convert_notebook_to_script(notebook_path)
    
    try:
        # Clean the script for testing
        clean_script_for_testing(script_path)
        
        # Check if this is a slow notebook that should skip execution
        if notebook_rel_path in SLOW_NOTEBOOKS:
            print(f"[INFO] Slow notebook detected - performing syntax check only")
            with open(script_path, 'r', encoding='utf-8') as f:
                code = f.read()
            try:
                compile(code, str(script_path), 'exec')
                print(f"[SUCCESS] Syntax validation passed")
                return
            except SyntaxError as e:
                pytest.fail(f"Syntax error in converted script: {e}")
        
        # Try to import/execute the script to check for errors
        print(f"[INFO] Validating script: {script_path}")
        
        # Run the script with Python to check for errors
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=script_path.parent,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode != 0:
            print(f"[ERROR] Script execution failed:")
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")
            pytest.fail(f"Script execution failed with return code {result.returncode}")
        else:
            print(f"[SUCCESS] Script executed successfully")
            
    finally:
        # Cleanup: optionally remove the generated script
        # Uncomment the next line if you want to remove scripts after testing
        # if script_path.exists():
        #     script_path.unlink()
        pass


@pytest.mark.parametrize("notebook_rel_path", NOTEBOOKS_TO_TEST)
def test_notebook_syntax_only(notebook_rel_path: str):
    """
    Test that notebooks have valid Python syntax when converted to scripts.
    This is a lighter test that doesn't execute the code.
    
    Args:
        notebook_rel_path: Relative path to notebook from examples/ directory
    """
    notebook_path = EXAMPLES_DIR / notebook_rel_path
    
    if not notebook_path.exists():
        pytest.skip(f"Notebook not found: {notebook_path}")
    
    print(f"\n[INFO] Checking syntax for: {notebook_rel_path}")
    
    # Convert notebook to script
    script_path = convert_notebook_to_script(notebook_path)
    
    try:
        # Clean the script
        clean_script_for_testing(script_path)
        
        # Check syntax by compiling
        with open(script_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        try:
            compile(code, str(script_path), 'exec')
            print(f"[SUCCESS] Syntax is valid")
        except SyntaxError as e:
            pytest.fail(f"Syntax error in converted script: {e}")
            
    finally:
        # Keep the script for inspection if syntax check fails
        pass


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
