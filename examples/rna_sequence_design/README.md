# RNA Sequence Design Examples

This directory contains examples and tutorials for designing RNA sequences that fold into specific secondary structures using OmniGenBench.

## Overview

RNA sequence design is the inverse problem of RNA structure prediction: given a target secondary structure (in dot-bracket notation), design one or more RNA sequences that fold into that structure. OmniGenBench uses a genetic algorithm enhanced with masked language modeling to efficiently explore the sequence space.

## Files in this Directory

- **`rna_design_examples.py`**: Comprehensive examples demonstrating various use cases
- **`easy_rna_design_emoo.py`**: Original implementation with multi-objective optimization
- **`web_rna_design.py`**: Web interface for interactive RNA design (Gradio-based)
- **`RNA_Design_Tutorial.ipynb`**: Jupyter notebook tutorial (if available)
- **`eterna100_vienna2.txt`**: Test structures from Eterna100 benchmark

## Quick Start

### Command-Line Interface (CLI)

The simplest way to design RNA sequences is using the CLI:

```bash
# Basic design for a simple hairpin
ogb rna_design --structure "(((...)))"

# Design with custom parameters
ogb rna_design \
    --structure "(((...)))" \
    --model yangheng/OmniGenome-186M \
    --mutation-ratio 0.3 \
    --num-population 200 \
    --num-generation 150 \
    --output-file results.json
```

**Note**: RNA design is now available through the unified `ogb` command interface.

### Python API

For programmatic access and batch processing:

```python
from omnigenbench import OmniModelForRNADesign, OmniTokenizer

# Initialize tokenizer
tokenizer = OmniTokenizer.from_pretrained(
    "yangheng/OmniGenome-186M",
    trust_remote_code=True
)

# Initialize model
model = OmniModelForRNADesign(
    "yangheng/OmniGenome-186M",
    tokenizer=tokenizer
)

# Design sequences
sequences = model.design(
    structure="(((...)))",
    mutation_ratio=0.5,
    num_population=100,
    num_generation=100
)

print(f"Designed {len(sequences)} sequences:")
for seq in sequences[:5]:
    print(f"  {seq}")
```

## Running Examples

### Run All Examples

```bash
cd examples/rna_sequence_design
python rna_design_examples.py
```

This will run 5 comprehensive examples:
1. Simple hairpin design
2. Custom parameters for complex structures
3. Batch design multiple structures
4. Validate designed sequences
5. Save and load design results

### Run Specific Examples

```python
from rna_design_examples import example_1_simple_hairpin, example_3_batch_design

# Run individual examples
example_1_simple_hairpin()
example_3_batch_design()
```

## Algorithm Details

### Evolutionary Algorithm with MLM

The design process combines:

1. **Initialization**: Generate initial population using MLM-conditioned sampling
2. **Crossover**: Multi-point recombination between parent sequences
3. **Mutation**: MLM-guided mutations with configurable mutation rate
4. **Evaluation**: Structure prediction using ViennaRNA
5. **Selection**: Multi-objective optimization (NSGA-II-like) based on:
   - Structure similarity (Hamming distance)
   - Thermodynamic stability (Minimum Free Energy)

### Key Parameters

- **`structure`**: Target secondary structure in dot-bracket notation
  - `(` = opening base pair
  - `)` = closing base pair
  - `.` = unpaired base
  
- **`mutation_ratio`** (0.0-1.0, default: 0.5): Fraction of nucleotides mutated per generation
  - Lower (0.1-0.3): Conservative, slower but stable
  - Higher (0.5-0.8): Aggressive, faster but may be unstable
  
- **`num_population`** (default: 100): Number of candidate sequences per generation
  - Larger populations explore more solutions but run slower
  - Recommended: 100-500 depending on complexity
  
- **`num_generation`** (default: 100): Maximum evolutionary iterations
  - Algorithm terminates early if perfect solution found
  - Recommended: 50-200 depending on difficulty

## Parameter Tuning Guide

After mastering basic RNA design, optimize performance with these advanced parameter settings:

### Mutation Ratio (`--mutation-ratio`)

Controls how aggressively sequences are modified each generation.

**Effect on Design**:
```bash
# Conservative approach - slow but stable (good for complex structures)
ogb rna_design \
    --structure "((((....))))" \
    --mutation-ratio 0.1 \
    --num-generation 200

# Balanced approach - recommended starting point
ogb rna_design \
    --structure "((((....))))" \
    --mutation-ratio 0.5 \
    --num-generation 100

# Aggressive approach - fast exploration (good for simple structures)
ogb rna_design \
    --structure "(((...)))" \
    --mutation-ratio 0.8 \
    --num-generation 50
```

**When to Use**:
- **0.1-0.3**: Complex nested structures (>50 bp), pseudoknots, multi-stem structures
- **0.4-0.6**: Medium complexity structures (20-50 bp), standard hairpins
- **0.7-0.9**: Simple structures (<20 bp), quick prototyping

**Trade-offs**:
- Lower mutation ratio: More stable convergence, better for local optimization
- Higher mutation ratio: Better global exploration, risk of unstable convergence

### Population Size (`--num-population`)

Number of candidate sequences maintained per generation.

```bash
# Small population - fastest (testing/simple structures)
ogb rna_design \
    --structure "(((...)))" \
    --num-population 50 \
    --num-generation 100

# Medium population - balanced (recommended)
ogb rna_design \
    --structure "((((....))))" \
    --num-population 100 \
    --num-generation 100

# Large population - thorough search (complex structures)
ogb rna_design \
    --structure "((((.((....)).)..)))" \
    --num-population 500 \
    --num-generation 150
```

**Guidelines**:
- **50-100**: Simple hairpins (<30 bp)
- **100-200**: Standard structures (30-60 bp)
- **200-500**: Complex multi-loop structures (>60 bp)
- **500+**: Very difficult structures (use with caution - slow)

**Performance Impact**:
- Each doubling of population roughly doubles computation time per generation
- Larger populations increase diversity and reduce early convergence
- Diminishing returns above 500 for most structures

### Number of Generations (`--num-generation`)

Maximum evolutionary iterations before termination.

```bash
# Quick design (testing, simple structures)
ogb rna_design --structure "(((...)))" --num-generation 50

# Standard design (most use cases)
ogb rna_design --structure "((((....))))" --num-generation 100

# Thorough design (difficult structures)
ogb rna_design --structure "((((.((....)).))))" --num-generation 200

# Extended design (very challenging structures)
ogb rna_design --structure "complex_structure" --num-generation 500
```

**Early Termination**:
The algorithm stops automatically when a perfect solution is found, so setting a high `num_generation` is safe:

```python
# Algorithm will stop early if solution found at generation 25
sequences = model.design(
    structure="(((...)))",
    num_generation=200  # Safety net, may not reach 200
)
```

**When to Increase**:
- No solutions found with default settings
- Structure similarity plateaus below 95%
- Complex multi-stem or pseudoknot structures
- Publication-quality designs requiring optimization

### Combined Parameter Strategies

**Strategy 1: Fast Prototyping**
```bash
ogb rna_design \
    --structure TARGET \
    --mutation-ratio 0.7 \
    --num-population 50 \
    --num-generation 50
# Time: ~30 seconds, Good for: Quick testing
```

**Strategy 2: Balanced Production** (Recommended)
```bash
ogb rna_design \
    --structure TARGET \
    --mutation-ratio 0.5 \
    --num-population 100 \
    --num-generation 100
# Time: ~2-3 minutes, Good for: Most structures
```

**Strategy 3: High-Quality Design**
```bash
ogb rna_design \
    --structure TARGET \
    --mutation-ratio 0.3 \
    --num-population 200 \
    --num-generation 200
# Time: ~10-15 minutes, Good for: Complex structures, publications
```

**Strategy 4: Maximum Exploration**
```bash
ogb rna_design \
    --structure TARGET \
    --mutation-ratio 0.5 \
    --num-population 500 \
    --num-generation 300
# Time: ~30-60 minutes, Good for: Extremely difficult structures
```

### Model Selection (`--model`)

Different foundation models have different characteristics:

```bash
# OmniGenome-186M (recommended - balanced)
ogb rna_design \
    --structure "(((...)))" \
    --model yangheng/OmniGenome-186M

# OmniGenome-52M (faster, less accurate)
ogb rna_design \
    --structure "(((...)))" \
    --model yangheng/OmniGenome-52M

# PlantRNA-FM (specialized for plant RNA)
ogb rna_design \
    --structure "(((...)))" \
    --model yangheng/PlantRNA-FM
```

**Model Comparison**:
| Model | Speed | Quality | Best For |
|-------|-------|---------|----------|
| OmniGenome-52M | Fastest | Good | Testing, simple structures |
| OmniGenome-186M | Medium | Best | Production, complex structures |
| PlantRNA-FM | Medium | Excellent | Plant-specific sequences |

### Python API Advanced Configuration

For full programmatic control:

```python
from omnigenbench import OmniModelForRNADesign, OmniTokenizer

tokenizer = OmniTokenizer.from_pretrained(
    "yangheng/OmniGenome-186M",
    trust_remote_code=True
)

model = OmniModelForRNADesign(
    "yangheng/OmniGenome-186M",
    tokenizer=tokenizer
)

# Design with custom parameters (supported by API)
sequences = model.design(
    structure="((((....))))",
    mutation_ratio=0.5,   # Moderate exploration
    num_population=200,   # Large population for diversity
    num_generation=150    # Allow thorough search
)

# Analyze results
for i, seq in enumerate(sequences[:5]):
    print(f"Solution {i+1}: {seq}")
    # Validate with ViennaRNA
    import RNA
    (predicted_structure, mfe) = RNA.fold(seq)
    print(f"  MFE: {mfe:.2f} kcal/mol")
    print(f"  Structure: {predicted_structure}")
```

Note: The current API supports `structure`, `mutation_ratio`, `num_population`, and `num_generation`. Early stopping returns all perfect matches immediately; otherwise up to 25 best candidates are returned from the final population.

### Performance Benchmarks

Approximate runtime on standard hardware (Intel i7, NVIDIA RTX 3090):

| Structure Complexity | Parameters | Time |
|---------------------|------------|------|
| Simple (<20 bp) | Default | 30s - 1min |
| Medium (20-50 bp) | Default | 1-3 min |
| Complex (50-100 bp) | Default | 5-10 min |
| Very Complex (>100 bp) | High population | 15-60 min |

**Optimization Tips**:
1. Start with default parameters
2. If no solution found, increase `num_generation` first
3. If convergence is slow, adjust `mutation_ratio`
4. For difficult structures, increase `num_population` last (most expensive)
5. Use GPU for faster MLM predictions

## Structure Notation

### Dot-Bracket Notation

Secondary structures are represented using dot-bracket notation:

- `(` and `)` represent base pairs (stem regions)
- `.` represents unpaired bases (loops)
- Balanced parentheses indicate stem structures
- Nested parentheses indicate nested structures

### Common Structure Patterns

```python
# Simple hairpin (3bp stem, 3nt loop)
"(((...)))"

# Stem-loop-stem (nested)
"(((..(((...)))..)))"

# Multi-loop structure
"(((...(((...)))..(((...))).)))"

# Long stem
"((((((((....))))))))"
```

## Output Format

### Console Output

```
Best RNA sequences for (((...))):
- GCGAAACGC
- GCUAAAGCC
- GCCGCCGGC
...
```

### JSON Output (with --output-file)

```json
{
  "structure": "(((...)))",
  "parameters": {
    "model": "yangheng/OmniGenome-186M",
    "mutation_ratio": 0.5,
    "num_population": 100,
    "num_generation": 100
  },
  "best_sequences": [
    "GCGAAACGC",
    "GCUAAAGCC",
    ...
  ]
}
```

## Performance Tips

### 1. Start Simple
Test with small structures first (< 20 nt) to verify functionality.

### 2. Adjust Parameters Based on Complexity
- Simple structures (< 20 nt): Default parameters work well
- Medium structures (20-50 nt): Increase population to 200-300
- Complex structures (> 50 nt): Use population 300-500, generations 150-200

### 3. Monitor Progress
The algorithm shows real-time progress with:
- Progress bar showing generation count
- Best score (Hamming distance to target)
- Early termination when perfect matches are found

### 4. GPU Acceleration
- Automatically uses GPU for MLM inference if available
- Can significantly speed up large population designs
- Set `CUDA_VISIBLE_DEVICES` to control GPU selection

### 5. Parallel Structure Evaluation
Enable parallel folding for faster evaluation (Python API only):

```python
model = OmniModelForRNADesign(model="yangheng/OmniGenome-186M", parallel=True)
```

## Validation

Always validate designed sequences to ensure they fold correctly:

```python
import RNA  # ViennaRNA Python bindings

# Design sequences
sequences = model.design(structure="(((...)))")

# Validate each sequence
for seq in sequences[:5]:
    predicted_structure, mfe = RNA.fold(seq)
    matches = predicted_structure == "(((...)))"
    status = "[MATCH]" if matches else "[MISMATCH]"
    print(f"{seq}: {status} (MFE: {mfe:.2f})")
```

## Troubleshooting

### Issue: Poor convergence (no perfect matches)
**Solution**: 
- Decrease mutation ratio (0.2-0.3)
- Increase population size (200-500)
- Increase number of generations (150-200)
- Try a different random seed

### Issue: Too slow
**Solution**:
- Reduce population size (50-100)
- Reduce number of generations (30-50)
- Use GPU acceleration
- Enable parallel evaluation (`parallel=True`)

### Issue: Out of memory
**Solution**:
- Reduce batch size in MLM inference
- Use CPU instead of GPU
- Reduce population size

### Issue: Invalid structure notation
**Solution**:
- Ensure balanced parentheses
- Only use `(`, `)`, and `.` characters
- Check structure with: `structure.count('(') == structure.count(')')`

## Advanced Usage

### Batch Processing

Process multiple structures efficiently:

```python
structures = ["(((...)))", "(((..)))", "((((....))))"]
model = OmniModelForRNADesign(model="yangheng/OmniGenome-186M")

results = {}
for structure in structures:
    sequences = model.design(structure=structure, num_population=100, num_generation=50)
    results[structure] = sequences
```

### Custom Scoring

For advanced users, modify the fitness function in `model.py` to incorporate additional constraints (GC content, specific motifs, etc.).

### Integration with Workflows

Combine with downstream analysis:

```python
from omnigenbench import OmniModelForRNADesign
import RNA  # ViennaRNA Python bindings

# Design
model = OmniModelForRNADesign(model="yangheng/OmniGenome-186M")
sequences = model.design(structure="(((...)))")

# Analyze thermodynamics
for seq in sequences:
    _, mfe = RNA.fold(seq)
    gc_content = (seq.count('G') + seq.count('C')) / len(seq)
    print(f"Seq: {seq}, MFE: {mfe:.2f}, GC: {gc_content:.2%}")
```

## Citation

If you use this RNA design functionality in your research, please cite:

```bibtex
@software{omnigenbench2024,
  title={OmniGenBench: A Comprehensive Benchmark for Genomic Foundation Models},
  author={Yang, Heng and others},
  year={2024},
  url={https://github.com/yangheng95/OmniGenBench}
}
```

## References

- **ViennaRNA**: Lorenz et al., "ViennaRNA Package 2.0", Algorithms for Molecular Biology, 2011
- **Eterna100**: Anderson-Lee et al., "Principles for predicting RNA secondary structure design difficulty", Journal of Molecular Biology, 2016
- **NSGA-II**: Deb et al., "A fast and elitist multiobjective genetic algorithm: NSGA-II", IEEE Transactions on Evolutionary Computation, 2002

## Support

For issues, questions, or contributions:
- GitHub Issues: https://github.com/yangheng95/OmniGenBench/issues
- Documentation: https://omnigenbench.readthedocs.io
- Email: hy345@exeter.ac.uk

## License

This software is released under the MIT License. See LICENSE file for details.
