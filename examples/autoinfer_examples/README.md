# AutoInfer CLI Examples

This folder contains example input files and usage patterns for the `ogb autoinfer` command-line tool.

## Quick Start

### 1. Single Sequence Inference

```bash
ogb autoinfer \
  --model yangheng/ogb_tfb_finetuned \
  --sequence "ATCGATCGATCGATCGATCGATCGATCGATCG" \
  --output-file tfb_predictions.json
```

### 2. Multiple Sequences (Comma-Separated)

```bash
ogb autoinfer \
  --model yangheng/ogb_te_finetuned \
  --sequence "ATCGATCGATCG,GCGCGCGCGCGC,TATATATATATAT" \
  --output-file predictions.json
```

### 3. Batch Inference from JSON File

```bash
ogb autoinfer \
  --model yangheng/ogb_tfb_finetuned \
  --input-file sequences.json \
  --batch-size 64 \
  --output-file results.json
```

### 4. Inference from CSV File

```bash
ogb autoinfer \
  --model yangheng/ogb_tfb_finetuned \
  --input-file data.csv \
  --output-file predictions.json \
  --device cuda:0
```

### 5. Inference from Text File (One Sequence Per Line)

```bash
ogb autoinfer \
  --model yangheng/ogb_te_finetuned \
  --sequence sequences.txt \
  --output-file predictions.json
```

### 6. With Custom Batch Size and Device

```bash
ogb autoinfer \
  --model yangheng/ogb_tfb_finetuned \
  --input-file sequences.json \
  --batch-size 128 \
  --device cuda:0 \
  --output-file results.json
```

## Python API Examples

For programmatic access and integration into pipelines:

```python
from omnigenbench import ModelHub, OmniTokenizer

# Load model and tokenizer (optionally choose device)
tokenizer = OmniTokenizer.from_pretrained(
  "yangheng/ogb_tfb_finetuned",
  trust_remote_code=True
)

model = ModelHub.load("yangheng/ogb_tfb_finetuned", device="cuda:0")

# Single sequence inference
sequence = "ATCGATCGATCGATCGATCGATCGATCGATCG"
result = model.inference(sequence)

print(f"Prediction: {result['predictions']}")
print(f"Confidence: {float(result['confidence']) if 'confidence' in result else 'N/A'}")

# Batch inference (loop over sequences)
sequences = [
  "ATCGATCGATCG",
  "GCGCGCGCGCGC",
  "TATATATATATAT"
]

batch_results = [model.inference(seq) for seq in sequences]
for i, res in enumerate(batch_results, 1):
  print(f"Sequence {i}: Prediction={res['predictions']}, Confidence={float(res['confidence']) if 'confidence' in res else 'N/A'}")
```
```

## Input File Formats

### JSON Format 1: Simple Sequence List

**sequences.json**:
```json
{
  "sequences": [
    "ATCGATCGATCGATCGATCGATCGATCGATCG",
    "GCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGC",
    "TATATATATATATATATATATATATATATATAT"
  ]
}
```

### JSON Format 2: With Metadata

**sequences_with_metadata.json**:
```json
{
  "data": [
    {
      "sequence": "ATCGATCGATCGATCGATCGATCGATCGATCG",
      "gene_id": "gene_001",
      "description": "Promoter region",
      "label": "high"
    },
    {
      "sequence": "GCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGC",
      "gene_id": "gene_002",
      "description": "5' UTR",
      "label": "low"
    }
  ]
}
```

### CSV Format

**data.csv**:
```csv
sequence,gene_id,description,label
ATCGATCGATCGATCGATCGATCGATCGATCG,gene_001,Promoter region,high
GCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGC,gene_002,5' UTR,low
TATATATATATATATATATATATATATATATAT,gene_003,Random sequence,low
```

### Text File Format

**sequences.txt**:
```
ATCGATCGATCGATCGATCGATCGATCGATCG
GCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGC
TATATATATATATATATATATATATATATATAT
```

## Output Format

The output is always saved as a JSON file with the following structure:

```json
{
  "model": "yangheng/ogb_tfb_finetuned",
  "total_sequences": 3,
  "results": [
    {
      "sequence": "ATCGATCGATCGATCGATCGATCGATCGATCG",
      "metadata": {
        "index": 0,
        "gene_id": "gene_001",
        "description": "Promoter region"
      },
      "predictions": [1, 0, 1, 0, 1, ...],
      "probabilities": [0.92, 0.15, 0.88, 0.23, ...]
    },
    {
      "sequence": "GCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGC",
      "metadata": {
        "index": 1,
        "gene_id": "gene_002",
        "description": "5' UTR"
      },
      "predictions": [0, 0, 0, 1, 0, ...],
      "probabilities": [0.12, 0.08, 0.15, 0.89, ...]
    }
  ]
}
```

## Command-Line Arguments Reference

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--model` | Yes | - | Path or name of the fine-tuned model |
| `--sequence` | No* | - | Input sequence(s) or path to text file |
| `--input-file` | No* | - | Path to JSON/CSV file with input data |
| `--output-file` | No | `inference_results.json` | Output file path |
| `--batch-size` | No | 32 | Batch size for inference |
| `--device` | No | Auto-detect | Device to run on (e.g., `cuda:0`, `cpu`) |

*Note: Either `--sequence` or `--input-file` must be provided.

## Advanced Usage Examples

### Large-Scale Batch Processing with GPU

```bash
ogb autoinfer \
  --model yangheng/ogb_tfb_finetuned \
  --input-file large_dataset.csv \
  --batch-size 128 \
  --device cuda:0 \
  --output-file large_results.json
```

### Processing Multiple Files in Pipeline

```bash
#!/bin/bash
# Process multiple input files
for file in data/*.csv; do
  output="${file%.csv}_predictions.json"
  ogb autoinfer \
    --model yangheng/ogb_tfb_finetuned \
    --input-file "$file" \
    --output-file "$output"
done
```

### CPU-Only Inference

```bash
ogb autoinfer \
  --model yangheng/ogb_te_finetuned \
  --input-file sequences.json \
  --device cpu \
  --batch-size 16 \
  --output-file predictions.json
```

## Model Support

The `autoinfer` command supports any fine-tuned OmniGenBench model, including:

- **TFB Models**: Transcription factor binding prediction
  - `yangheng/ogb_tfb_finetuned`
  
- **Translation Efficiency Models**: mRNA translation efficiency prediction
  - `yangheng/ogb_te_finetuned`
  
- **RNA Structure Models**: Secondary structure prediction
  - Custom fine-tuned models from your training

- **Custom Models**: Any model saved with OmniGenBench's `ModelHub`

## Error Handling

If a sequence fails to process, the error will be logged in the output:

```json
{
  "sequence": "INVALID_SEQUENCE",
  "metadata": {"index": 5},
  "error": "Invalid nucleotide character: X"
}
```

The inference will continue processing remaining sequences.

## Advanced Configuration

### Batch Size Tuning

The `--batch-size` parameter controls how many sequences are processed simultaneously. Larger batch sizes can speed up inference but require more memory.

**Recommended Settings**:
```bash
# Small GPU (4-8GB VRAM) or CPU
ogb autoinfer --model MODEL --input-file data.json --batch-size 16

# Medium GPU (8-16GB VRAM)
ogb autoinfer --model MODEL --input-file data.json --batch-size 32

# Large GPU (16GB+ VRAM)
ogb autoinfer --model MODEL --input-file data.json --batch-size 64

# Very large datasets (24GB+ VRAM)
ogb autoinfer --model MODEL --input-file data.json --batch-size 128
```

**Performance Tips**:
- Start with a smaller batch size and increase gradually
- Monitor GPU memory usage with `nvidia-smi` (Linux) or Task Manager (Windows)
- If you encounter out-of-memory errors, reduce batch size by half
- For CPU inference, smaller batches (8-16) typically work better

### Device Selection

The `--device` parameter specifies which hardware to use for inference.

**Options**:
```bash
# Automatic device selection (default - uses CUDA if available)
ogb autoinfer --model MODEL --sequence "ATCG"

# Force CPU usage
ogb autoinfer --model MODEL --sequence "ATCG" --device cpu

# Specific GPU by index
ogb autoinfer --model MODEL --sequence "ATCG" --device cuda:0
ogb autoinfer --model MODEL --sequence "ATCG" --device cuda:1

# Multi-GPU systems: select fastest GPU
ogb autoinfer --model MODEL --input-file data.json --device cuda:0 --batch-size 64
```

**When to Use Which Device**:
- **`cuda:0` (default)**: Best for speed if GPU is available
- **`cpu`**: When GPU is unavailable, or for small-scale inference (<100 sequences)
- **`cuda:1+`**: On multi-GPU systems, distribute different jobs to different GPUs

### Output File Formats

By default, results are saved in JSON format. You can customize the output:

```bash
# Default JSON output
ogb autoinfer --model MODEL --input-file data.json --output-file results.json

# Custom output path
ogb autoinfer --model MODEL --input-file data.json --output-file /path/to/results.json

# Organized by experiment
ogb autoinfer --model MODEL --input-file data.json --output-file experiments/exp1/predictions.json
```

**Output Structure**:
```json
[
  {
    "sequence": "ATCGATCGATCG",
    "prediction": 1,
    "probabilities": [0.23, 0.77],
    "metadata": {"sample_id": "sample_001"}
  }
]
```

### Python API Advanced Configuration

For programmatic control over inference:

```python
from omnigenbench import ModelHub
import torch

# Load model with specific device
model = ModelHub.load("yangheng/ogb_tfb_finetuned", device="cuda:0")

# Configure inference parameters
sequences = ["ATCGATCG" * 10, "GCGCGCGC" * 10]
results = model.batch_inference(
    sequences,
    batch_size=32,           # Control memory usage
    device="cuda:0",         # Explicit device selection
    return_embeddings=False, # Set True to get hidden representations
    max_length=512          # Truncate long sequences
)

# Access detailed outputs
for i, result in enumerate(results['predictions']):
    print(f"Sequence {i}: Prediction={result}")
    print(f"  Confidence: {max(results['probabilities'][i]):.4f}")
```

**Additional Python API Options**:
- `return_embeddings=True`: Get model embeddings alongside predictions
- `max_length`: Control sequence truncation (default: model's max length)
- `device`: Override model device for specific inference
- `batch_size`: Process sequences in batches to manage memory

## Tips

1. **Start Simple**: Begin with default parameters, then optimize if needed

2. **Input Format**: Use CSV format when you have metadata to track alongside sequences

3. **Output Analysis**: The JSON output can be easily loaded into Python for further analysis:
   ```python
   import json
   with open('predictions.json', 'r') as f:
       results = json.load(f)
   ```

4. **Model Selection**: Use models fine-tuned for your specific task for best results

5. **Memory Management**: If you experience crashes, reduce `--batch-size` first

## Getting Help

For more information about available options:

```bash
ogb autoinfer --help
```

**Note**: The legacy `autoinfer` command is still supported for backward compatibility, but `ogb autoinfer` is the recommended interface.

For issues or questions, visit: https://github.com/yangheng95/OmniGenBench/issues
