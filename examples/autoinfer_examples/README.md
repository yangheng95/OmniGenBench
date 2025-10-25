# AutoInfer CLI Examples

This folder contains example input files and usage patterns for the `autoinfer` command-line tool.

## Quick Start

### 1. Single Sequence Inference

```bash
autoinfer \
  --model yangheng/ogb_tfb_finetuned \
  --sequence "ATCGATCGATCGATCGATCGATCGATCGATCG" \
  --output-file tfb_predictions.json
```

### 2. Multiple Sequences (Comma-Separated)

```bash
autoinfer \
  --model yangheng/ogb_te_finetuned \
  --sequence "ATCGATCGATCG,GCGCGCGCGCGC,TATATATATATAT" \
  --output-file predictions.json
```

### 3. Batch Inference from JSON File

```bash
autoinfer \
  --model yangheng/ogb_tfb_finetuned \
  --input-file sequences.json \
  --batch-size 64 \
  --output-file results.json
```

### 4. Inference from CSV File

```bash
autoinfer \
  --model yangheng/ogb_tfb_finetuned \
  --input-file data.csv \
  --output-file predictions.json \
  --device cuda:0
```

### 5. Inference from Text File (One Sequence Per Line)

```bash
autoinfer \
  --model yangheng/ogb_te_finetuned \
  --sequence sequences.txt \
  --output-file predictions.json
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
autoinfer \
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
  autoinfer \
    --model yangheng/ogb_tfb_finetuned \
    --input-file "$file" \
    --output-file "$output"
done
```

### CPU-Only Inference

```bash
autoinfer \
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

## Tips

1. **Batch Size**: Adjust `--batch-size` based on your GPU memory:
   - Small GPU (4-8GB): `--batch-size 16`
   - Medium GPU (8-16GB): `--batch-size 32`
   - Large GPU (16GB+): `--batch-size 64` or higher

2. **Input Format**: Use CSV format when you have metadata to track alongside sequences.

3. **Output Analysis**: The JSON output can be easily loaded into Python for further analysis:
   ```python
   import json
   with open('predictions.json', 'r') as f:
       results = json.load(f)
   ```

4. **Model Selection**: Use models fine-tuned for your specific task for best results.

## Getting Help

For more information about available options:

```bash
autoinfer --help
```

For issues or questions, visit: https://github.com/yangheng95/OmniGenBench/issues
