# Data Template and Supported Formats for OmniGenBench

OmniGenBench supports flexible and robust data loading for genomic machine learning tasks. To ensure seamless integration, your data should follow a simple template and be saved in one of the supported formats.

## Data Template: `{sequence, label}` Structure

Each data sample should be represented as a dictionary with at least two keys:
- `sequence`: The biological sequence (DNA, RNA, or protein) as a string.
- `label`: The target value for the task (classification, regression, etc.).

### Example for Classification

For a binary classification task, your data in a JSON file should look like this:

```json
[
  {"sequence": "ATCGATCGATCG", "label": "0"},
  {"sequence": "GCTAGCTAGCTA", "label": "1"}
]
```

### Example for Regression

For a regression task, the `label` should be a numerical value:

```json
[
  {"sequence": "ATCGATCGATCG", "label": 0.75},
  {"sequence": "GCTAGCTAGCTA", "label": -1.2}
]
```

### Flexible Key Names

OmniGenBench will automatically standardize common key names. For example, `seq` or `text` will be treated as `sequence`, and `label` will be standardized to `labels` internally.

## Supported Data Formats

OmniGenBench can load data from the following formats:

1. **JSON (`.json`)**: Recommended. A list of dictionaries as shown above. Also supports JSON Lines (`.jsonl`).
2. **CSV (`.csv`)**: Must have columns for `sequence` and `label`.

    ```csv
    sequence,label
    ATCGATCGATCG,0
    GCTAGCTAGCTA,1
    ```

3. **Parquet (`.parquet`)**: Columns for `sequence` and `label`.
4. **FASTA (`.fasta`, `.fa`, etc.)**: Sequence data only. Labels must be provided separately or inferred.
5. **FASTQ (`.fastq`, `.fq`)**: Sequence and quality scores. Labels must be provided separately or inferred.
6. **BED (`.bed`)**: Genomic intervals. Sequence and label columns may need to be added.
7. **Numpy (`.npy`, `.npz`)**: Array of dictionaries with `sequence` and optional `label`.

## How to Organize Your Data

- For supervised tasks, ensure every sample has both a `sequence` and a `label`.
- For unsupervised or sequence-only tasks, only the `sequence` key is required.
- Save your data in one of the supported formats listed above.

## Configuration Example

Specify your data files in the task configuration (e.g., `config.py`):

```python
config_dict = {
    "train_file": "path/to/train.json",
    "valid_file": "path/to/valid.json",
    "test_file": "path/to/test.json",
    # ...other config...
}
```

## Summary

**OmniGenBench supports 7 data formats:**
- JSON, JSONL, CSV, Parquet, FASTA, FASTQ, BED, Numpy

Follow the `{sequence, label}` template for best compatibility. For more details, see the documentation or example datasets.
