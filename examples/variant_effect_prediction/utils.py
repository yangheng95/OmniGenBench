"""
Utilities for the Variant Effect Prediction (VEP) notebook.

This module centralizes reusable logic so the notebook stays concise:
- Reference genome download helper
- Annotation loader and sequence extraction
- Variant application utility
- Model-agnostic embedding extractors
- Main VEP analysis pipeline
"""

from typing import List, Tuple, Optional

import os
import zipfile
import requests
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from scipy import spatial
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel
from Bio import SeqIO

# Optional utilities used by helpers below
import findfile


def download_vep_dataset(local_dir):
    if not findfile.find_cwd_dir(local_dir, disable_alert=True):
        os.makedirs(local_dir, exist_ok=True)

    url_to_download = "https://huggingface.co/datasets/yangheng/variant_effect_prediction/resolve/main/vep_dataset.zip"
    zip_path = os.path.join(local_dir, "vep_dataset.zip")
    if not os.path.exists(zip_path):
        print(f"Downloading vep_dataset.zip from {url_to_download}...")
        response = requests.get(url_to_download, stream=True)
        response.raise_for_status()

        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {zip_path}")

    # Unzip the dataset if the zip file exists
    ZIP_DATASET = findfile.find_cwd_file("vep_dataset.zip")
    if ZIP_DATASET:
        with zipfile.ZipFile(ZIP_DATASET, 'r') as zip_ref:
            zip_ref.extractall(local_dir)
        print(f"Extracted vep_dataset.zip into {local_dir}")
        os.remove(ZIP_DATASET)
    else:
        print("vep_dataset.zip not found. Skipping extraction.")


# -----------------------------
# Reference genome utilities
# -----------------------------
def download_ncbi_reference_genome() -> str:
    """Download and extract the hg38 reference genome if not found locally.

    Returns:
        Path to the FASTA file for hg38.
    """
    import requests
    import gzip
    import shutil

    found_genome = findfile.find_cwd_file(or_key=["hg38.fa", "GRCh38.primary_assembly.genome.fa"], exclude_key=[".gz"])
    if found_genome:
        print(f"Reference genome already exists: {found_genome}")
        return found_genome

    url = "http://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz"
    fasta_path_gz = "hg38.fa.gz"
    fasta_path = "hg38.fa"

    print(f"Downloading reference genome from {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        with open(fasta_path_gz, 'wb') as f, tqdm(total=total_size, unit='B', unit_scale=True, desc=fasta_path_gz) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
    except requests.RequestException as e:
        raise Exception(f"Failed to download reference genome: {e}")

    print(f"Extracting {fasta_path_gz}...")
    with gzip.open(fasta_path_gz, 'rb') as f_in:
        with open(fasta_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    os.remove(fasta_path_gz)
    print(f"Reference genome ready at {fasta_path}")
    return fasta_path


# -----------------------------
# Annotation and sequence utils
# -----------------------------
class Annotation:
    """Handles loading variant annotations and extracting DNA sequences with context."""

    def __init__(self, annotation_path: str, reference_genome_path: str, context_size: int):
        self.context_size = context_size
        self.annotation = pd.read_csv(annotation_path, sep='\t')
        self.annotation['orig_start'] = self.annotation['start']
        self.annotation['orig_end'] = self.annotation['end']
        self.annotation['variant_offset'] = self.annotation['start'] - self.annotation['orig_start']

        print(f"Loading reference genome from {reference_genome_path}...")
        self.genome_dict = SeqIO.to_dict(SeqIO.parse(reference_genome_path, "fasta"))
        print(f"Loaded {len(self.genome_dict)} chromosomes.")

        self.extend_segments()

    def extend_segments(self):
        """Calculates new start/end coordinates to include the context window."""
        df = self.annotation
        df['start'] = (df['orig_start'] - self.context_size).clip(lower=0)
        df['end'] = df['orig_end'] + self.context_size
        df['mutation_position'] = (df['variant_offset'] + self.context_size).astype(int)
        self.annotation = df

    def get_dna_segment(self, index: int) -> str:
        """Extracts a DNA segment for a given variant index."""
        item = self.annotation.iloc[index]
        chrom = item.get('chromosome', item.get('chr'))
        start, end = int(item['start']), int(item['end'])

        if chrom not in self.genome_dict:
            chrom_alt = f"chr{chrom}" if not chrom.startswith('chr') else chrom.replace('chr', '')
            if chrom_alt in self.genome_dict:
                chrom = chrom_alt
            else:
                return ""

        seq_obj = self.genome_dict[chrom]
        end = min(end, len(seq_obj.seq))
        return str(seq_obj.seq[start:end]).upper()


def apply_variant(sequence: str, ref_allele: str, alt_allele: str, mut_pos: int) -> str:
    """Applies a single nucleotide variant (SNV) to a reference sequence."""
    if not (len(ref_allele) == 1 and len(alt_allele) == 1):
        return sequence
    if mut_pos < 0 or mut_pos >= len(sequence):
        return sequence
    if sequence[mut_pos].upper() != ref_allele.upper():
        pass
    return sequence[:mut_pos] + alt_allele.upper() + sequence[mut_pos + 1:]


# -----------------------------
# Model tokenization/embedding
# -----------------------------
def _tokenize(batch_seqs: List[str], tokenizer, context_len: int, device: torch.device, add_spaces: bool = False):
    return tokenizer(
        batch_seqs,
        return_tensors="pt",
        padding="max_length",
        max_length=context_len,
        truncation=True,
        add_special_tokens=False
    ).to(device)


def _batch_extract_embeddings(batch_seqs, batch_pos, model, tokenizer, context_len, add_spaces=False):
    tokens = _tokenize(batch_seqs, tokenizer, context_len, model.device, add_spaces=add_spaces)
    with torch.no_grad():
        outputs = model(input_ids=tokens['input_ids'], output_hidden_states=True)
        hiddens = outputs.hidden_states[-1]

    cls_embs, mut_embs = [], []
    for j, pos in enumerate(batch_pos):
        cls_embs.append(hiddens[j, 0, :].cpu())
        mut_embs.append(hiddens[j, pos, :].cpu())
    return cls_embs, mut_embs


def compute_batch_omnigenome_outputs(sequences, mut_positions, context_length, model, tokenizer, batch_size):
    cls_embs, mut_embs = [], []
    for i in tqdm(range(0, len(sequences), batch_size), desc="OmniGenome"):
        batch_seqs = sequences[i:i + batch_size]
        batch_pos = mut_positions[i:i + batch_size]
        c, m = _batch_extract_embeddings(batch_seqs, batch_pos, model, tokenizer, context_length, add_spaces=True)
        cls_embs.extend(c)
        mut_embs.extend(m)
    return cls_embs, mut_embs


def compute_batch_dnabert_outputs(sequences, mut_positions, context_length, model, tokenizer, batch_size):
    cls_embs, mut_embs = [], []
    for i in tqdm(range(0, len(sequences), batch_size), desc="DNABERT"):
        batch_seqs = sequences[i:i + batch_size]
        batch_pos = mut_positions[i:i + batch_size]
        c, m = _batch_extract_embeddings(batch_seqs, batch_pos, model, tokenizer, context_length, add_spaces=False)
        cls_embs.extend(c)
        mut_embs.extend(m)
    return cls_embs, mut_embs


def compute_batch_hyena_outputs(sequences, mut_positions, context_length, model, tokenizer, batch_size):
    cls_embs, mut_embs = [], []
    for i in tqdm(range(0, len(sequences), batch_size), desc="HyenaDNA"):
        batch_seqs = sequences[i:i + batch_size]
        batch_pos = mut_positions[i:i + batch_size]
        c, m = _batch_extract_embeddings(batch_seqs, batch_pos, model, tokenizer, context_length, add_spaces=False)
        cls_embs.extend(c)
        mut_embs.extend(m)
    return cls_embs, mut_embs


def compute_batch_nucleotide_transformer_outputs(sequences, mut_positions, context_length, model, tokenizer, batch_size):
    k = 6  # 6-mer tokenizer
    cls_embs, mut_embs = [], []
    for i in tqdm(range(0, len(sequences), batch_size), desc="Nucleotide Transformer"):
        batch_seqs = sequences[i:i + batch_size]
        batch_pos = mut_positions[i:i + batch_size]
        tok_pos = [p // k for p in batch_pos]
        c, m = _batch_extract_embeddings(batch_seqs, tok_pos, model, tokenizer, context_length // k, add_spaces=False)
        cls_embs.extend(c)
        mut_embs.extend(m)
    return cls_embs, mut_embs


def compute_batch_splice_outputs(sequences, mut_positions, context_length, model, tokenizer, batch_size):
    return compute_batch_dnabert_outputs(sequences, mut_positions, context_length, model, tokenizer, batch_size)


def compute_batch_multimolecule_outputs(sequences, mut_positions, context_length, model, tokenizer, batch_size):
    return compute_batch_omnigenome_outputs(sequences, mut_positions, context_length, model, tokenizer, batch_size)


# -----------------------------
# Main VEP analysis
# -----------------------------
def run_vep_analysis(
    model_name: str,
    bed_file: str,
    fasta_file: Optional[str],
    context_size: int,
    batch_size: int,
    device: torch.device,
    max_variants: Optional[int] = None,
) -> pd.DataFrame:
    """Main pipeline for running the Variant Effect Prediction analysis.

    Args:
        model_name: Hugging Face model id or local path
        bed_file: Path to BED/TSV file
        fasta_file: Path to reference genome (if None, will attempt to download hg38)
        context_size: Context size in base pairs on each side
        batch_size: Inference batch size
        device: torch device
        max_variants: Optional subsample size

    Returns:
        DataFrame with distances and optional AUC.
    """

    # 1. Setup Device and Check Files
    print("--- Step 1: Initializing ---")
    if not fasta_file:
        print("Reference genome not found. Attempting to download...")
        fasta_file = download_ncbi_reference_genome()
    if not os.path.exists(bed_file):
        raise FileNotFoundError(f"BED file not found at: {bed_file}")

    # 2. Load Genomic Annotation
    print("--- Step 2: Loading Annotations ---")
    genome_annotation = Annotation(bed_file, fasta_file, context_size)
    df = genome_annotation.annotation
    if max_variants is not None:
        df = df.sample(n=max_variants, random_state=42).reset_index(drop=True)
        print(f"Processing {len(df)} variants (truncated to max_variants={max_variants}).")
    else:
        print(f"Processing {len(df)} variants.")

    # 3. Load Model and Tokenizer
    print(f"--- Step 3: Loading Model: {model_name} ---")
    if "multimolecule" in model_name.lower():
        from multimolecule import AutoModelForTokenPrediction, RnaTokenizer
        model_loader = AutoModelForTokenPrediction
        tokenizer = RnaTokenizer.from_pretrained(model_name, trust_remote_code=True)
    elif 'dnabert' in model_name.lower() or 'nucleotide-transformer' in model_name.lower():
        model_loader = AutoModelForMaskedLM
        from omnigenbench import OmniTokenizer
        tokenizer = OmniTokenizer.from_pretrained(model_name, trust_remote_code=True)
    else:
        model_loader = AutoModel
        from omnigenbench import OmniTokenizer
        tokenizer = OmniTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model = model_loader.from_pretrained(model_name, trust_remote_code=True)
    if hasattr(model, 'base_model'):
        model = model.base_model
    model.to(device).eval().half()
    print(f"Model loaded on {device} with {sum(p.numel() for p in model.parameters()):} parameters.")

    # 4. Select the correct embedding function
    model_name_lower = model_name.lower()
    if 'omnigenome' in model_name_lower:
        compute_func = compute_batch_omnigenome_outputs
    elif 'dnabert' in model_name_lower:
        compute_func = compute_batch_dnabert_outputs
    elif 'hyenadna' in model_name_lower:
        compute_func = compute_batch_hyena_outputs
    elif 'nucleotide-transformer' in model_name_lower:
        compute_func = compute_batch_nucleotide_transformer_outputs
    elif 'splice' in model_name_lower:
        compute_func = compute_batch_splice_outputs
    elif 'multimolecule' in model_name_lower:
        compute_func = compute_batch_multimolecule_outputs
    else:
        raise ValueError(f"No compute function found for model: {model_name}")

    # 5. Generate Reference and Alternative Sequences
    print("--- Step 4: Generating Sequences ---")
    seq_ref_list, seq_alt_list, mut_pos_list, valid_indices = [], [], [], []
    for idx, item in tqdm(df.iterrows(), total=len(df), desc="Generating Sequences"):
        seq_ref = genome_annotation.get_dna_segment(idx)
        if not seq_ref:
            continue
        mut_pos = int(item['mutation_position'])
        seq_alt = apply_variant(seq_ref, item['ref'], item['alt'], mut_pos)

        seq_ref_list.append(seq_ref)
        seq_alt_list.append(seq_alt)
        mut_pos_list.append(mut_pos)
        valid_indices.append(idx)

    # 6. Compute Embeddings
    print("--- Step 5: Computing Embeddings ---")
    context_length = context_size * 2
    r_cls, r_mut = compute_func(seq_ref_list, mut_pos_list, context_length, model, tokenizer, batch_size)
    a_cls, a_mut = compute_func(seq_alt_list, mut_pos_list, context_length, model, tokenizer, batch_size)

    # 7. Calculate Distances and Finalize Results
    print("--- Step 6: Calculating Scores ---")
    results = []
    def norm(x):
        return x / (torch.linalg.norm(x) + 1e-8)

    for i, original_idx in enumerate(valid_indices):
        cls_dist = spatial.distance.cosine(norm(a_cls[i]), norm(r_cls[i]))
        mut_dist = spatial.distance.cosine(norm(a_mut[i]), norm(r_mut[i]))
        row = df.loc[original_idx].to_dict()
        row.update({'cls_dist': cls_dist, 'mut_dist': mut_dist})
        results.append(row)

    results_df = pd.DataFrame(results)
    if 'label' in results_df.columns:
        overall_auc = roc_auc_score(results_df['label'], results_df['mut_dist'])
        print(f"Overall AUC based on 'mut_dist': {overall_auc:.4f}")
        results_df['overall_auc'] = overall_auc

    return results_df


__all__ = [
    'download_ncbi_reference_genome',
    'Annotation',
    'apply_variant',
    'run_vep_analysis',
    "compute_batch_omnigenome_outputs",
    "compute_batch_dnabert_outputs",
    "compute_batch_hyena_outputs",
    "compute_batch_nucleotide_transformer_outputs",
    "compute_batch_splice_outputs",
    "compute_batch_multimolecule_outputs",
    'download_vep_dataset',
]


