# -*- coding: utf-8 -*-
# file: baselines_tfb.py
# Train and evaluate CNN/RNN baselines on DeepSEA TFB using OmniGenBench trainers

import os
import argparse

import torch
import autocuda
from transformers import AutoTokenizer

from omnigenbench import AccelerateTrainer, ClassificationMetric
import sys
import pathlib

# Make local utils importable when running as a script
_here = pathlib.Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from utils import (
    download_deepsea_dataset,
    build_datasets,
    create_dataloaders,
    OmniCNNBaseline,
    OmniRNNBaseline,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="deepsea_data")
    parser.add_argument("--max_length", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--model", type=str, choices=["cnn", "rnn"], default="cnn")
    parser.add_argument("--save_dir", type=str, default="tfb_baseline")
    parser.add_argument("--limit_train", type=int, default=1_000_000)
    parser.add_argument("--limit_eval", type=int, default=10_000)
    parser.add_argument("--tokenizer", type=str, default="yangheng/OmniGenome-52M")
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    download_deepsea_dataset(args.data_dir)

    train_file = os.path.join(args.data_dir, "train.jsonl")
    test_file = os.path.join(args.data_dir, "test.jsonl")
    valid_file = os.path.join(args.data_dir, "valid.jsonl")

    device = autocuda.auto_cuda()

    # Tokenizer provides vocab and padding behavior used by baselines
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # Infer number of labels from a small sample of dataset metadata
    # Here we rely on utils.build_datasets to construct OmniDataset which reads labels on the fly
    # We pass label_indices=None to use full label set in files
    train_set, valid_set, test_set = build_datasets(
        tokenizer=tokenizer,
        train_file=train_file,
        test_file=test_file,
        valid_file=valid_file if os.path.exists(valid_file) else None,
        max_length=args.max_length,
        max_examples=None,  # full set; DataLoader will control batching
        label_indices=None,
    )

    # Peek first item to get label dimension
    sample = train_set[0]
    num_labels = int(sample["labels"].numel()) if sample.get("labels") is not None else 919

    if args.model == "cnn":
        model = OmniCNNBaseline(
            tokenizer=tokenizer,
            num_labels=num_labels,
            dropout=0.1,
            embed_dim=128,
            num_filters=128,
            kernel_sizes=(3, 5, 7),
        )
    else:
        model = OmniRNNBaseline(
            tokenizer=tokenizer,
            num_labels=num_labels,
            dropout=0.1,
            embed_dim=128,
            hidden_dim=256,
            num_layers=1,
            bidirectional=True,
        )

    model = model.to(device).to(torch.float32)

    # Rebuild datasets with caps to avoid OOM in demos
    train_set.max_examples = args.limit_train
    if valid_set is not None:
        valid_set.max_examples = args.limit_eval
    test_set.max_examples = args.limit_eval

    train_loader, valid_loader, test_loader = create_dataloaders(
        train_set, valid_set, test_set, batch_size=args.batch_size
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    metric_fns = [ClassificationMetric(ignore_y=-100, average="macro").roc_auc_score]

    trainer = AccelerateTrainer(
        model=model,
        train_loader=train_loader,
        eval_loader=valid_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        epochs=args.epochs,
        compute_metrics=metric_fns,
        patience=args.patience,
        device=device,
    )

    history = trainer.train(path_to_save=args.save_dir, overwrite=True)
    print(history)
    trainer.save_model(args.save_dir, overwrite=True)

    # Quick test metrics
    test_metrics = trainer.test()
    print("Test metrics:", test_metrics)


if __name__ == "__main__":
    main()


