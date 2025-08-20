import os
import zipfile

import findfile
import requests
import torch
from omnigenbench import (
    ClassificationMetric,
    OmniTokenizer,
    OmniModelForSequenceClassification,
    OmniDatasetForSequenceClassification,
    Trainer,
)




def download_te_dataset(local_dir):
    if not findfile.find_cwd_dir(local_dir, disable_alert=True):
        os.makedirs(local_dir, exist_ok=True)
    url_to_download = "https://huggingface.co/datasets/yangheng/translation_efficiency_prediction/resolve/main/translation_efficiency_prediction.zip"
    zip_path = os.path.join(local_dir, "te_rice_dataset.zip")
    if not os.path.exists(zip_path):
        print(f"Downloading te_rice_dataset.zip from {url_to_download}...")
        response = requests.get(url_to_download, stream=True)
        response.raise_for_status()

        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {zip_path}")

    # Unzip the dataset if the zip file exists
    ZIP_DATASET = findfile.find_cwd_file("te_rice_dataset.zip")
    if ZIP_DATASET:
        with zipfile.ZipFile(ZIP_DATASET, 'r') as zip_ref:
            zip_ref.extractall(local_dir)
        print(f"Extracted te_rice_dataset.zip into {local_dir}")
        os.remove(ZIP_DATASET)
    else:
        print("te_rice_dataset.zip not found. Skipping extraction.")


class TEClassificationDataset(OmniDatasetForSequenceClassification):
    def __init__(self, data_source, tokenizer, max_length, **kwargs):
        super().__init__(data_source, tokenizer, max_length, **kwargs)

    def prepare_input(self, instance, **kwargs):
        sequence, labels = instance["sequence"], instance["label"]

        tokenized_inputs = self.tokenizer(
            sequence,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            **kwargs
        )
        tokenized_inputs["labels"] = torch.tensor(int(labels), dtype=torch.long)
        # Remove the batch dimension that gets added by return_tensors="pt"
        for col in tokenized_inputs:
            tokenized_inputs[col] = tokenized_inputs[col].squeeze(0)

        if labels is not None:
            label_id = self.label2id.get(str(labels), -100)
            tokenized_inputs["labels"] = torch.tensor(label_id, dtype=torch.long)

        return tokenized_inputs

def run_finetuning(
    model_name,
    train_file,
    valid_file,
    test_file,
    label2id,
    epochs,
    learning_rate,
    weight_decay,
    batch_size,
    max_length,
    seed,
):
    """
    Runs the full TE classification analysis pipeline.
    """
    # 1. Model & Tokenizer Initialization
    tokenizer = OmniTokenizer.from_pretrained(model_name, trust_remote_code=True)
    ssp_model = OmniModelForSequenceClassification(
        model_name,
        tokenizer=tokenizer,
        label2id=label2id,
        trust_remote_code=True,
    )
    print(f"Model '{model_name}' and tokenizer loaded successfully.")

    # 2. Data Loading & Preparation
    train_set = TEClassificationDataset(data_source=train_file, tokenizer=tokenizer, label2id=label2id, max_length=max_length)
    valid_set = TEClassificationDataset(data_source=valid_file, tokenizer=tokenizer, label2id=label2id, max_length=max_length)
    test_set = TEClassificationDataset(data_source=test_file, tokenizer=tokenizer, label2id=label2id, max_length=max_length)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)
    print("Datasets and DataLoaders created.")

    # 3. Training & Evaluation Setup
    compute_metrics = [ClassificationMetric(ignore_y=-100, average="macro").f1_score]
    optimizer = torch.optim.AdamW(ssp_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    trainer = Trainer(
        model=ssp_model,
        train_loader=train_loader,
        eval_loader=valid_loader,
        test_loader=test_loader,
        batch_size=batch_size,
        epochs=epochs,
        optimizer=optimizer,
        compute_metrics=compute_metrics,
        seeds=seed,
    )

    # 4. Run Training
    metrics = trainer.train()
    trainer.save_model("finetuned_te_model")
    print("Training completed!")

    return metrics


