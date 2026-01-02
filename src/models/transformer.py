from __future__ import annotations

import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed as hf_set_seed,
)

from ..evaluation import ModelResult, bootstrap_ci, compute_classification_metrics
from ..utils import ensure_dir


class TextClassificationDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def train_transformer_model(
    train_texts: List[str],
    train_labels: List[int],
    val_texts: List[str],
    val_labels: List[int],
    test_texts: List[str],
    test_labels: List[int],
    config: dict,
    output_dir: str | Path,
    figures_dir: str | Path,
    seed: int,
    logger=None,
    predictions_dir: str | Path | None = None,
) -> List[ModelResult]:
    output_dir = ensure_dir(output_dir)
    ensure_dir(figures_dir)
    pred_dir = ensure_dir(predictions_dir) if predictions_dir else output_dir
    model_name = config.get("model_name", "bert-base-uncased")
    max_length = int(config.get("max_length", 128))
    batch_size = int(config.get("batch_size", 16))
    learning_rate = float(config.get("learning_rate", 2e-5))
    num_train_epochs = int(config.get("num_train_epochs", 5))
    warmup_ratio = float(config.get("warmup_ratio", 0.1))
    patience = int(config.get("patience", 2))

    hf_set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=max_length)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=max_length)

    train_dataset = TextClassificationDataset(train_encodings, train_labels)
    val_dataset = TextClassificationDataset(val_encodings, val_labels)
    test_dataset = TextClassificationDataset(test_encodings, test_labels)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    def compute_metrics_fn(eval_pred):
        logits = eval_pred.predictions
        labels = eval_pred.label_ids
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": (preds == labels).mean(),
            "precision": precision_score(labels, preds, average="binary", zero_division=0),
            "recall": recall_score(labels, preds, average="binary", zero_division=0),
            "f1": f1_score(labels, preds, average="macro"),
        }

    training_args = TrainingArguments(
        output_dir=str(output_dir / "bert_runs"),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        warmup_ratio=warmup_ratio,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=1,
        report_to="none",
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
    )

    if logger:
        logger.info("Starting BERT fine-tuning...")
    train_start = time.time()
    trainer.train()
    train_time = time.time() - train_start

    # Evaluate on test set
    predict_start = time.time()
    predictions = trainer.predict(test_dataset)
    predict_time = time.time() - predict_start

    logits = predictions.predictions
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]
    y_pred = np.argmax(logits, axis=1)

    metrics_dict = compute_classification_metrics(test_labels, y_pred, probs)
    f1_ci = bootstrap_ci(np.array(test_labels), np.array(y_pred), lambda y_t, y_p: f1_score(y_t, y_p, average="macro"), seed=seed)

    n_infer = min(1000, len(test_dataset))
    subset = TextClassificationDataset(
        {k: v[:n_infer] for k, v in test_encodings.items()},
        test_labels[:n_infer],
    )
    infer_start = time.time()
    _ = trainer.predict(subset)
    infer_time = time.time() - infer_start
    inference_time_per_1k = infer_time * (1000 / n_infer)

    preds_df = pd.DataFrame(
        {
            "text": test_texts,
            "gold_label": test_labels,
            "prediction": y_pred,
            "probability": probs,
        }
    )
    preds_df.to_csv(Path(pred_dir) / "predictions_bert.csv", index=False)

    best_dir = ensure_dir(output_dir / "bert_best")
    trainer.model.save_pretrained(best_dir)
    tokenizer.save_pretrained(best_dir)

    result = ModelResult(
        name="bert_base_uncased",
        y_true=np.array(test_labels),
        y_pred=np.array(y_pred),
        y_prob=probs,
        metrics=metrics_dict,
        f1_ci=f1_ci,
        train_time=train_time,
        inference_time_per_1k=inference_time_per_1k,
        params=model.num_parameters(),
        notes="bert-base-uncased",
    )
    return [result]


__all__ = ["train_transformer_model"]
