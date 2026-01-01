#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bangla Nostalgia Detection (Binary) — Research Baseline Comparison
Single-file experiment runner with a CONFIG section.

Dataset schema expected (your file matches this):
- id, text, clean_text, label, reference_time

Labels:
- label: 0 = not nostalgic, 1 = nostalgic

Baselines included:
1) Majority class
2) Keyword-rule baseline (nostalgia lexicon)
3) TF-IDF (word) + Logistic Regression
4) TF-IDF (char) + Logistic Regression
5) TF-IDF (word) + Linear SVM
6) TF-IDF (char) + Linear SVM
7) TF-IDF (word) + Multinomial NB
8) TF-IDF (char) + Multinomial NB

Optional (configurable):
- A HuggingFace Transformer fine-tuning baseline (BanglaBERT/BanglishBERT/MuRIL/XLM-R)

Outputs:
- results CSV with mean±std over folds
- per-fold predictions CSV for significance/error analysis
"""

# =========================
# CONFIG
# =========================

CONFIG = {
    # Data
    "DATA_PATH": "data/bengali_nostalgia_labeled.csv",
    "TEXT_COL_PRIORITY": ["clean_text", "text"],  # will pick the first existing
    "LABEL_COL": "label",
    "ID_COL": "id",

    # Validation strategy
    "SPLIT_METHOD": "stratified_kfold",  # "stratified_kfold" or "train_test"
    "N_SPLITS": 5,                        # used for stratified_kfold
    "TEST_SIZE": 0.2,                     # used for train_test
    "RANDOM_SEED": 42,

    # Preprocessing
    "USE_BN_NORMALIZER": True,            # uses csebuetnlp-normalizer if installed
    "LOWERCASE": False,                   # for Bangla script usually keep False
    "REMOVE_URLS": True,
    "KEEP_EMOJIS": True,                  # True => do not strip emojis
    "MIN_TEXT_LEN": 1,                    # filter out empty/very short if needed

    # Vectorizer params
    "WORD_TFIDF": {
        "analyzer": "word",
        "ngram_range": (1, 2),
        "min_df": 2,
        "max_df": 0.95,
        "sublinear_tf": True,
    },
    "CHAR_TFIDF": {
        "analyzer": "char",
        "ngram_range": (3, 5),
        "min_df": 2,
        "max_df": 0.95,
        "sublinear_tf": True,
    },

    # Keyword baseline
    # (You should expand this list as you review errors; keep it in the paper appendix)
    "NOSTALGIA_KEYWORDS_BN": [
        "মনে পড়ে", "মনে পড়ে", "মনে পড়ে", "মনে পড়ে",
        "মনে আছে", "স্মৃতি", "স্মৃতির", "পুরনো দিন", "পুরানো দিন",
        "সেই দিন", "সেই সময়", "সেই সময়", "একসময়", "এক সময়",
        "ছোটবেলা", "শৈশব", "স্কুল জীবন", "কলেজ জীবন",
        "মিস করি", "খুব মিস", "ফিরে যেতে চাই", "আবার যদি",
        "কি দিন ছিল", "কি দিন ছিলো", "আহা", "ইশ",
    ],
    # If any keyword matched => predict nostalgic
    "KEYWORD_BASELINE_THRESHOLD": 1,

    # Models to run
    "RUN_BASELINES": True,
    "RUN_TRANSFORMER": False,  # set True if you want transformer baseline too

    # Transformer baseline config (only used if RUN_TRANSFORMER=True)
    "TRANSFORMER": {
        "MODEL_NAME": "csebuetnlp/banglabert",   # try csebuetnlp/banglishbert or google/muril-base-cased
        "MAX_LEN": 128,
        "EPOCHS": 3,
        "LR": 2e-5,
        "BATCH_TRAIN": 16,
        "BATCH_EVAL": 32,
        "WEIGHT_DECAY": 0.01,
        "USE_CLASS_WEIGHTS": True,
    },

    # Output
    "OUT_DIR": "./nostalgia_experiments",
    "RESULTS_CSV": "results_summary.csv",
    "PREDICTIONS_CSV": "fold_predictions.csv",
}

# =========================
# IMPORTS
# =========================

import os
import re
import json
import math
import time
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB

warnings.filterwarnings("ignore")


# =========================
# UTILITIES
# =========================

URL_RE = re.compile(r"https?://\S+|www\.\S+")
WS_RE = re.compile(r"\s+")

def ensure_out_dir(path: str):
    os.makedirs(path, exist_ok=True)

def try_bn_normalize(text: str) -> str:
    """
    Optional: uses csebuetnlp-normalizer if installed
    pip install csebuetnlp-normalizer
    """
    if not CONFIG["USE_BN_NORMALIZER"]:
        return text
    try:
        from normalizer import normalize
        return normalize(text)
    except Exception:
        return text

def preprocess_text(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.strip()

    if CONFIG["REMOVE_URLS"]:
        s = URL_RE.sub(" ", s)

    # Keep emojis by default. If you want to remove non-Bangla chars, do it here.
    if not CONFIG["KEEP_EMOJIS"]:
        # Remove basic emoji ranges (not perfect)
        s = re.sub(r"[\U00010000-\U0010ffff]", " ", s)

    s = try_bn_normalize(s)

    if CONFIG["LOWERCASE"]:
        s = s.lower()

    s = WS_RE.sub(" ", s).strip()
    return s

def pick_text_column(df: pd.DataFrame) -> str:
    for c in CONFIG["TEXT_COL_PRIORITY"]:
        if c in df.columns:
            return c
    raise ValueError(f"No text column found. Looked for: {CONFIG['TEXT_COL_PRIORITY']}")

def compute_metrics(y_true, y_pred, y_prob=None):
    """Binary metrics; y_prob is probability for positive class (optional)."""
    metrics = {}
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro")
    metrics["f1_pos"] = f1_score(y_true, y_pred, pos_label=1)
    metrics["precision_pos"] = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    metrics["recall_pos"] = recall_score(y_true, y_pred, pos_label=1, zero_division=0)

    if y_prob is not None:
        # roc_auc requires both classes present
        if len(np.unique(y_true)) == 2:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        else:
            metrics["roc_auc"] = np.nan
    else:
        metrics["roc_auc"] = np.nan

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    metrics["tn"] = int(tn); metrics["fp"] = int(fp); metrics["fn"] = int(fn); metrics["tp"] = int(tp)
    return metrics

def summarize_results(rows):
    """
    rows: list[dict] each includes 'model_name', 'fold', and metric keys
    returns summary df with mean/std per model
    """
    df = pd.DataFrame(rows)
    metric_cols = [c for c in df.columns if c not in ("model_name", "fold")]
    agg = df.groupby("model_name")[metric_cols].agg(["mean", "std"]).reset_index()

    # flatten columns
    agg.columns = ["model_name"] + [f"{m}_{stat}" for (m, stat) in agg.columns[1:]]
    return agg, df


# =========================
# KEYWORD BASELINE
# =========================

class KeywordRuleClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, keywords, threshold=1):
        self.keywords = keywords
        self.threshold = threshold

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        preds = []
        for t in X:
            t = "" if t is None else str(t)
            count = 0
            for kw in self.keywords:
                if kw in t:
                    count += 1
                    if count >= self.threshold:
                        break
            preds.append(1 if count >= self.threshold else 0)
        return np.array(preds, dtype=int)

    def predict_proba(self, X):
        # crude probabilities: 0.8 if keyword hit else 0.2
        yhat = self.predict(X)
        p1 = np.where(yhat == 1, 0.8, 0.2)
        p0 = 1.0 - p1
        return np.vstack([p0, p1]).T


# =========================
# MODEL FACTORY
# =========================

def build_models():
    models = {}

    # 1) Majority class
    models["majority_dummy"] = DummyClassifier(strategy="most_frequent")

    # 2) Keyword rule
    models["keyword_rule"] = KeywordRuleClassifier(
        keywords=CONFIG["NOSTALGIA_KEYWORDS_BN"],
        threshold=CONFIG["KEYWORD_BASELINE_THRESHOLD"]
    )

    # 3-4) TF-IDF + Logistic Regression
    models["tfidf_word_lr"] = Pipeline([
        ("tfidf", TfidfVectorizer(**CONFIG["WORD_TFIDF"])),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced")),
    ])
    models["tfidf_char_lr"] = Pipeline([
        ("tfidf", TfidfVectorizer(**CONFIG["CHAR_TFIDF"])),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced")),
    ])

    # 5-6) TF-IDF + Linear SVM
    # LinearSVC doesn't give probabilities; we wrap with calibration for AUC
    models["tfidf_word_svm"] = Pipeline([
        ("tfidf", TfidfVectorizer(**CONFIG["WORD_TFIDF"])),
        ("clf", CalibratedClassifierCV(LinearSVC(class_weight="balanced"), method="sigmoid", cv=3)),
    ])
    models["tfidf_char_svm"] = Pipeline([
        ("tfidf", TfidfVectorizer(**CONFIG["CHAR_TFIDF"])),
        ("clf", CalibratedClassifierCV(LinearSVC(class_weight="balanced"), method="sigmoid", cv=3)),
    ])

    # 7-8) TF-IDF + MultinomialNB (works best with non-negative tf-idf; OK here)
    models["tfidf_word_mnb"] = Pipeline([
        ("tfidf", TfidfVectorizer(**CONFIG["WORD_TFIDF"])),
        ("clf", MultinomialNB()),
    ])
    models["tfidf_char_mnb"] = Pipeline([
        ("tfidf", TfidfVectorizer(**CONFIG["CHAR_TFIDF"])),
        ("clf", MultinomialNB()),
    ])

    return models


# =========================
# TRANSFORMER (OPTIONAL)
# =========================

def run_transformer_cv(X, y, ids, out_dir):
    """
    Minimal transformer fine-tuning baseline using HuggingFace Transformers.
    Runs the same CV splits as the rest of the pipeline.

    Requirements:
      pip install transformers datasets evaluate accelerate torch

    Note:
      This is intentionally compact. For a paper, you may want to:
      - log seeds, hardware, train time
      - try multiple models (BanglaBERT/BanglishBERT/MuRIL/XLM-R)
      - run more epochs, hyperparam sweeps, early stopping
    """
    from datasets import Dataset
    import torch
    from torch import nn
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
        set_seed,
    )
    from sklearn.utils.class_weight import compute_class_weight

    tcfg = CONFIG["TRANSFORMER"]
    set_seed(CONFIG["RANDOM_SEED"])

    tokenizer = AutoTokenizer.from_pretrained(tcfg["MODEL_NAME"])

    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=tcfg["MAX_LEN"]
        )

    skf = StratifiedKFold(n_splits=CONFIG["N_SPLITS"], shuffle=True, random_state=CONFIG["RANDOM_SEED"])
    all_rows = []
    pred_rows = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        id_te = ids[test_idx]

        ds_train = Dataset.from_dict({"text": X_tr.tolist(), "labels": y_tr.tolist()}).map(tokenize_batch, batched=True)
        ds_test  = Dataset.from_dict({"text": X_te.tolist(), "labels": y_te.tolist()}).map(tokenize_batch, batched=True)

        ds_train.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        ds_test.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        model = AutoModelForSequenceClassification.from_pretrained(tcfg["MODEL_NAME"], num_labels=2)

        class_weights = None
        if tcfg.get("USE_CLASS_WEIGHTS", True):
            cw = compute_class_weight(class_weight="balanced", classes=np.array([0,1]), y=y_tr)
            class_weights = torch.tensor(cw, dtype=torch.float)

        class WeightedTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False):
                labels = inputs.get("labels")
                outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
                logits = outputs.logits
                if class_weights is not None:
                    loss_fct = nn.CrossEntropyLoss(weight=class_weights.to(logits.device))
                else:
                    loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, 2), labels.view(-1))
                return (loss, outputs) if return_outputs else loss

        fold_dir = os.path.join(out_dir, f"transformer_fold{fold}")
        ensure_out_dir(fold_dir)

        args = TrainingArguments(
            output_dir=fold_dir,
            learning_rate=tcfg["LR"],
            per_device_train_batch_size=tcfg["BATCH_TRAIN"],
            per_device_eval_batch_size=tcfg["BATCH_EVAL"],
            num_train_epochs=tcfg["EPOCHS"],
            weight_decay=tcfg["WEIGHT_DECAY"],
            eval_strategy="no",
            save_strategy="no",
            logging_steps=50,
            seed=CONFIG["RANDOM_SEED"],
            fp16=torch.cuda.is_available(),
            report_to=[],
        )

        trainer = WeightedTrainer(
            model=model,
            args=args,
            train_dataset=ds_train,
            eval_dataset=ds_test,
            tokenizer=tokenizer,
        )

        trainer.train()

        # Predict
        preds = trainer.predict(ds_test)
        logits = preds.predictions
        probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
        y_pred = (probs >= 0.5).astype(int)

        m = compute_metrics(y_te, y_pred, probs)
        m["model_name"] = "transformer"
        m["fold"] = fold
        all_rows.append(m)

        for i in range(len(test_idx)):
            pred_rows.append({
                "model_name": "transformer",
                "fold": fold,
                "id": id_te[i],
                "y_true": int(y_te[i]),
                "y_pred": int(y_pred[i]),
                "y_prob": float(probs[i]),
                "text": X_te[i],
            })

    return all_rows, pred_rows


# =========================
# RUNNER
# =========================

def main():
    ensure_out_dir(CONFIG["OUT_DIR"])

    # ---- Load ----
    df = pd.read_csv(CONFIG["DATA_PATH"])
    text_col = pick_text_column(df)
    label_col = CONFIG["LABEL_COL"]
    id_col = CONFIG["ID_COL"] if CONFIG["ID_COL"] in df.columns else None

    # Basic checks
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in CSV.")
    if not set(df[label_col].unique()).issubset({0, 1}):
        raise ValueError(f"Labels must be 0/1. Found: {sorted(df[label_col].unique().tolist())}")

    # ---- Preprocess ----
    df["__text__"] = df[text_col].astype(str).apply(preprocess_text)
    df = df[df["__text__"].str.len() >= CONFIG["MIN_TEXT_LEN"]].copy()

    X = df["__text__"].values
    y = df[label_col].astype(int).values
    ids = df[id_col].astype(str).values if id_col else np.array([str(i) for i in range(len(df))])

    # ---- Save config used (for reproducibility) ----
    with open(os.path.join(CONFIG["OUT_DIR"], "config_used.json"), "w", encoding="utf-8") as f:
        json.dump(CONFIG, f, ensure_ascii=False, indent=2)

    # ---- Prepare splits ----
    rows = []
    pred_rows = []

    if CONFIG["SPLIT_METHOD"] == "train_test":
        X_tr, X_te, y_tr, y_te, id_tr, id_te = train_test_split(
            X, y, ids,
            test_size=CONFIG["TEST_SIZE"],
            random_state=CONFIG["RANDOM_SEED"],
            stratify=y
        )
        splits = [(1, (np.arange(len(X_tr)), np.arange(len(X_te))), (X_tr, X_te, y_tr, y_te, id_te))]
    else:
        skf = StratifiedKFold(n_splits=CONFIG["N_SPLITS"], shuffle=True, random_state=CONFIG["RANDOM_SEED"])
        splits = []
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            id_te = ids[test_idx]
            splits.append((fold, (train_idx, test_idx), (X_tr, X_te, y_tr, y_te, id_te)))

    # ---- Run baselines ----
    if CONFIG["RUN_BASELINES"]:
        models = build_models()

        for model_name, clf in models.items():
            for fold, (_, _), (X_tr, X_te, y_tr, y_te, id_te) in splits:
                t0 = time.time()
                clf.fit(X_tr, y_tr)
                y_pred = clf.predict(X_te)

                # probability if available
                y_prob = None
                if hasattr(clf, "predict_proba"):
                    try:
                        y_prob = clf.predict_proba(X_te)[:, 1]
                    except Exception:
                        y_prob = None

                m = compute_metrics(y_te, y_pred, y_prob)
                m["model_name"] = model_name
                m["fold"] = fold
                m["train_time_sec"] = round(time.time() - t0, 4)
                rows.append(m)

                # store fold predictions
                for i in range(len(X_te)):
                    pred_rows.append({
                        "model_name": model_name,
                        "fold": fold,
                        "id": id_te[i],
                        "y_true": int(y_te[i]),
                        "y_pred": int(y_pred[i]),
                        "y_prob": float(y_prob[i]) if y_prob is not None else np.nan,
                        "text": X_te[i],
                    })

    # ---- Transformer baseline (optional) ----
    if CONFIG["RUN_TRANSFORMER"]:
        t_rows, t_pred_rows = run_transformer_cv(X, y, ids, CONFIG["OUT_DIR"])
        rows.extend(t_rows)
        pred_rows.extend(t_pred_rows)

    # ---- Save outputs ----
    summary_df, per_fold_df = summarize_results(rows)

    out_summary = os.path.join(CONFIG["OUT_DIR"], CONFIG["RESULTS_CSV"])
    out_preds = os.path.join(CONFIG["OUT_DIR"], CONFIG["PREDICTIONS_CSV"])

    summary_df.to_csv(out_summary, index=False, encoding="utf-8-sig")
    pd.DataFrame(pred_rows).to_csv(out_preds, index=False, encoding="utf-8-sig")

    # Print top models by positive F1 mean (paper-friendly)
    sort_col = "f1_pos_mean" if "f1_pos_mean" in summary_df.columns else summary_df.columns[1]
    print("\n=== SUMMARY (sorted) ===")
    print(summary_df.sort_values(sort_col, ascending=False).to_string(index=False))

    print(f"\n[OK] Saved:\n- {out_summary}\n- {out_preds}\n- {os.path.join(CONFIG['OUT_DIR'], 'config_used.json')}")


if __name__ == "__main__":
    main()
