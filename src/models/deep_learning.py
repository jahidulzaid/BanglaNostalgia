from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from gensim.models import KeyedVectors
from tensorflow.keras import Model, layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.metrics import f1_score

from ..evaluation import ModelResult, bootstrap_ci, compute_classification_metrics
from ..utils import ensure_dir


class BinaryF1(tf.keras.metrics.Metric):
    """Custom binary F1 score for Keras."""

    def __init__(self, name="f1", **kwargs):
        super().__init__(name=name, **kwargs)
        self.tp = self.add_weight(name="tp", initializer="zeros")
        self.fp = self.add_weight(name="fp", initializer="zeros")
        self.fn = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, (-1,)), tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred, (-1,)) > 0.5, tf.float32)

        tp = tf.reduce_sum(y_true * y_pred)
        fp = tf.reduce_sum((1 - y_true) * y_pred)
        fn = tf.reduce_sum(y_true * (1 - y_pred))

        self.tp.assign_add(tp)
        self.fp.assign_add(fp)
        self.fn.assign_add(fn)

    def result(self):
        precision = self.tp / (self.tp + self.fp + 1e-8)
        recall = self.tp / (self.tp + self.fn + 1e-8)
        return 2 * precision * recall / (precision + recall + 1e-8)

    def reset_states(self):
        for var in self.variables:
            var.assign(0.0)


class BahdanauAttention(layers.Layer):
    def __init__(self, units=64):
        super().__init__()
        self.W1 = layers.Dense(units)
        self.W2 = layers.Dense(units)
        self.V = layers.Dense(1)

    def call(self, query, values):
        # query: (batch, hidden), values: (batch, time, hidden)
        query_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(query_with_time_axis)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector


def _load_embeddings(tokenizer: Tokenizer, embedding_path: Path, embedding_dim: int, vocab_size: int, logger=None) -> tuple[np.ndarray, bool]:
    """Load FastText embeddings if available; otherwise return random matrix."""
    matrix = np.random.uniform(-0.05, 0.05, (vocab_size, embedding_dim)).astype(np.float32)
    used_pretrained = False
    if embedding_path.exists():
        if logger:
            logger.info(f"Loading pretrained embeddings from {embedding_path}")
        vectors = KeyedVectors.load_word2vec_format(embedding_path, binary=False)
        word_index = tokenizer.word_index
        for word, idx in word_index.items():
            if idx >= vocab_size:
                continue
            if word in vectors:
                matrix[idx] = vectors[word]
                used_pretrained = True
    else:
        if logger:
            logger.warning(f"Embedding file not found at {embedding_path}, using random initialization.")
    return matrix, used_pretrained


def _build_model(
    architecture: str,
    vocab_size: int,
    embedding_dim: int,
    max_len: int,
    embedding_matrix: Optional[np.ndarray],
    dropout: float = 0.5,
    trainable_embeddings: bool = True,
) -> Model:
    inputs = layers.Input(shape=(max_len,), dtype="int32")
    embedding_layer = layers.Embedding(
        vocab_size,
        embedding_dim,
        weights=[embedding_matrix] if embedding_matrix is not None else None,
        trainable=trainable_embeddings,
    )
    x = embedding_layer(inputs)

    if architecture == "cnn":
        x = layers.Conv1D(128, 5, activation="relu")(x)
        x = layers.GlobalMaxPooling1D()(x)
        x = layers.Dense(64, activation="relu")(x)
    elif architecture == "lstm":
        x = layers.LSTM(128, return_sequences=False)(x)
        x = layers.Dense(64, activation="relu")(x)
    elif architecture == "bilstm":
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=False))(x)
        x = layers.Dense(64, activation="relu")(x)
    elif architecture == "attn_bilstm":
        lstm_out = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
        query = layers.Lambda(lambda t: t[:, -1, :])(lstm_out)
        context = BahdanauAttention(64)(query, lstm_out)
        x = layers.Dense(64, activation="relu")(context)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = Model(inputs, outputs, name=architecture)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[BinaryF1(name="f1"), "accuracy"],
    )
    return model


def _prepare_sequences(
    train_texts: List[str],
    val_texts: List[str],
    test_texts: List[str],
    vocab_size: int,
    max_len: int,
) -> tuple[Tokenizer, np.ndarray, np.ndarray, np.ndarray]:
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="[OOV]")
    tokenizer.fit_on_texts(train_texts + val_texts)
    train_seq = pad_sequences(tokenizer.texts_to_sequences(train_texts), maxlen=max_len, padding="post", truncating="post")
    val_seq = pad_sequences(tokenizer.texts_to_sequences(val_texts), maxlen=max_len, padding="post", truncating="post")
    test_seq = pad_sequences(tokenizer.texts_to_sequences(test_texts), maxlen=max_len, padding="post", truncating="post")
    return tokenizer, train_seq, val_seq, test_seq


def train_deep_models(
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
    """Train CNN/LSTM/BiLSTM/Attention-BiLSTM models."""
    ensure_dir(output_dir)
    ensure_dir(figures_dir)
    pred_dir = ensure_dir(predictions_dir) if predictions_dir else ensure_dir(output_dir)

    max_len = int(config.get("max_len", 200))
    vocab_size = int(config.get("vocab_size", 50000))
    embedding_dim = int(config.get("embedding_dim", 300))
    batch_size = int(config.get("batch_size", 32))
    epochs = int(config.get("epochs", 20))
    patience = int(config.get("patience", 3))
    embedding_path = Path(config.get("embedding_path", "data/wiki.bn.vec"))
    dropout = float(config.get("dropout", 0.5))

    tokenizer, train_seq, val_seq, test_seq = _prepare_sequences(
        train_texts, val_texts, test_texts, vocab_size=vocab_size, max_len=max_len
    )

    embedding_matrix, used_pretrained = _load_embeddings(tokenizer, embedding_path, embedding_dim, vocab_size, logger=logger)
    trainable_embeddings = True

    results: List[ModelResult] = []
    architectures = ["cnn", "lstm", "bilstm", "attn_bilstm"]

    for arch in architectures:
        if logger:
            logger.info(f"Training deep model: {arch}")

        model = _build_model(
            architecture=arch,
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            max_len=max_len,
            embedding_matrix=embedding_matrix,
            dropout=dropout,
            trainable_embeddings=trainable_embeddings,
        )

        callbacks = [
            EarlyStopping(monitor="val_f1", mode="max", patience=patience, restore_best_weights=True, verbose=1),
            ModelCheckpoint(
                filepath=Path(output_dir) / f"{arch}_best.keras",
                monitor="val_f1",
                mode="max",
                save_best_only=True,
                save_weights_only=False,
                verbose=1,
            ),
        ]

        t_start = time.time()
        history = model.fit(
            train_seq,
            np.array(train_labels),
            validation_data=(val_seq, np.array(val_labels)),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=2,
        )
        train_time = time.time() - t_start

        probs = model.predict(test_seq, batch_size=batch_size).flatten()
        y_pred = (probs >= 0.5).astype(int)

        metrics_dict = compute_classification_metrics(test_labels, y_pred, probs)
        f1_ci = bootstrap_ci(
            np.array(test_labels), np.array(y_pred), lambda y_t, y_p: f1_score(y_t, y_p, average="macro"), seed=seed
        )

        n_infer = min(1000, len(test_seq))
        infer_start = time.time()
        _ = model.predict(test_seq[:n_infer], batch_size=batch_size)
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
        preds_df.to_csv(Path(pred_dir) / f"predictions_{arch}.csv", index=False)

        results.append(
            ModelResult(
                name=arch,
                y_true=np.array(test_labels),
                y_pred=np.array(y_pred),
                y_prob=probs,
                metrics=metrics_dict,
                f1_ci=f1_ci,
                train_time=train_time,
                inference_time_per_1k=inference_time_per_1k,
                params=model.count_params(),
                notes="pretrained_embeddings" if used_pretrained else "random_embeddings",
            )
        )

    return results


__all__ = ["train_deep_models"]
