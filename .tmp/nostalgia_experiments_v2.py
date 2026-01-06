import logging
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("run")


df = pd.read_csv("data/bengali_nostalgia_labeled.csv")

X = df["text"].astype(str).values
y = df["label"].values

logger.info(f"Total samples used: {len(X)}")


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
splits = list(skf.split(X, y))



models = {
    "tfidf_word_lr": Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            max_features=50000
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            n_jobs=-1
        ))
    ]),

    "tfidf_char_svm": Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            max_features=50000
        )),
        ("clf", LinearSVC())
    ]),

    "tfidf_word_rf": Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            max_features=50000
        )),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            n_jobs=-1,
            random_state=42
        ))
    ])
}



results = []

total_steps = len(models) * len(splits)
pbar = tqdm(total=total_steps, desc="Running classical ML", leave=True)

for model_name, pipeline in models.items():
    for fold, (train_idx, test_idx) in enumerate(splits, start=1):

        logger.info(f"Running {model_name} (fold {fold}/5)")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro")
        weighted_f1 = f1_score(y_test, y_pred, average="weighted")

        results.append({
            "model": model_name,
            "fold": fold,
            "accuracy": acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1
        })

        pbar.update(1)

pbar.close()



results_df = pd.DataFrame(results)
results_df.to_csv("results/classical_ml_results.csv", index=False)

logger.info("Finished all classical ML experiments.")



