# file: classify_zero_shot.py
import pandas as pd
from transformers import pipeline

clf = pipeline(
    "zero-shot-classification",
    model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
    device=0,  # change to 0 for GPU
)

df = pd.read_csv("data/bengali_nostalgia_dataset.csv")
labels = ["nostalgic", "not nostalgic"]

def predict(text: str) -> int:
    out = clf(text, labels, hypothesis_template="This text is {}.")
    return 1 if out["labels"][0] == "nostalgic" else 0  # 1 = nostalgic, 0 = not

df["label"] = df["clean_text"].fillna(df["text"]).apply(predict)
df.to_csv("data/bengali_nostalgia_labeled.csv", index=True)
print("Wrote data/bengali_nostalgia_labeled.csv")
