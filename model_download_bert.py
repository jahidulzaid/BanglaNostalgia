from transformers import BertTokenizer, BertModel

model_name = "google-bert/bert-base-multilingual-cased"

# Download tokenizer
tokenizer = BertTokenizer.from_pretrained(model_name)

# Download model
model = BertModel.from_pretrained(model_name)

print("Model and tokenizer downloaded successfully!")
