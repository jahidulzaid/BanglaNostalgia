from transformers import XLMRobertaTokenizer, XLMRobertaModel

model_name = "FacebookAI/xlm-roberta-base"

# Download tokenizer
tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)

# Download model
model = XLMRobertaModel.from_pretrained(model_name)

print("XLM-RoBERTa model and tokenizer downloaded successfully!")
