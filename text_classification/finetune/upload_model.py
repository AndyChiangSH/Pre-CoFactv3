from huggingface_hub import login
from transformers import AutoModelForSequenceClassification, AutoTokenizer

login()

model_name = "./text_classification/finetune/model/deberta-v3-large_text_11"
repo_name = "Pre-CoFactv3-Text-Classification"

model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.push_to_hub(repo_name)
model.push_to_hub(repo_name)
